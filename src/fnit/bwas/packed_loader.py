"""Reusable pinned and CUDA buffers for packed BWAS voxel tiles."""

from time import perf_counter

import numpy as np
import torch


class _Slot:
    def __init__(self):
        self.host_left = self.gpu_left = None
        self.ready = self.done = None


class PackedTileLoader:
    def __init__(self, shards, device: torch.device, async_h2d: bool = True,
                 measure: bool = False, cache_row: bool = False):
        if device.type != "cuda":
            raise ValueError("packed GPU loader requires CUDA")
        self.shards = shards
        self.device = device
        self.async_h2d = async_h2d
        self.measure = measure
        self.cache_row = cache_row
        self.row_index = None
        self.row_buffer = None
        self.row_views = []
        self.copy_stream = torch.cuda.Stream(device=device) if async_h2d else None
        self.slots = [_Slot(), _Slot()]
        self.cache_read_seconds = 0.0
        self.h2d_events = []
        self.correlation_events = []
        self.fisher_events = []

    def _view(self, slot: _Slot, name: str, shape: tuple[int, ...], *, host: bool):
        needed = int(np.prod(shape))
        buffer = getattr(slot, name, None)
        if buffer is None or buffer.numel() < needed:
            buffer = (torch.empty(needed, dtype=torch.float32, pin_memory=True)
                      if host else torch.empty(needed, dtype=torch.float32,
                                            device=self.device))
            setattr(slot, name, buffer)
        return buffer[:needed].view(shape)

    def _load_row(self, i: int, n_i: int):
        shapes = [(len(lengths), n_i, int(lengths.max())) for _, lengths in self.shards]
        needed = sum(int(np.prod(shape)) for shape in shapes)
        if self.row_buffer is None or self.row_buffer.numel() < needed:
            self.row_buffer = torch.empty(needed, dtype=torch.float32, device=self.device)
        self.row_views = []
        offset = 0
        for group_index, ((packed, _), shape) in enumerate(zip(self.shards, shapes)):
            slot = self.slots[group_index % 2]
            if slot.done is not None:
                slot.done.synchronize()
            host = self._view(slot, "host_left", shape, host=True)
            begin = perf_counter()
            np.copyto(host.numpy(), packed[i:i+n_i].transpose(1, 0, 2))
            self.cache_read_seconds += perf_counter()-begin
            size = int(np.prod(shape))
            view = self.row_buffer[offset:offset+size].view(shape)
            if self.measure:
                first = torch.cuda.Event(enable_timing=True)
                first.record()
            view.copy_(host, non_blocking=True)
            slot.done = torch.cuda.Event(enable_timing=self.measure)
            slot.done.record()
            if self.measure:
                self.h2d_events.append((first, slot.done))
            self.row_views.append(view)
            offset += size
        torch.cuda.current_stream(self.device).synchronize()
        self.row_index = i

    def fisher_blocks(self, i: int, j: int, n_i: int, n_j: int):
        for _, start, stop, values in self.fisher_tiles(i, [(j, n_j)], n_i):
            yield start, stop, values
            del values

    def fisher_tiles(self, i: int, columns: list[tuple[int, int]], n_i: int):
        """Reuse each subject group's row block across the supplied column tiles."""
        main_stream = torch.cuda.current_stream(self.device)
        if self.cache_row and self.row_index != i:
            self._load_row(i, n_i)
        def prepare(group_index):
            packed, lengths = self.shards[group_index]
            count, time = len(lengths), int(lengths.max())
            slot = self.slots[group_index % 2]
            if slot.done is not None:
                slot.done.synchronize()
            shape_left = (count, n_i, time)
            begin = perf_counter()
            if not self.cache_row:
                host_left = self._view(slot, "host_left", shape_left, host=True)
                np.copyto(host_left.numpy(), packed[i:i+n_i].transpose(1, 0, 2))
            host_right = []
            for column, (j, n_j) in enumerate(columns):
                right = self._view(slot, f"host_right_{column}",
                                   (count, n_j, time), host=True)
                np.copyto(right.numpy(), packed[j:j+n_j].transpose(1, 0, 2))
                host_right.append(right)
            self.cache_read_seconds += perf_counter()-begin
            stream = self.copy_stream if self.async_h2d else main_stream
            with torch.cuda.stream(stream):
                gpu_left = (self.row_views[group_index] if self.cache_row else
                            self._view(slot, "gpu_left", shape_left, host=False))
                gpu_right = [self._view(slot, f"gpu_right_{column}", right.shape,
                                        host=False)
                             for column, right in enumerate(host_right)]
                if self.measure:
                    copy_start = torch.cuda.Event(enable_timing=True)
                    copy_start.record(stream)
                if not self.cache_row:
                    gpu_left.copy_(host_left, non_blocking=True)
                for left, right in zip(gpu_right, host_right):
                    left.copy_(right, non_blocking=True)
                if self.async_h2d:
                    gpu_left.record_stream(stream)
                    for right in gpu_right:
                        right.record_stream(stream)
                slot.ready = torch.cuda.Event(enable_timing=self.measure)
                slot.ready.record(stream)
                if self.measure:
                    self.h2d_events.append((copy_start, slot.ready))
            return slot, lengths, gpu_left, gpu_right

        if not self.shards:
            return
        prepared = prepare(0)
        start_subject = 0
        for group_index in range(len(self.shards)):
            slot, lengths, gpu_left, gpu_right = prepared
            count = len(lengths)
            main_stream.wait_event(slot.ready)
            if self.async_h2d:
                gpu_left.record_stream(main_stream)
                for right in gpu_right:
                    right.record_stream(main_stream)
            for column, right in enumerate(gpu_right):
                if self.measure:
                    correlation_start = torch.cuda.Event(enable_timing=True)
                    correlation_start.record(main_stream)
                correlation = torch.bmm(gpu_left, right.transpose(1, 2))
                correlation.div_(torch.as_tensor(lengths, device=self.device,
                                                 dtype=torch.float32)[:, None, None])
                correlation.masked_fill_(correlation > 0.9999, 0).clamp_(
                    -0.999999, 0.999999)
                if self.measure:
                    correlation_end = torch.cuda.Event(enable_timing=True)
                    correlation_end.record(main_stream)
                    self.correlation_events.append((correlation_start, correlation_end))
                values = correlation.atanh_().reshape(count, -1)
                if self.measure:
                    fisher_end = torch.cuda.Event(enable_timing=True)
                    fisher_end.record(main_stream)
                    self.fisher_events.append((correlation_end, fisher_end))
                # CPU preparation and H2D of the next group overlap GPU work here.
                if column == len(columns)-1 and group_index+1 < len(self.shards):
                    prepared = prepare(group_index+1)
                yield column, start_subject, start_subject+count, values
                del values, correlation
            slot.done = torch.cuda.Event()
            slot.done.record(main_stream)
            start_subject += count

    def drain_timings(self):
        torch.cuda.synchronize(self.device)
        result = {
            "cache_read": self.cache_read_seconds,
            "h2d": sum(first.elapsed_time(last) for first, last in self.h2d_events) / 1000,
            "correlation": sum(first.elapsed_time(last)
                               for first, last in self.correlation_events) / 1000,
            "fisher": sum(first.elapsed_time(last)
                          for first, last in self.fisher_events) / 1000,
        }
        self.cache_read_seconds = 0.0
        self.h2d_events.clear()
        self.correlation_events.clear()
        self.fisher_events.clear()
        return result
