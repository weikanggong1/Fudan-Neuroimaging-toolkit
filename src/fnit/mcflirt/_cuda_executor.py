"""Reuse MCFLIRT sampling buffers and replay its unchanged NCC reduction."""

import torch

from ._cost_cuda import FusedMotionSampler
from ._cuda_graph import cuda_graph_capture_enabled


class _DeviceMotionSampler(FusedMotionSampler):
    def prepare(self, coefficients_np_float32):
        # Triton launches on the current device; tensor arguments alone do
        # not select another device as an ATen operation would.
        with torch.cuda.device(self.moving.device):
            return super().prepare(coefficients_np_float32)


class CudaMotionCostExecutor:
    """Keep one sampling workspace and reduction graph for a reference grid.

    The caller supplies the existing compiled ``_normcorr_reduce``. Capturing
    that function preserves its row/plane accumulation and running counts;
    this class does not introduce a different reduction kernel. Each MCFLIRT
    run owns one executor per reference scale. The returned scalar and sampler
    buffers are overwritten by the next evaluation.
    """

    def __init__(self, reference_zyx, moving_contiguous, moving_sizes, reducer):
        self.moving = torch.empty_like(moving_contiguous)
        self.sampler = _DeviceMotionSampler(reference_zyx, self.moving, moving_sizes)
        self.compiled_reducer = reducer
        self.graph = None
        self.output = None
        self.capture_stream = None
        self.capture_enabled = cuda_graph_capture_enabled()
        self.set_moving(moving_contiguous)

    def set_moving(self, moving_contiguous):
        """Copy a frame without changing the addresses captured by the graph."""
        if (moving_contiguous.shape != self.moving.shape or
                moving_contiguous.dtype != self.moving.dtype or
                moving_contiguous.device != self.moving.device or
                not moving_contiguous.is_contiguous()):
            raise ValueError("motion executor requires the same contiguous moving grid")
        self.moving.copy_(moving_contiguous)

    def _capture(self, inputs):
        # Inductor compilation and any lazy initialization must happen before
        # capture. Warm the same reduction on a side stream; all three input
        # addresses remain fixed throughout this executor's lifetime.
        device = self.moving.device
        with torch.cuda.device(device):
            current_stream = torch.cuda.current_stream(device)
            self.capture_stream = torch.cuda.Stream(device=device)
            self.capture_stream.wait_stream(current_stream)
            with torch.cuda.stream(self.capture_stream):
                for _ in range(3):
                    self.compiled_reducer(*inputs)
            current_stream.wait_stream(self.capture_stream)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=self.capture_stream):
                self.output = self.compiled_reducer(*inputs)
            current_stream.wait_stream(self.capture_stream)

    def reduce(self, reference_values, moving_values, weights):
        """Replay the original compiled reducer on the sampler's three buffers."""
        inputs = (reference_values, moving_values, weights)
        expected = (self.sampler.reference_values, self.sampler.moving_values,
                    self.sampler.weights)
        if any(value is not buffer for value, buffer in zip(inputs, expected)):
            raise ValueError("motion executor reduction requires its sampler buffers")
        with torch.cuda.device(self.moving.device):
            if not self.capture_enabled:
                self.output = self.compiled_reducer(*inputs)
                return self.output
            if self.graph is None:
                self._capture(inputs)
            self.graph.replay()
        return self.output
