"""Batched execution of the existing FSL-compatible affine costs.

Only execution changes here: reference binning, FSL float coefficients,
trilinear arithmetic, edge taper and cost formula come from the prepared
reference cost.  Candidate ordering and search decisions remain with the
caller.  Images are stored once, regardless of the candidate batch size.
"""

import numpy as np
import torch



class BatchedAffineCost:
    """Evaluate affine candidates without per-candidate device synchronization.

    ``reference_cost`` is a prepared ``FSLCorrelationRatio`` or
    ``FSLNormalizedMutualInformation``.  ``evaluate`` accepts input-to-reference
    FSL scaled-mm matrices with shape ``[B, 4, 4]`` and returns device float32
    costs in the same order.  A single ``[4, 4]`` matrix returns one cost.

    NumPy/CPU candidates retain the reference NumPy double inverse followed
    by float32 coefficient assignment.  CUDA candidates use a double inverse;
    singular CUDA lanes receive the corresponding failed-registration cost.
    No half precision or TF32 matrix multiplication is used in either path.
    """

    def __init__(self, reference_cost, *, max_batch_size=128, memory_budget_gb=20):
        self.cost = reference_cost
        self.device = reference_cost.device
        self.max_batch_size = int(max_batch_size)
        self.memory_budget_bytes = int(float(memory_budget_gb) * 1024**3)
        if self.max_batch_size < 1 or self.memory_budget_bytes <= 0:
            raise ValueError("batch size and memory budget must be positive")
        self.nmi = hasattr(reference_cost, "test_factor")
        self.grid = reference_cost.grid
        self.bin_sort_order = reference_cost.bin_sort_order
        self.bin_lengths = reference_cost.bin_lengths
        # CUB selects float4 loads from each segment's address.  Candidate
        # strides must preserve the scalar reference's 16-byte alignment;
        # otherwise odd grid sizes change reduction grouping by lane.
        self._segment_padding = (-self.grid.shape[1]) % 4 if self.device.type == "cuda" else 0
        self._segment_bin_lengths = self.bin_lengths
        if self._segment_padding:
            self._segment_bin_lengths = torch.cat((
                self.bin_lengths, self.bin_lengths.new_tensor([self._segment_padding]),
            ))
        self._segment_lengths = {}
        self.bin_offsets = reference_cost.bin_index * (reference_cost.bins + 1)
        self.moving_sampling_inverse = np.diag(
            [*[1 / value for value in reference_cost.moving_voxel_sizes], 1.0]
        )
        self.reference_sampling = np.diag(
            [*reference_cost.reference_voxel_sizes, 1.0]
        )
        self.moving_sampling_tensor = torch.as_tensor(
            self.moving_sampling_inverse, dtype=torch.float64, device=self.device
        )
        self.reference_sampling_tensor = torch.as_tensor(
            self.reference_sampling, dtype=torch.float64, device=self.device
        )
        self._planned_chunk_size = None
        self._cuda_sum_kernel = None
        self._cuda_sum_imported = False
        self._cuda_sample_kernel = None
        self._cuda_sample_imported = False
        self.reduction_lanes = torch.arange(
            128, dtype=torch.long, device=self.device
        )[None, None]
        self._moving_key = None
        self._refresh()

    def _refresh(self):
        # FLIRT can refresh the input while keeping the same reference level.
        key = (id(self.cost.moving), self.cost.smooth_size,
               id(getattr(self.cost, "moving_weight", None)))
        if key == self._moving_key:
            return
        self._moving_key = key
        self._planned_chunk_size = None
        self.moving = self.cost.moving.contiguous()
        self.moving_flat = self.moving.reshape(-1)
        self.upper = torch.tensor(
            [size - 1.0001 for size in self.moving.shape],
            dtype=torch.float32, device=self.device,
        )[None, :, None]
        self.maximum_lower = torch.tensor(
            [size - 2 for size in self.moving.shape],
            dtype=torch.long, device=self.device,
        )[None, :, None]
        self.smooth = torch.tensor(
            [self.cost.smooth_size / value
             for value in self.cost.moving_voxel_sizes],
            dtype=torch.float32, device=self.device,
        )[None, :, None]
        if self.cost.weighted:
            self.moving_weight_flat = self.cost.moving_weight.contiguous().reshape(-1)

    def _chunk_size(self, requested):
        # Conservative live-temporary allowance covers interpolation indices,
        # eight corners, coordinates, masks, sorted sums and NMI fuzzy bins.
        if self._planned_chunk_size is None:
            bytes_per_candidate = self.grid.shape[1] * 256
            bytes_per_candidate += (self.cost.bins + 1)**2 * 16
            available = self.memory_budget_bytes
            if self.device.type == "cuda":
                free, _ = torch.cuda.mem_get_info(self.device)
                allocated = torch.cuda.memory_allocated(self.device)
                available = min(int(free * 0.8), max(0, available - allocated))
            maximum = max(1, available // max(1, bytes_per_candidate))
            self._planned_chunk_size = min(self.max_batch_size, maximum)
        return min(self._planned_chunk_size, int(requested))

    def _coefficients(self, matrices):
        if isinstance(matrices, torch.Tensor) and matrices.device.type == "cuda":
            affine = matrices.to(device=self.device, dtype=torch.float64)
            inverse, info = torch.linalg.inv_ex(affine, check_errors=False)
            pull = (self.moving_sampling_tensor @ inverse
                    @ self.reference_sampling_tensor)
            return pull[:, :3].float(), info == 0
        if isinstance(matrices, torch.Tensor):
            matrices = matrices.detach().numpy()
        affine = np.asarray(matrices, dtype=np.float64)
        if not np.isfinite(affine).all():
            raise ValueError("affine candidates must contain only finite values")
        pull = (self.moving_sampling_inverse @ np.linalg.inv(affine)
                @ self.reference_sampling)
        coefficients = np.asarray(pull[:, :3], dtype=np.float32)
        prepared = torch.from_numpy(coefficients)
        if self.device.type == "cuda":
            # Each tiny pinned allocation is protected by PyTorch's transfer
            # event, including when several chunks are queued before return.
            return prepared.pin_memory().to(self.device, non_blocking=True), None
        return prepared.to(self.device), None

    def _coordinates(self, coefficients):
        x, y, z = self.grid.unbind(0)
        rows = []
        for row in coefficients.unbind(1):
            # Preserve the reference's float32 grouping; no GEMM/TF32/FMA.
            coordinate = torch.add(torch.mul(y, row[:, 1:2]),
                                   torch.mul(z, row[:, 2:3]))
            coordinate = torch.add(coordinate, row[:, 3:4])
            coordinate = torch.add(coordinate, torch.mul(x, row[:, 0:1]))
            rows.append(coordinate)
        return torch.stack(rows, dim=1)

    def _interpolate(self, data, coordinates):
        lower = torch.floor(coordinates).long()
        lower = torch.minimum(lower, self.maximum_lower).clamp_min_(0)
        delta = coordinates - lower.float()
        ix, iy, iz = lower.unbind(1)
        dx, dy, dz = delta.unbind(1)
        stride_x, stride_y = self.moving.shape[1] * self.moving.shape[2], self.moving.shape[2]
        index = ix * stride_x + iy * stride_y + iz
        v000 = data[index]
        v001 = data[index + 1]
        v010 = data[index + stride_y]
        v011 = data[index + stride_y + 1]
        v100 = data[index + stride_x]
        v101 = data[index + stride_x + 1]
        v110 = data[index + stride_x + stride_y]
        v111 = data[index + stride_x + stride_y + 1]
        temp1 = (v100 - v000) * dx + v000
        temp2 = (v101 - v001) * dx + v001
        temp3 = (v110 - v010) * dx + v010
        temp4 = (v111 - v011) * dx + v011
        temp5 = (temp3 - temp1) * dy + temp1
        temp6 = (temp4 - temp2) * dy + temp2
        return (temp6 - temp5) * dz + temp5

    def _sample_cpu(self, coefficients):
        # Reuse strict float32 voxel sampling while retaining the existing
        # batched reductions below. Candidate ordering/chunking is unchanged.
        values, weights = [], []
        for coefficient in coefficients:
            sampled, contribution, _valid = self.cost._sample_cpu(
                coefficient, nmi=self.nmi,
            )
            values.append(sampled)
            weights.append(contribution)
        return torch.stack(values), torch.stack(weights)

    def _sample(self, coefficients):
        # Keep the established tensor exception for one-voxel input axes.
        if self.device.type == "cpu" and min(self.moving.shape) >= 2:
            return self._sample_cpu(coefficients)
        if self.device.type == "cuda" and not self.cost.weighted:
            if not self._cuda_sample_imported:
                self._cuda_sample_imported = True
                try:
                    from ._batched_cuda import _sample_kernel
                    self._cuda_sample_kernel = _sample_kernel
                except ImportError:
                    pass
            if self._cuda_sample_kernel is not None:
                coefficients = coefficients.contiguous()
                size = self.grid.shape[1]
                values = torch.empty((len(coefficients), size), dtype=torch.float32,
                                     device=self.device)
                weights = torch.empty_like(values)
                with torch.cuda.device(self.device):
                    self._cuda_sample_kernel[((size + 255) // 256, len(coefficients))](
                        self.grid, coefficients, self.moving_flat, self.upper, self.smooth,
                        values, weights, N=size, SIZE_X=self.moving.shape[0],
                        SIZE_Y=self.moving.shape[1], SIZE_Z=self.moving.shape[2],
                        TAPER=self.cost.smooth_size > 0 or self.nmi, BLOCK=256,
                        num_warps=4, enable_fp_fusion=False,
                    )
                return values, weights
        return self._sample_tensor(coefficients)

    def _sample_tensor(self, coefficients):
        coordinates = self._coordinates(coefficients)
        valid = ((coordinates >= 0) & (coordinates <= self.upper)).all(dim=1)
        interpolation_coordinates = torch.minimum(coordinates.clamp_min(0), self.upper)
        values = self._interpolate(self.moving_flat, interpolation_coordinates)
        if self.cost.smooth_size > 0 or self.nmi:
            far_distance = self.upper - coordinates
            weight_per_axis = torch.where(
                coordinates < self.smooth, coordinates / self.smooth,
                torch.where(far_distance < self.smooth,
                            far_distance / self.smooth, 1.0),
            )
            weights = weight_per_axis.prod(dim=1).clamp_min_(0)
        else:
            weights = torch.ones_like(values)
        # The reference NMI class presently uses only its geometric weight.
        # Do not silently change its established weighted-cost behaviour.
        if self.cost.weighted and not self.nmi:
            moving_weights = self._interpolate(
                self.moving_weight_flat, interpolation_coordinates
            )
            weights = (weights * moving_weights
                       * self.cost.reference_weight_values[None]).clamp_min_(0)
        return values, weights * valid

    def _corratio(self, values, weights):
        order = self.bin_sort_order
        batch_size = values.shape[0]
        if batch_size not in self._segment_lengths:
            self._segment_lengths[batch_size] = self._segment_bin_lengths.repeat(batch_size)
        lengths = self._segment_lengths[batch_size]
        sorted_weights = weights[:, order]
        sorted_values = values[:, order]

        def segmented_sum(data):
            # Keep the one-dimensional CUB segmented-reduction path used by
            # the scalar CUDA reference.  A [B,N] axis=1 reduction instead
            # selects PyTorch's serial-per-bin CUDA kernel and changes sums.
            # Cached bincount lengths are nonnegative and cover the data;
            # unsafe skips redundant CPU-synchronized length validation.
            # Ignore the padding-only segment after preserving each row's
            # scalar allocation alignment for CUB's vectorized full tiles.
            if self._segment_padding:
                data = torch.nn.functional.pad(data, (0, self._segment_padding))
            reduced = torch.segment_reduce(data.reshape(-1), "sum", lengths=lengths,
                                           unsafe=True, initial=0).reshape(batch_size, -1)
            return reduced[:, :self.cost.bins]

        counts = segmented_sum(sorted_weights)
        sums = segmented_sum(sorted_weights * sorted_values)
        sums2 = segmented_sum((sorted_weights * sorted_values) * sorted_values)
        keep = counts > 2
        n = torch.where(keep, counts, 0)
        y = torch.where(keep, sums, 0)
        y2 = torch.where(keep, sums2, 0)
        safe_n = torch.where(keep, n, 3)
        within = (y2 - y.square() / safe_n) / (safe_n - 1)
        # Match the reference's kept-bin order without nonzero()/boolean
        # indexing, which would synchronize to size a CUDA output tensor.
        positions = keep.long().cumsum(dim=1).sub_(1).clamp_min_(0)
        packed = torch.zeros((values.shape[0], 4, self.cost.bins),
                             dtype=torch.float32, device=self.device)
        packed.scatter_add_(2, positions[:, None].expand(-1, 4, -1),
                            torch.stack((n, y, y2, within * n), dim=1))
        totals = self._compact_sum(packed, keep.sum(dim=1))
        total_n, total_sum, total_sum2, numerator = totals.unbind(dim=1)
        safe_total = total_n.clamp_min(2)
        total_variance = (total_sum2 - total_sum.square() / safe_total) / (safe_total - 1)
        cost = numerator / safe_total / total_variance
        cost = 1.0 - (1.0 - cost)
        return torch.where((total_n > 1) & (total_variance > 0), cost, 1.0)

    def _cpu_corratio_statistics(self, coefficients):
        statistics = [self.cost._corratio_cpu(coefficient)[:3]
                      for coefficient in coefficients]
        return tuple(torch.stack([row[column] for row in statistics])
                     for column in range(3))

    def _corratio_cpu(self, coefficients):
        counts, sums, sums2 = self._cpu_corratio_statistics(coefficients)
        keep = counts > 2
        n = torch.where(keep, counts, 0)
        y = torch.where(keep, sums, 0)
        y2 = torch.where(keep, sums2, 0)
        safe_n = torch.where(keep, n, 3)
        within = (y2 - y.square() / safe_n) / (safe_n - 1)
        # Match the reference's kept-bin order without nonzero()/boolean
        # indexing, which would synchronize to size a CUDA output tensor.
        positions = keep.long().cumsum(dim=1).sub_(1).clamp_min_(0)
        packed = torch.zeros((counts.shape[0], 4, self.cost.bins),
                             dtype=torch.float32, device=self.device)
        packed.scatter_add_(2, positions[:, None].expand(-1, 4, -1),
                            torch.stack((n, y, y2, within * n), dim=1))
        totals = self._compact_sum(packed, keep.sum(dim=1))
        total_n, total_sum, total_sum2, numerator = totals.unbind(dim=1)
        safe_total = total_n.clamp_min(2)
        total_variance = (total_sum2 - total_sum.square() / safe_total) / (safe_total - 1)
        cost = numerator / safe_total / total_variance
        cost = 1.0 - (1.0 - cost)
        return torch.where((total_n > 1) & (total_variance > 0), cost, 1.0)


    def _compact_sum(self, packed, lengths):
        """Preserve CUDA's one-dimensional sum grouping for kept bins.

        PyTorch changes reduction grouping when multiple rows are passed to
        sum().  Correlation-ratio variance subtracts large float32 values,
        making that last-bit difference measurable.  Reproduce its scalar
        input reduction grouping while keeping row lengths on the device.
        """
        if self.device.type != "cuda" or self.cost.bins > 256:
            return packed.sum(dim=2)
        if not self._cuda_sum_imported:
            self._cuda_sum_imported = True
            try:
                from ._batched_cuda import _compact_sum_kernel
                self._cuda_sum_kernel = _compact_sum_kernel
            except ImportError:
                pass
        if self._cuda_sum_kernel is not None:
            output = torch.empty(packed.shape[:2], dtype=packed.dtype, device=self.device)
            lanes = 128 if packed.shape[2] <= 512 else 512
            with torch.cuda.device(self.device):
                self._cuda_sum_kernel[(output.numel(),)](
                    packed, lengths, output, BINS=packed.shape[2],
                    ROWS=packed.shape[1], LANES=lanes,
                    num_warps=lanes // 32, enable_fp_fusion=False,
                )
            return output
        if packed.shape[2] > 512:
            return packed.sum(dim=2)
        length = lengths[:, None, None]
        lanes = self.reduction_lanes
        vectorized = length > 128
        input_width = torch.where(vectorized, length // 4, length).clamp_min(1)
        width = torch.pow(2.0, torch.floor(torch.log2(input_width.float()))).long()

        def load(index, valid):
            index = index.expand(-1, packed.shape[1], -1)
            selected = torch.gather(packed, 2, index.clamp(0, packed.shape[2] - 1))
            return torch.where(valid, selected, 0)

        first = torch.where(vectorized, lanes * 4, lanes)
        value = load(first, torch.where(vectorized, first + 3 < length, first < length))
        second = torch.where(vectorized, first + width * 4, first + width)
        value = value + load(second, torch.where(vectorized, second + 3 < length,
                                                second < length))
        tail = length - length.remainder(4) + lanes
        value = value + load(tail, vectorized & (lanes < 4) & (tail < length))
        # CUDA input vectorization keeps four separate accumulators before
        # combining them in this order; only fully present vectors are read.
        for component in (1, 2, 3):
            partial = load(first + component, vectorized & (first + 3 < length))
            partial = partial + load(second + component,
                                     vectorized & (second + 3 < length))
            value = value + partial
        value = torch.where(lanes < width, value, 0)
        # Inter-warp pairing precedes the ascending shuffle reduction.
        value = torch.where(width > 64,
                            value[..., :64] + value[..., 64:128], value[..., :64])
        value = torch.where(width > 32,
                            value[..., :32] + value[..., 32:64], value[..., :32])
        for _ in range(5):
            value = value[..., ::2] + value[..., 1::2]
        return value[..., 0]

    def _normmi(self, values, weights):
        bin_float = values * self.cost.test_factor + self.cost.test_offset
        # Failed CUDA inverse lanes contain NaNs and receive the failure cost
        # below.  Keep their unused scatter indices within the histogram.
        bin_float = torch.where(torch.isfinite(bin_float), bin_float, 0)
        truncated = torch.trunc(bin_float)
        raw_centre = truncated.long()
        minus = (raw_centre - 1).clamp_min(0)
        plus = (raw_centre + 1).clamp_max(self.cost.bins - 1)
        centre = raw_centre.clamp(0, self.cost.bins - 1)
        fractional = (bin_float - truncated).abs()
        centre_weight = torch.where(
            fractional < 0.5, 0.5 + fractional,
            torch.where(fractional > 0.5, 1.5 - fractional, 1.0),
        ).clamp(0, 1)
        minus_weight = torch.where(fractional < 0.5, 1 - centre_weight, 0)
        plus_weight = torch.where(fractional > 0.5, 1 - centre_weight, 0)
        stride = self.cost.bins + 1
        joint = torch.zeros((values.shape[0], stride * stride),
                            dtype=torch.float64, device=self.device)
        for bin_id, bin_weight in ((centre, centre_weight),
                                   (minus, minus_weight), (plus, plus_weight)):
            joint.scatter_add_(1, self.bin_offsets[None] + bin_id,
                               (weights * bin_weight).double())
        joint = joint.reshape(-1, stride, stride).float()
        first, second = joint.sum(dim=2), joint.sum(dim=1)
        total_lengths = torch.full((values.shape[0],), stride,
                                   dtype=torch.long, device=self.device)
        total = self._compact_sum(second[:, None], total_lengths)[:, 0]
        safe_total = torch.where(total > 0, total, 1)

        def entropy(histogram):
            probabilities = histogram.reshape(values.shape[0], -1) / safe_total[:, None]
            terms = probabilities * probabilities.clamp_min(torch.finfo(torch.float32).tiny).log()
            positive = probabilities > 0
            positions = positive.long().cumsum(dim=1).sub_(1).clamp_min_(0)
            packed = torch.zeros_like(terms)
            packed.scatter_add_(1, positions, torch.where(positive, terms, 0))
            return -self._compact_sum(packed[:, None], positive.sum(dim=1))[:, 0]

        joint_entropy = entropy(joint)
        nonzero_entropy = joint_entropy.abs() >= 1e-9
        safe_entropy = torch.where(nonzero_entropy, joint_entropy, 1)
        result = -(entropy(first) + entropy(second)) / safe_entropy
        result = torch.where(nonzero_entropy, result, 0)
        return torch.where(total > 0, result, -1.0)

    def _joint_histogram_cpu(self, values, weights):
        from fnit.flirt._cpu import histogram_nmi
        joints = []
        try:
            for value, weight in zip(values, weights):
                array = histogram_nmi(
                    value.numpy(), weight.numpy(), self.cost.bin_index.numpy(),
                    self.cost.bins, np.float32(self.cost.test_factor),
                    np.float32(self.cost.test_offset),
                )
                if not np.isfinite(array).all():
                    return None
                joints.append(torch.from_numpy(array))
        except ValueError:
            return None
        return torch.stack(joints)

    def _normmi_cpu(self, values, weights):
        joint = self._joint_histogram_cpu(values, weights)
        if joint is None:
            return self._normmi(values, weights)
        stride = self.cost.bins + 1
        joint = joint.float()
        first, second = joint.sum(dim=2), joint.sum(dim=1)
        total_lengths = torch.full((values.shape[0],), stride,
                                   dtype=torch.long, device=self.device)
        total = self._compact_sum(second[:, None], total_lengths)[:, 0]
        safe_total = torch.where(total > 0, total, 1)

        def entropy(histogram):
            probabilities = histogram.reshape(values.shape[0], -1) / safe_total[:, None]
            terms = probabilities * probabilities.clamp_min(torch.finfo(torch.float32).tiny).log()
            positive = probabilities > 0
            positions = positive.long().cumsum(dim=1).sub_(1).clamp_min_(0)
            packed = torch.zeros_like(terms)
            packed.scatter_add_(1, positions, torch.where(positive, terms, 0))
            return -self._compact_sum(packed[:, None], positive.sum(dim=1))[:, 0]

        joint_entropy = entropy(joint)
        nonzero_entropy = joint_entropy.abs() >= 1e-9
        safe_entropy = torch.where(nonzero_entropy, joint_entropy, 1)
        result = -(entropy(first) + entropy(second)) / safe_entropy
        result = torch.where(nonzero_entropy, result, 0)
        return torch.where(total > 0, result, -1.0)


    @torch.no_grad()
    def evaluate(self, matrices, *, chunk_size=None):
        """Return device costs, preserving the input order and all candidates."""
        self._refresh()
        if not isinstance(matrices, torch.Tensor):
            matrices = np.asarray(matrices, dtype=np.float64)
        if matrices.ndim == 2:
            matrices = matrices[None]
        if matrices.ndim != 3 or matrices.shape[1:] != (4, 4):
            raise ValueError("affine candidates must have shape [B, 4, 4]")
        count = matrices.shape[0]
        if count == 0:
            return torch.empty(0, dtype=torch.float32, device=self.device)
        requested = self.max_batch_size if chunk_size is None else int(chunk_size)
        if requested < 1:
            raise ValueError("chunk_size must be positive")
        size = self._chunk_size(requested)
        output = []
        for start in range(0, count, size):
            coefficients, invertible = self._coefficients(matrices[start:start + size])
            if self.device.type == "cpu" and not self.nmi and min(self.moving.shape) >= 2:
                result = self._corratio_cpu(coefficients)
            else:
                values, weights = self._sample(coefficients)
                if self.device.type == "cpu" and self.nmi and min(self.moving.shape) >= 2:
                    result = self._normmi_cpu(values, weights)
                else:
                    result = self._normmi(values, weights) if self.nmi else self._corratio(values, weights)
            if invertible is not None:
                result = torch.where(invertible, result, -1.0 if self.nmi else 1.0)
            output.append(result)
        return torch.cat(output)

    __call__ = evaluate
