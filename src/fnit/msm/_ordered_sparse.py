"""Deterministic GPU area-corrected interpolation in the existing CSR order.

No atomic float sums or sparse-matmul reduction trees are used. SciPy's
intermediate diagonal products reverse row indices; its row sums use
NumPy reduceat (first element plus a pairwise sum of the remainder). These
orders matter to later registration trajectories and are preserved here.
"""
import torch


class SparseLayoutTooLarge(RuntimeError):
    """Use the existing CPU CSR path for an unusually dense padded layout."""


def _pairwise(values):
    """NumPy float64 pairwise sum for one fixed-length batch of rows."""
    n = values.shape[1]
    if n < 8:
        result = torch.full((len(values),), -0.0, dtype=values.dtype, device=values.device)
        for index in range(n):
            result = result+values[:, index]
        return result
    if n <= 128:
        accumulators = [values[:, index] for index in range(8)]
        end = n-n % 8
        for index in range(8, end, 8):
            for slot in range(8):
                accumulators[slot] = accumulators[slot]+values[:, index+slot]
        result = ((accumulators[0]+accumulators[1])+(accumulators[2]+accumulators[3]))+(
            (accumulators[4]+accumulators[5])+(accumulators[6]+accumulators[7]))
        for index in range(end, n):
            result = result+values[:, index]
        return result
    split = n//2
    split -= split % 8
    return _pairwise(values[:, :split])+_pairwise(values[:, split:])


def _group_table(groups, order, count):
    """Padded indices in a specified stable group/entry order."""
    sorted_groups = groups[order]
    lengths = torch.bincount(sorted_groups, minlength=count)
    maximum = int(lengths.max().item()) if count else 0
    if count*maximum > 16_000_000:
        raise SparseLayoutTooLarge("ordered sparse padding exceeds bounded GPU layout")
    table = torch.zeros((count, maximum), dtype=torch.long, device=groups.device)
    if len(order):
        offsets = torch.cumsum(lengths, 0)-lengths
        position = torch.arange(len(order), device=groups.device)-offsets[sorted_groups]
        table[sorted_groups, position] = order
    return table, lengths


def _sequential_sum(values, table, lengths):
    # The C++ CSR column sum and matvec accumulate from zero in entry order.
    shape = (len(table),)+tuple(values.shape[1:])
    result = torch.zeros(shape, dtype=values.dtype, device=values.device)
    for slot in range(table.shape[1]):
        term = values[table[:, slot]]
        active = slot < lengths
        if values.ndim > 1:
            active = active[:, None]
        result = result+torch.where(active, term, torch.zeros_like(term))
    return result


def adaptive_values(forward_ids, forward_weights, reverse_ids, reverse_weights,
                    values, old_area, new_area):
    """Match the reference scalar/matrix CSR algorithm without host weights."""
    device = forward_weights.device
    m, n = len(forward_ids), len(reverse_ids)
    forward_rows = torch.arange(m, device=device).repeat_interleave(3)
    reverse_columns = torch.arange(n, device=device).repeat_interleave(3)
    # Triangle vertex IDs are distinct; CSR conversion retains zero-valued
    # entries for this degree comparison, before the diagonal products.
    reverse_rows = reverse_ids.reshape(-1)
    reverse_lengths = torch.bincount(reverse_rows, minlength=m)
    choose = reverse_lengths > 3
    take_forward = ~choose[forward_rows]
    take_reverse = choose[reverse_rows]
    rows = torch.cat((forward_rows[take_forward], reverse_rows[take_reverse]))
    columns = torch.cat((forward_ids.reshape(-1)[take_forward], reverse_columns[take_reverse]))
    weights = torch.cat((forward_weights.reshape(-1)[take_forward], reverse_weights.reshape(-1)[take_reverse]))
    nonzero = weights != 0
    rows, columns, weights = rows[nonzero], columns[nonzero], weights[nonzero]
    order = torch.argsort(rows*n+columns, stable=True)
    rows, columns, weights = rows[order], columns[order], weights[order]
    old_area = torch.as_tensor(old_area, dtype=torch.float64, device=device)
    new_area = torch.as_tensor(new_area, dtype=torch.float64, device=device)
    weighted = new_area[rows]*weights
    # Column sums visit rows in ascending order; every row/column pair is
    # unique, so the reversal within each intermediate CSR row is irrelevant.
    column_order = torch.argsort(columns, stable=True)
    column_table, column_lengths = _group_table(columns, column_order, n)
    correction = _sequential_sum(weighted, column_table, column_lengths)
    factors = torch.where(correction > 0, old_area/correction, torch.zeros_like(correction))
    weighted = weighted*factors[columns]
    row_table, row_lengths = _group_table(rows, torch.arange(len(rows), device=device), m)
    sums = torch.zeros(m, dtype=torch.float64, device=device)
    for length in torch.unique(row_lengths).cpu().tolist():
        if length == 0:
            continue
        selected = torch.nonzero(row_lengths == length).flatten()
        entries = weighted[row_table[selected, :length]]
        sums[selected] = entries[:, 0] if length == 1 else entries[:, 0]+_pairwise(entries[:, 1:])
    reciprocal = torch.where(sums > 0, 1./sums, torch.zeros_like(sums))
    normalized = reciprocal[rows]*weighted
    samples = torch.as_tensor(values, dtype=torch.float64, device=device)
    product = normalized[:, None]*samples[columns] if samples.ndim > 1 else normalized*samples[columns]
    # Final left-diagonal multiplication reverses ascending column order.
    reverse_table = row_table.clone()
    for length in torch.unique(row_lengths).cpu().tolist():
        if length:
            selected = torch.nonzero(row_lengths == length).flatten()
            reverse_table[selected, :length] = row_table[selected, :length].flip(1)
    return _sequential_sum(product, reverse_table, row_lengths)
