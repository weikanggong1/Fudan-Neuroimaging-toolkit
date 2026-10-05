"""Numerical acceptance shared by complete real-data benchmark controllers."""


def compare_histories(old, new):
    """Keep iteration identity separate from floating-point state differences."""
    counts_old = [{key: row[key] for key in ("outer", "em")} for row in old]
    counts_new = [{key: row[key] for key in ("outer", "em")} for row in new]
    errors = {}
    for key in ("kappa", "cost"):
        pairs = [(float(left[key]), float(right[key])) for left, right in zip(old, new)]
        errors[key] = {
            "max_absolute_error": max((abs(a - b) for a, b in pairs), default=0.0),
            "max_relative_error": max((abs(a - b) / max(abs(a), abs(b), 1e-300)
                                       for a, b in pairs), default=0.0),
        }
    return {"history_exact": old == new, "iterations_exact": counts_old == counts_new,
            "baseline_iterations": counts_old, "candidate_iterations": counts_new,
            "state_errors": errors}
