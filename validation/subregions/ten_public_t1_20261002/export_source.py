"""Freeze current native FNIT using the audited source export implementation."""
from pathlib import Path
import runpy
import sys

HELPER = Path(__file__).parents[1] / "reproducibility_20261002/export_final_source.py"

if __name__ == "__main__":
    # The helper copies source to the caller's external output directory only.
    module = runpy.run_path(str(HELPER), run_name="fnit_cohort_source_export")
    module["main"].__globals__["NAME"] = "source_ten_public_t1_20261002"
    module["main"]()
