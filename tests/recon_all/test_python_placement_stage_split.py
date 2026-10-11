"""复用完整放置接口的阶段拆分，不以这些接线测试替代影像回归。"""
from unittest.mock import patch

import pytest

from fnit.recon_all import native_free


def prepare_subject(tmp_path):
    for name in ("surf", "scripts", "mri", "label"):
        (tmp_path / name).mkdir()
    return tmp_path


def test_split_white_leaves_pial_and_metrics_pending(tmp_path):
    subject = prepare_subject(tmp_path)
    with patch("fnit.recon_all.surface_placement_python.run_final_white_python",
               return_value={"output": "white", "seconds": 1.}) as white, \
         patch("fnit.recon_all.surface_placement_python.run_pial_python") as pial, \
         patch("fnit.recon_all.native_free._finish_cortical_metrics") as metrics:
        result = native_free._hemisphere_operation(subject=subject, hemi="lh", device="cpu",
            threads=2, operation="final_white", assets=subject,
            surface_placement_backend="python")
    white.assert_called_once_with(subject_dir=subject, hemi="lh", device="cpu")
    pial.assert_not_called(); metrics.assert_not_called()
    assert result["result"]["placement_pending"] and result["result"]["metrics_pending"]
    assert result["stages"][0]["name"] == "final_white_lh"


def test_split_pial_copies_complete_surface_and_defers_metrics(tmp_path):
    subject = prepare_subject(tmp_path)
    source = subject / "surf/lh.pial.T1"
    source.write_bytes(b"complete-pial")
    with patch("fnit.recon_all.surface_placement_python.run_pial_python",
               return_value={"output": str(source), "seconds": 1.}) as pial, \
         patch("fnit.recon_all.surface_placement_python.run_final_white_python") as white, \
         patch("fnit.recon_all.native_free._finish_cortical_metrics") as metrics:
        result = native_free._hemisphere_operation(subject=subject, hemi="lh", device="cpu",
            threads=2, operation="pial_surface", assets=subject,
            surface_placement_backend="python")
    pial.assert_called_once_with(subject_dir=subject, hemi="lh", device="cpu")
    white.assert_not_called(); metrics.assert_not_called()
    assert (subject / "surf/lh.pial").read_bytes() == source.read_bytes()
    assert not result["result"]["placement_pending"] and result["result"]["metrics_pending"]


@pytest.mark.parametrize("operation", ["final_white", "pial_surface"])
def test_native_control_cannot_enter_python_split_operations(tmp_path, operation):
    subject = prepare_subject(tmp_path)
    with patch("fnit.recon_all.native_free.subprocess.run",
               side_effect=AssertionError("original stage program called")), \
         pytest.raises(ValueError, match="require the Python backend"):
        native_free._hemisphere_operation(subject=subject, hemi="lh", device="cpu",
            threads=2, operation=operation, assets=subject,
            surface_placement_backend="native")


def test_combined_python_entry_reuses_both_steps_and_preserves_deferred_metrics(tmp_path):
    subject = prepare_subject(tmp_path)
    (subject / "surf/lh.pial.T1").write_bytes(b"complete-pial")
    calls = []
    def white(**kwargs):
        calls.append("white")
        return {"output": "white"}
    def pial(**kwargs):
        calls.append("pial")
        return {"output": "pial"}
    with patch("fnit.recon_all.surface_placement_python.run_final_white_python", side_effect=white), \
         patch("fnit.recon_all.surface_placement_python.run_pial_python", side_effect=pial), \
         patch("fnit.recon_all.native_free._finish_cortical_metrics") as metrics:
        result = native_free._finish_cortical_surface(subject=subject, hemi="lh", binary=None,
            assets=subject, device="cpu", threads=2, surface_placement_backend="python",
            defer_metrics=True)
    assert calls == ["white", "pial"]
    assert result == {"final_white_report": {"output": "white"},
                      "pial_report": {"output": "pial"},
                      "placement_pending": False, "metrics_pending": True}
    metrics.assert_not_called()
