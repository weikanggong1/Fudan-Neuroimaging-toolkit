"""Raw EDDY and tractography random seeds have independent scopes."""

from types import SimpleNamespace

from fnit.connectome.pipeline import UKBConnectome_pipeline


def test_bids_entry_routes_eddy_seed_only_to_preprocessing(monkeypatch):
    import fnit.connectome.bids as bids

    preparation = {}
    tracking = {}

    def prepare(*args, **kwargs):
        preparation.update(kwargs)
        return SimpleNamespace(dwi="corrected.nii.gz", bvals="raw.bval",
                               bvecs="rotated.bvec", freesurfer_subject_dir="fs")

    def compute(self, *args, **kwargs):
        tracking.update(kwargs)
        return "result"

    monkeypatch.setattr(bids, "prepare_bids_connectome", prepare)
    monkeypatch.setattr(UKBConnectome_pipeline, "__call__", compute)
    result = UKBConnectome_pipeline(device="cpu").run_bids(
        "raw_bids", "output", subject="01", n_seeds=100000,
        eddy_gp_seed=12345, seed=7,
    )
    assert result == "result"
    assert preparation["eddy_gp_seed"] == 12345
    assert "seed" not in preparation
    assert tracking["seed"] == 7
    assert "eddy_gp_seed" not in tracking
