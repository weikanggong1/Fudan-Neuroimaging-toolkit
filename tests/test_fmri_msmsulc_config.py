"""Keep custom MSM schedules explicit at the BIDS surface entry point."""

from types import SimpleNamespace

import pytest

from fnit.fmri import MSMSulcConfig, fMRISurface_pipeline


def test_provided_spheres_cannot_silently_ignore_msm_configuration():
    with pytest.raises(ValueError, match="supplied registered_spheres"):
        fMRISurface_pipeline(
            "/missing/bids", "/missing/derivatives", subject="0001",
            recon_all="/missing/recon-all", hcp_assets_dir="/missing/assets",
            registered_spheres=("/missing/L.surf.gii", "/missing/R.surf.gii"),
            msm_config=MSMSulcConfig.ssd_affine(),
        )


@pytest.mark.parametrize("custom", [None, "/absolute/MSMSulcStrainFinalconf"])
def test_surface_cli_forwards_requested_msm_config(monkeypatch, custom):
    from fnit.fmri import cli

    seen = {}

    def surface(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(dtseries="/result.dtseries.nii")

    monkeypatch.setattr(cli, "fMRISurface_pipeline", surface)
    arguments = [
        "surface", "--bids-root", "/bids", "--derivatives-root", "/derivatives",
        "--subject", "0001", "--recon-all", "/recon-all",
        "--surface-assets-dir", "/assets",
    ]
    if custom is not None:
        arguments += ["--msm-config", custom]
    assert cli.main(arguments) == 0
    assert seen["msm_config"] == custom
    assert seen["msm_execution"] == "optimized"
    assert seen["msmsulc_qc_policy"] == "report"


@pytest.mark.parametrize("policy", ["report", "repair", "error"])
def test_surface_cli_forwards_msmsulc_qc_policy(monkeypatch, policy):
    from fnit.fmri import cli

    seen = {}
    def surface(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(dtseries="/result.dtseries.nii")

    monkeypatch.setattr(cli, "fMRISurface_pipeline", surface)
    assert cli.main([
        "surface", "--bids-root", "/bids", "--derivatives-root", "/derivatives",
        "--subject", "0001", "--recon-all", "/recon-all",
        "--surface-assets-dir", "/assets", "--msmsulc-qc-policy", policy,
    ]) == 0
    assert seen["msmsulc_qc_policy"] == policy


def test_invalid_msmsulc_qc_policy_is_rejected_before_loading_bids():
    with pytest.raises(ValueError, match="msmsulc_qc_policy"):
        fMRISurface_pipeline(
            "/missing/bids", "/missing/derivatives", subject="0001",
            recon_all="/missing/recon-all", hcp_assets_dir="/missing/assets",
            msmsulc_qc_policy="unknown",
        )


def test_invalid_msm_execution_is_rejected_before_loading_bids():
    with pytest.raises(ValueError, match="msm_execution"):
        fMRISurface_pipeline(
            "/missing/bids", "/missing/derivatives", subject="0001",
            recon_all="/missing/recon-all", hcp_assets_dir="/missing/assets",
            msm_execution="shorter_iterations",
        )
