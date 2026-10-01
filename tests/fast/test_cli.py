"""Lightweight CLI checks for TorchFAST output and overwrite behavior."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from fnit import cli
from fnit import fast as fast_module


OUTPUTS = {
    "_pve_0.nii.gz": "pve_csf",
    "_pve_1.nii.gz": "pve_gm",
    "_pve_2.nii.gz": "pve_wm",
    "_seg.nii.gz": "hard_segmentation",
    "_pveseg.nii.gz": "pve_segmentation",
    "_mixeltype.nii.gz": "mixel_type",
    "_bias.nii.gz": "bias_field",
    "_restore.nii.gz": "restored",
}


class _Volume:
    def __init__(self, value, saved):
        self.value = value
        self.saved = saved

    def save(self, path):
        path = Path(path)
        self.saved.append(path)
        path.write_text(self.value)


def _fake_fast(captured, saved):
    class FakeFast:
        def __init__(self, **options):
            captured["options"] = options

        def __call__(self, image, mask=None):
            captured["call"] = (image, mask)
            return SimpleNamespace(**{
                field: _Volume(field, saved) for field in OUTPUTS.values()
            })

    return FakeFast


@pytest.mark.parametrize("execution", ["tensor", "fsl"])
def test_fast_cli_writes_fsl_names_through_temporary_files(tmp_path, monkeypatch, capsys, execution):
    captured, saved = {}, []
    monkeypatch.setattr(fast_module, "TorchFAST", _fake_fast(captured, saved))
    prefix = tmp_path / "nested" / "subject"
    previous = Path(f"{prefix}_pve_1.nii.gz")
    previous.parent.mkdir(parents=True)
    previous.write_text("old")

    cli.main([
        "fast", "-i", "input.nii.gz", "-o", str(prefix), "--mask", "mask.nii.gz",
        "--device", "cuda:7", "--threads", "3", "-W", "2", "-I", "5",
        "-O", "6", "-l", "12", "-f", "0.03", "-H", "0.2", "-R", "0.4",
        "--pve-steps", "11", "-N", "-b", "-B", "--overwrite",
        "--execution", execution,
    ])

    assert captured == {
        "options": {
            "device": "cuda:7", "threads": 3, "init_iterations": 2,
            "bias_iterations": 5, "fixed_iterations": 6, "bias_fwhm_mm": 0.0,
            "init_mrf": 0.03, "mrf": 0.2, "mixel_mrf": 0.4, "pve_steps": 11,
            "execution": execution,
        },
        "call": ("input.nii.gz", "mask.nii.gz"),
    }
    expected = [Path(f"{prefix}{suffix}") for suffix in OUTPUTS]
    assert [Path(line) for line in capsys.readouterr().out.splitlines()] == expected
    assert [path.read_text() for path in expected] == list(OUTPUTS.values())
    assert all(path.parent == prefix.parent and path not in expected for path in saved)
    assert all(path.name.endswith(".nii.gz") for path in saved)
    assert not list(prefix.parent.glob(".*.tmp-*"))


def test_fast_cli_refuses_existing_output_before_inference(tmp_path, monkeypatch):
    captured, saved = {}, []
    monkeypatch.setattr(fast_module, "TorchFAST", _fake_fast(captured, saved))
    prefix = tmp_path / "subject"
    existing = Path(f"{prefix}_seg.nii.gz")
    existing.write_text("old")

    with pytest.raises(FileExistsError, match="use --overwrite"):
        cli.main(["fast", "-i", "input.nii.gz", "-o", str(prefix)])

    assert "call" not in captured
    assert existing.read_text() == "old"
    assert saved == []


def test_atomic_save_failure_preserves_existing_file_and_cleans_temporary(tmp_path):
    destination = tmp_path / "result.nii.gz"
    destination.write_text("old")
    observed = []

    class BrokenVolume:
        def save(self, path):
            path = Path(path)
            observed.append(path)
            path.write_text("partial")
            raise RuntimeError("save failed")

    with pytest.raises(RuntimeError, match="save failed"):
        cli._atomic_save(BrokenVolume(), destination)

    assert destination.read_text() == "old"
    assert len(observed) == 1
    assert observed[0] != destination
    assert observed[0].parent == destination.parent
    assert observed[0].name.endswith(".nii.gz")
    assert not observed[0].exists()


def test_multi_subject_cli_is_unavailable(capsys):
    with pytest.raises(SystemExit) as error:
        cli.main(["batch", "jobs.json"])
    assert error.value.code == 2
    assert "invalid choice: 'batch'" in capsys.readouterr().err
