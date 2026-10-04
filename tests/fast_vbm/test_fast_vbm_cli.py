"""The public FastVBM CLI is a single-subject entry point."""

from pathlib import Path

import pytest

from fnit import cli
from fnit import fast_vbm as fast_vbm_module


def _fake_pipeline(captured):
    class Result:
        def save(self, output_dir, overwrite=False):
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            paths = {}
            for name, filename in fast_vbm_module.OUTPUT_FILENAMES.items():
                path = output_dir / filename
                path.write_text(name)
                paths[name] = str(path)
            (output_dir / "fast_vbm_report.json").write_text("{}")
            captured["save"] = (output_dir, overwrite)
            return paths

    class Pipeline:
        def __init__(self, **options):
            captured["options"] = options

        def __call__(
            self, image, template, brain_mask=None, reference_mask=None
        ):
            captured["call"] = (
                image, template, brain_mask, reference_mask
            )
            return Result()

    return Pipeline


@pytest.mark.parametrize("execution", [None, "fsl"])
def test_fast_vbm_cli_runs_one_subject_and_prints_all_outputs(
        tmp_path, monkeypatch, capsys, execution):
    captured = {}
    monkeypatch.setattr(fast_vbm_module, "FastVBM", _fake_pipeline(captured))
    output = tmp_path / "subject"

    arguments = [
        "fast-vbm", "-i", "T1w.nii.gz", "--template", "template.nii.gz",
        "-o", str(output), "--brain-mask", "mask.nii.gz",
        "--reference-mask", "reference-mask.nii.gz",
        "--synthstrip-weights", "strip.pt", "--synthmorph-weights", "morph.h5",
        "--device", "cuda:2", "--threads", "3",
        "--synthmorph-extent", "192", "--synthmorph-hyper", "0.4",
        "--synthmorph-steps", "6", "--no-bias", "--overwrite",
    ]
    if execution is not None:
        arguments.extend(["--fast-execution", execution])
    cli.main(arguments)

    assert captured == {
        "options": {
            "device": "cuda:2", "threads": 3,
            "synthstrip_weights": "strip.pt",
            "synthmorph_weights": "morph.h5", "bias_correction": False,
            "registration_backend": "synthmorph",
            "fast_execution": execution or "tensor",
            "synthmorph_extent": 192, "synthmorph_hyper": 0.4,
            "synthmorph_steps": 6,
            "fnirt_strides": (4, 2, 1, 1),
            "fnirt_steps": (5, 5, 10, 5),
            "fnirt_input_fwhm_mm": (6.0, 4.0, 2.0, 2.0),
            "fnirt_reference_fwhm_mm": (4.0, 2.0, 0.0, 0.0),
            "fnirt_warp_resolution_mm": 10.0,
            "fnirt_regularization": (150.0, 75.0, 50.0, 30.0),
        },
        "call": (
            "T1w.nii.gz", "template.nii.gz", "mask.nii.gz",
            "reference-mask.nii.gz",
        ),
        "save": (output, True),
    }
    expected = [output / name for name in fast_vbm_module.OUTPUT_FILENAMES.values()]
    expected.append(output / "fast_vbm_report.json")
    assert [Path(line) for line in capsys.readouterr().out.splitlines()] == expected
    assert all(path.is_file() for path in expected)


def test_fast_vbm_cli_rejects_existing_output_before_model_load(
        tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr(fast_vbm_module, "FastVBM", _fake_pipeline(captured))
    output = tmp_path / "subject"
    output.mkdir()
    existing = output / "T1_GM_JAC_nl.nii.gz"
    existing.write_text("old")

    with pytest.raises(FileExistsError, match="use --overwrite"):
        cli.main([
            "fast-vbm", "-i", "T1w.nii.gz", "--template", "template.nii.gz",
            "-o", str(output),
        ])

    assert captured == {}
    assert existing.read_text() == "old"


def test_multi_subject_cli_is_not_registered(capsys):
    with pytest.raises(SystemExit) as error:
        cli.main(["batch"])

    assert error.value.code == 2
    assert "invalid choice: 'batch'" in capsys.readouterr().err


def test_fast_vbm_cli_selects_fnirt_backend(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr(fast_vbm_module, "FastVBM", _fake_pipeline(captured))

    cli.main([
        "fast-vbm", "-i", "T1w.nii.gz", "--template", "template.nii.gz",
        "-o", str(tmp_path / "subject"), "--registration-backend", "fnirt",
        "--fnirt-warp-resolution-mm", "8",
        "--fnirt-steps", "4", "3", "2", "1",
    ])

    assert captured["options"]["registration_backend"] == "fnirt"
    assert captured["options"]["fnirt_warp_resolution_mm"] == 8
    assert captured["options"]["fnirt_steps"] == (4, 3, 2, 1)


@pytest.mark.parametrize(
    "removed_option",
    (
        "--linear-strides",
        "--linear-steps",
        "--linear-learning-rates",
        "--fnirt-learning-rates",
        "--fnirt-jacobian-penalty",
    ),
)
def test_fast_vbm_cli_rejects_removed_retired_options(removed_option, capsys):
    with pytest.raises(SystemExit) as error:
        cli.main([
            "fast-vbm",
            "-i",
            "T1w.nii.gz",
            "--template",
            "template.nii.gz",
            "-o",
            "result",
            removed_option,
            "1",
        ])

    assert error.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err
