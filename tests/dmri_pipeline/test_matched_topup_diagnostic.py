"""Configuration and observer contracts; these fixtures are not benchmarks."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

SOURCE = Path(__file__).resolve().parents[2] / "validation/dmri_pipeline/public10_20261002/benchmark_matched_topup_pair.py"
spec = importlib.util.spec_from_file_location("matched_pair_diagnostic", SOURCE)
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


@pytest.fixture
def configuration():
    from fnit.topup import TOPUPConfig
    config = TOPUPConfig()
    schedules = {"warpres": config.warp_resolution_mm, "subsamp": config.subsampling,
                 "fwhm": config.fwhm_mm, "miter": config.maximum_iterations,
                 "lambda": config.regularization}
    options = {"ssqlambda": "1", "regmod": "bending_energy",
               "estmov": "1,1,1,1,1,0,0,0,0", "minmet": "0,0,0,0,0,1,1,1,1",
               "splineorder": "3", "numprec": "double", "interp": "spline", "scale": "1"}
    entries = {**{key: ",".join(map(str, values)) for key, values in schedules.items()}, **options}
    return config, entries


def literal(entries):
    return "\n".join("--" + key + "=" + value for key, value in entries.items())


def test_literal_nine_level_options_match_values_not_name(configuration):
    config, entries = configuration
    result = diagnostic.parse_configuration("# literal diagnostic fixture\n" + literal(entries), config)
    assert result["literal_configuration_matches_frozen_defaults"]
    assert result["other_options"]["numprec"] == "double"


@pytest.mark.parametrize("option,value", [("miter", "1,1,1,1,1,1,1,1,1"),
                                           ("minmet", "1,1,1,1,1,0,0,0,0"),
                                           ("numprec", "float"), ("unknown", "1")])
def test_named_b02b0_with_different_values_is_rejected(configuration, option, value):
    config, entries = configuration
    entries[option] = value
    with pytest.raises(ValueError, match="configuration_"):
        diagnostic.parse_configuration(literal(entries), config)


def test_compiler_observer_preserves_arguments_return_exceptions_and_restores(monkeypatch):
    calls = []
    sentinel = object()

    def original(source, *, options):
        calls.append((source, options))
        if source == "failed":
            raise RuntimeError("underlying failure")
        return sentinel

    module = SimpleNamespace(compile=original)
    monkeypatch.setitem(sys.modules, "triton.compiler", module)
    with diagnostic.CompilerObserver() as observer:
        assert module.compile("source", options={"dtype": "double"}) is sentinel
        with pytest.raises(RuntimeError, match="underlying failure"):
            module.compile("failed", options={"dtype": "double"})
    assert module.compile is original
    assert calls == [("source", {"dtype": "double"}), ("failed", {"dtype": "double"})]
    report = observer.report()
    assert report["status"] == "observed" and report["compile_calls"] == 2
    assert [item["status"] for item in report["calls"]] == ["complete", "failed"]
    assert report["wall_seconds"] >= 0


def test_no_compile_observation_stays_unknown(monkeypatch):
    monkeypatch.setitem(sys.modules, "triton.compiler", SimpleNamespace(compile=lambda *a, **k: None))
    with diagnostic.CompilerObserver() as observer:
        pass
    assert observer.report()["status"] == "unknown"
    assert observer.report()["wall_seconds"] is None
