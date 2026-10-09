"""诊断编译适配器保留宏、优化参数和固定库；不代表影像 benchmark。"""
import importlib.util
from pathlib import Path
import pytest


PATH = Path(__file__).resolve().parents[2] / "validation/recon_all/python_gpu_port/native_probe_build.py"
SPEC = importlib.util.spec_from_file_location("native_probe_build_test", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_ninja_probe_commands_preserve_original_flags_and_libraries(tmp_path, monkeypatch):
    (tmp_path / "CMakeCache.txt").write_text("CMAKE_CXX_COMPILER:FILEPATH=/conda/bin/c++\n")
    (tmp_path / "build.ninja").write_text(
        "build utils/CMakeFiles/utils.dir/mrisurf_mri.cpp.o: CXX original.cpp\n"
        "  DEFINES = -DHAVE_OPENMP\n  INCLUDES = -I/source/include\n"
        "  FLAGS = -O3 -ffp-contract=off\n\n"
        "build mris_make_surfaces/mris_place_surface: LINK original.o\n"
        "  FLAGS = -O3\n  LINK_FLAGS = -Wl,-Map,ld_map.txt\n"
        "  LINK_LIBRARIES = utils/libutils.a packages/jpeg/libjpeg.a -lm\n\n")
    calls = []
    monkeypatch.setattr(MODULE.subprocess, "run", lambda command, **kwargs: calls.append((command, kwargs)))
    original = (tmp_path / "build.ninja").read_bytes()
    obj = tmp_path / "private/probe.o"
    command = MODULE.compile_probe(source=tmp_path / "source", build=tmp_path,
        target="utils/CMakeFiles/utils.dir/mrisurf_mri.cpp.o",
        patched=tmp_path / "private/probe.cpp", output=obj)
    assert command[0] == "/conda/bin/c++"
    assert all(flag in command for flag in ("-DHAVE_OPENMP", "-O3", "-ffp-contract=off"))
    linked = MODULE.link_probe(build=tmp_path, main_object=tmp_path / "private/main.o",
        output=tmp_path / "private/probe", extra_objects=(obj,))
    assert linked.index(str(obj)) < linked.index("utils/libutils.a")
    assert "packages/jpeg/libjpeg.a" in linked
    assert f"-Wl,-Map,{tmp_path}/private/link.map" in linked
    assert all(kwargs == {"cwd": tmp_path, "check": True} for _, kwargs in calls)
    assert (tmp_path / "build.ninja").read_bytes() == original


def test_ninja_adapter_rejects_unexpanded_values(tmp_path):
    (tmp_path / "build.ninja").write_text("build object.o: CXX source.cpp\n  FLAGS = $unexpanded\n")
    with pytest.raises(RuntimeError, match="unresolved"):
        MODULE._ninja_variables(tmp_path, "object.o")
