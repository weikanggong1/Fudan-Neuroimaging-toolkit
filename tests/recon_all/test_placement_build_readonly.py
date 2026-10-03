"""有限原生构建不能覆盖读取中的共享链接map；不模拟影像算法。"""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


class PlacementBuildReadOnlyTest(unittest.TestCase):
    def test_link_diagnostics_belong_to_private_output(self):
        script = Path(__file__).resolve().parents[2] / "tools/place_surface_hotspots/build_native.py"
        spec = importlib.util.spec_from_file_location("placement_builder", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, build, output = root / "source", root / "build", root / "private"
            for name in module.EXPECTED:
                path = source / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture source")
            (build / "utils").mkdir(parents=True)
            (build / "utils/libutils.a").write_bytes(b"fixture archive")
            shared_map = build / "ld_map.txt"
            shared_map.write_text("owned by original build")
            before = {p.relative_to(build): p.read_bytes() for p in build.rglob("*") if p.is_file()}

            def compile_fixture(build, original, own_source, obj, commands):
                obj.write_bytes(b"fixture object")
                return Path("/fixture/x86_64-conda-linux-gnu-c++"), ["fixture compiler"]

            def execute_fixture(argv, *, cwd=None, check=True):
                if "-o" in argv:
                    Path(argv[argv.index("-o") + 1]).write_bytes(b"fixture program")
                    for token in argv:
                        if token.startswith("-Wl,-Map,"):
                            path = Path(token.split(",", 2)[2])
                            if not path.is_absolute():
                                path = cwd / path
                            path.write_text("new link diagnostics")

            command = "cd build && conda-c++ main.o utils/libutils.a -Wl,-Map,ld_map.txt -o placement"
            with patch.object(module, "compile_source", side_effect=compile_fixture), \
                 patch.object(module.subprocess, "run", side_effect=execute_fixture):
                binary = module.build_one(build, source, output, [command], patched=False)
            after = {p.relative_to(build): p.read_bytes() for p in build.rglob("*") if p.is_file()}
            self.assertEqual(before, after)
            self.assertEqual((output / "ld_map.txt").read_text(), "new link diagnostics")
            self.assertTrue(binary.is_file())
            self.assertIn("-Wl,-Map," + str(output / "ld_map.txt"),
                          json.loads((output / "build.json").read_text())["link_command"])


if __name__ == "__main__":
    unittest.main()
