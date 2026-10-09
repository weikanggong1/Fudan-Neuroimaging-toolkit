"""GCSA worker缓存的真实文件版本失效合同；不运行几何优化。"""
from pathlib import Path
import os
import tempfile
import unittest

from fnit.recon_all.gcsa_label_python import GCSAFeatureCache, _file_version


class FeatureCacheVersions(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.subject = Path(self.directory.name)
        names = ("surf/lh.smoothwm", "surf/lh.sphere.reg",
                 "mri/aseg.presurf.mgz", "label/lh.cortex.label")
        self.paths = tuple(self.subject / name for name in names)
        for path in self.paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"first")
        self.cache = object.__new__(GCSAFeatureCache)
        self.cache.subject = self.subject.resolve()
        self.cache.hemi = "lh"
        self.cache.device = "cuda:1"
        self.cache._input_paths = self.paths
        self.cache.signature = tuple(_file_version(path) for path in self.paths)

    def test_unchanged_inputs_accept(self):
        self.assertIsNone(self.cache.validate(self.subject, "lh", "cuda:1"))

    def test_each_same_size_changed_input_rejects(self):
        for path in self.paths:
            with self.subTest(file=path.name):
                old = path.stat()
                path.write_bytes(b"other")  # 同样5字节，必须复核mtime而非只有大小。
                os.utime(path, ns=(old.st_atime_ns, old.st_mtime_ns + 1_000_000))
                with self.assertRaisesRegex(ValueError, "input file version changed"):
                    self.cache.validate(self.subject, "lh", "cuda:1")
                path.write_bytes(b"first")
                os.utime(path, ns=(old.st_atime_ns, old.st_mtime_ns))

    def test_deleted_input_is_not_silently_reused(self):
        self.paths[2].unlink()
        with self.assertRaises(FileNotFoundError):
            self.cache.validate(self.subject, "lh", "cuda:1")

    def test_other_worker_scope_rejects(self):
        for subject, hemi, device in ((self.subject / "other", "lh", "cuda:1"),
                                      (self.subject, "rh", "cuda:1"),
                                      (self.subject, "lh", "cuda:0")):
            with self.subTest(subject=subject, hemi=hemi, device=device):
                with self.assertRaisesRegex(ValueError, "another"):
                    self.cache.validate(subject, hemi, device)

    def test_symlink_target_change_rejects(self):
        path = self.paths[0]
        path.unlink()
        first = self.subject / "first.surface"
        second = self.subject / "second.surface"
        first.write_bytes(b"first")
        second.write_bytes(b"first")
        path.symlink_to(first)
        self.cache.signature = tuple(_file_version(p) for p in self.paths)
        path.unlink()
        path.symlink_to(second)
        with self.assertRaisesRegex(ValueError, "input file version changed"):
            self.cache.validate(self.subject, "lh", "cuda:1")


if __name__ == "__main__":
    unittest.main()
