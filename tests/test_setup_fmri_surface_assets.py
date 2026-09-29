"""Checks for the pinned public HCP surface-template installer."""

import hashlib
import importlib.util
import io
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path


_MODULE_PATH = Path(__file__).resolve().parents[1] / "src/fnit/fmri/assets_setup.py"
_SPEC = importlib.util.spec_from_file_location("fmri_assets_setup", _MODULE_PATH)
assets = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(assets)


class TestSurfaceAssets(unittest.TestCase):
    def test_release_is_preferred_for_pinned_hcp_asset(self):
        relative_path = "global/templates/example.gii"
        payload = b"verified release template"
        digest = hashlib.sha256(payload).hexdigest()
        requested = []

        def opener(url, timeout):
            requested.append(url)
            return io.BytesIO(payload)

        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(assets.RELEASE_CHECKSUMS, {relative_path: digest}):
                path = assets._install_one(Path(directory), relative_path, digest, opener)
            self.assertEqual(path.read_bytes(), payload)
        self.assertEqual(requested,
                         [assets.RELEASE_BASE + "hcp--global--templates--example.gii"])

    def test_download_verifies_checksum_and_reuses_valid_file(self):
        payload = b"public HCP template"
        calls = []

        def opener(url, timeout):
            calls.append((url, timeout))
            return io.BytesIO(payload)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            digest = hashlib.sha256(payload).hexdigest()
            path = assets._install_one(root, "global/templates/example.gii", digest, opener)
            self.assertEqual(path.read_bytes(), payload)
            self.assertEqual(assets._install_one(root, "global/templates/example.gii", digest, opener), path)
        self.assertEqual(calls, [(assets.BASE_URL + "global/templates/example.gii", 60)])

    def test_fallback_uses_same_commit_and_verifies_checksum(self):
        payload = b"same file from CDN"
        calls = []

        def opener(url, timeout):
            calls.append(url)
            if url.startswith(assets.BASE_URL):
                raise OSError("primary unavailable")
            return io.BytesIO(payload)

        with tempfile.TemporaryDirectory() as directory:
            path = assets._install_one(
                Path(directory), "global/templates/example.gii", hashlib.sha256(payload).hexdigest(), opener
            )
            self.assertEqual(path.read_bytes(), payload)
        self.assertEqual(calls, [assets.BASE_URL + "global/templates/example.gii", assets.FALLBACK_URL + "global/templates/example.gii"])

    def test_bad_download_is_not_published(self):
        def opener(url, timeout):
            return io.BytesIO(b"incorrect")

        relative_path = "global/templates/example.gii"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "Could not download verified asset"):
                assets._install_one(root, relative_path, hashlib.sha256(b"correct").hexdigest(), opener)
            self.assertFalse((root / relative_path).exists())
            self.assertEqual(list((root / "global/templates").iterdir()), [])


if __name__ == "__main__":
    unittest.main()
