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

    def test_wrn_templates_are_pinned_upstream_without_release_claim(self):
        self.assertEqual(len(assets.MSMALL_LOW_DIM_ASSETS), 15)
        self.assertEqual(sum(assets.ASSET_SIZES[path] for path, _ in
                             assets.MSMALL_LOW_DIM_ASSETS), 86180368)
        for dimension, (path, digest) in enumerate(assets.MSMALL_LOW_DIM_ASSETS, 7):
            self.assertIn(f".ica_d{dimension}_ROW_vn/", path)
            self.assertNotIn(path, assets.RELEASE_CHECKSUMS)
            self.assertEqual(len(digest), 64)

    def test_size_mismatch_is_rejected_even_when_digest_matches(self):
        relative_path = "global/templates/example.gii"
        payload = b"valid digest but truncated size"
        digest = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.dict(assets.ASSET_SIZES, {relative_path: len(payload) + 1}):
                with self.assertRaisesRegex(ValueError, "Could not download verified asset"):
                    assets._install_one(root, relative_path, digest,
                                        lambda url, timeout: io.BytesIO(payload))
            self.assertFalse((root / relative_path).exists())

    def test_msmall_installs_complete_wrn_template_set(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(assets, "_install_one") as install:
                assets.main(["--output-dir", directory, "--msmall"])
        installed = [call.args[1] for call in install.call_args_list]
        self.assertEqual(installed, [path for path, _ in
                                    assets.ASSETS + assets.MSMALL_ASSETS + assets.MSMALL_LOW_DIM_ASSETS])


if __name__ == "__main__":
    unittest.main()
