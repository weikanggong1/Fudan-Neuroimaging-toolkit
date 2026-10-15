"""持久半球调度与 GCA 上下文的调用合同；不是影像精度 benchmark。"""

import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from fnit.recon_all.hemisphere_parallel import HemisphereGroupError
from fnit.recon_all.hemisphere_pipeline import run_hemisphere_pipeline
from fnit.recon_all import mri_em_register_python as registration
from fnit.recon_all.native_free import _run_defects_volume
from fnit.recon_all.mri_em_register import GCA, GCARegistrationContext


WORKER = '''\
import time
from pathlib import Path

def execute(subject, hemi, device, threads, operation):
    subject = Path(subject)
    if operation == "surface":
        if hemi == "lh":
            time.sleep(.25)  # 让右侧先有机会完成，检验发布仍按左到右。
        (subject / "surf" / f"{hemi}.orig").write_bytes(hemi.encode())
        if hemi == "lh" and (subject / "mri" / "trigger-failure").exists():
            raise RuntimeError("left surface failed after private write")
    elif operation == "register":
        (subject / "surf" / f"{hemi}.sphere.reg").write_bytes(hemi.encode())
    return {"hemi": hemi, "operation": operation, "threads": threads}
'''


def _subject_with_worker(tmp_path):
    subject = tmp_path / "subject"
    for name in ("mri", "surf", "label", "stats", "scripts"):
        (subject / name).mkdir(parents=True)
    (tmp_path / "persistent_test_target.py").write_text(WORKER)
    return subject


def _run(subject, hook=None):
    return run_hemisphere_pipeline(
        subject=subject, operations=["surface", "register"], device="cpu", threads=4,
        callable_path="persistent_test_target:execute", stage_hook=hook,
        startup_wait_seconds=90,
    )


class PersistentPipelineContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.subject = _subject_with_worker(self.root)
        environment = patch.dict(os.environ, {
            "PYTHONPATH": str(self.root) + os.pathsep + os.environ.get("PYTHONPATH", ""),
        })
        environment.start()
        self.addCleanup(environment.stop)

    def test_surface_defects_hook_reads_left_before_right(self):
        shared = self.subject / "mri/surface.defects.mgz"
        (self.subject / "mri/orig.mgz").write_bytes(b"orig")
        for hemi in ("lh", "rh"):
            (self.subject / "surf" / f"{hemi}.defect_labels").write_bytes(b"labels")
            (self.subject / "surf" / f"{hemi}.orig.nofix").write_bytes(b"surface")
            (self.subject / "label" / f"{hemi}.nofix.cortex.label").write_bytes(b"cortex")
        observations = []

        def project_defects(*, template_file, output_file, merge, offset, **_kwargs):
            before = Path(template_file).read_bytes()
            observations.append((before, merge, offset))
            Path(output_file).write_bytes((before if merge else b"") +
                                          (b"rh" if merge else b"lh"))

        def publish_defects(operation, hemi, _value, _row):
            if operation == "surface":
                _run_defects_volume(
                    binary=None, subject=self.subject, hemi=hemi, assets=self.subject,
                    backend="torch", device="cpu",
                )

        with patch("fnit.recon_all.defects_label_volume_torch.defects_to_volume",
                   side_effect=project_defects):
            report = _run(self.subject, publish_defects)
        self.assertEqual(report["status"], "complete")
        self.assertEqual(observations, [(b"orig", False, 1000), (b"lh", True, 2000)])
        self.assertEqual(shared.read_bytes(), b"lhrh")
        surface_events = [row["hemi"] for row in report["publication_events"]
                          if row["operation"] == "surface"]
        self.assertEqual(surface_events, ["lh", "rh"])
        for hemi in ("lh", "rh"):
            self.assertEqual((self.subject / "surf" / f"{hemi}.orig").read_bytes(), hemi.encode())
            self.assertEqual((self.subject / "surf" / f"{hemi}.sphere.reg").read_bytes(), hemi.encode())

    def test_worker_failure_never_publishes_group_success(self):
        (self.subject / "mri/trigger-failure").write_bytes(b"1")
        with self.assertRaises(HemisphereGroupError) as caught:
            _run(self.subject)
        report = caught.exception.report
        persisted = json.loads((self.subject / "scripts/pipeline.hemisphere-group.json").read_text())
        self.assertEqual(report["status"], persisted["status"])
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["stages"]["surface"]["lh"]["status"], "failed")
        self.assertFalse((self.subject / "surf/lh.orig").exists())
        self.assertFalse((self.subject / "surf/rh.orig").exists())
        self.assertFalse((self.subject / "mri/surface.defects.mgz").exists())
        self.assertFalse(any(row["operation"] == "register" for row in report["publication_events"]))


def _register_with_recorded_context(tmp_path, *, reuse):
    atlas = object()
    cached = object()
    source = object()
    stable = SimpleNamespace(labels=np.array([2], np.int32))
    final = SimpleNamespace(labels=np.array([2, 2], np.int32))
    matrix = np.eye(4, dtype=np.float32)
    contexts = []

    def samples(which):
        def get_samples(received_atlas, *, context=None):
            assert received_atlas is atlas
            contexts.append((which, context))
            return stable if which == "stable" else final
        return get_samples

    def mean_volume(received_atlas, *, context=None):
        assert received_atlas is atlas
        contexts.append(("mean", context))
        return None

    def objective(received_atlas, received_samples, received_source, *, clamp, context=None):
        assert received_atlas is atlas and received_samples is stable and received_source is source
        contexts.append(("objective", context))
        return SimpleNamespace(samples=received_samples, clamp=clamp)

    def align(_objective, received_matrix, *, tolerance):
        np.testing.assert_array_equal(received_matrix, matrix)
        return matrix.copy(), {"passes": [], "outer_cost_after": 0.0, "tolerance": tolerance}

    with patch.object(registration, "read_gca", return_value=atlas), \
         patch.object(registration, "GCARegistrationContext", return_value=cached) as construct, \
         patch.object(registration, "read_masked_input", return_value=object()), \
         patch.object(registration, "estimate_image_white_matter_peak", return_value=(1, None, None)), \
         patch.object(registration, "atlas_label_peak", return_value=1), \
         patch.object(registration, "scale_input_intensity", return_value=source), \
         patch.object(registration, "find_stable_samples", side_effect=samples("stable")), \
         patch.object(registration, "find_all_samples", side_effect=samples("all")), \
         patch.object(registration, "gca_mean_volume", side_effect=mean_volume), \
         patch.object(registration, "gca_centroid", return_value=None), \
         patch.object(registration, "find_optimal_translation_source",
                      return_value=(matrix, [(0, (0, 0, 0), matrix)])), \
         patch.object(registration, "find_optimal_linear_transform_source", return_value=(matrix, [])), \
         patch.object(registration, "EMObjective", side_effect=objective), \
         patch.object(registration, "align_em", side_effect=align), \
         patch.object(registration, "write_voxel_lta") as write:
        result = registration.register_t1(
            nu_path=tmp_path / "nu.mgz", atlas_path=tmp_path / "fixed.gca",
            mask_path=tmp_path / "brainmask.mgz", output_path=tmp_path / "talairach.lta",
            device="cpu", search_backend="cpu", reuse_atlas_context=reuse,
        )
    assert construct.call_count == int(reuse)
    if reuse:
        construct.assert_called_once_with(atlas)
    assert contexts == [(name, cached if reuse else None)
                        for name in ("stable", "mean", "objective", "all")]
    np.testing.assert_array_equal(write.call_args.args[1], matrix)
    return result


class GCAContextContractTests(unittest.TestCase):
    def test_flag_changes_reuse_only_not_registration_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = _register_with_recorded_context(root, reuse=False)
            cached = _register_with_recorded_context(root, reuse=True)
        self.assertEqual(baseline["matrix"], cached["matrix"])
        self.assertEqual(baseline["em_complete_protocol"], cached["em_complete_protocol"])
        self.assertIs(baseline["reuse_atlas_context"], False)
        self.assertIs(cached["reuse_atlas_context"], True)

    def test_context_rejects_modified_or_different_atlas(self):
        atlas = GCA(
            prior_spacing=1., node_spacing=1., prior_shape=(1, 1, 1), node_shape=(1, 1, 1),
            node_offsets=np.array([0, 1], np.int32), node_training=np.array([10], np.int32),
            node_labels=np.array([2], np.int32), node_means=np.array([100.], np.float32),
            node_variances=np.array([4.], np.float32),
            prior_offsets=np.array([0, 1], np.int32), prior_training=np.array([10], np.int32),
            prior_labels=np.array([2], np.int32), prior_values=np.array([1.], np.float32),
            atlas_type=0, volume_shape=(1, 1, 1), direction_cosines=tuple(np.eye(3).ravel()),
            center_ras=(0., 0., 0.), voxel_sizes=(1., 1., 1.),
        )
        context = GCARegistrationContext(atlas)
        context.validate(atlas)
        self.assertFalse(context.classifiers.flags.writeable)
        self.assertFalse(context.covariance.variances.flags.writeable)
        with self.assertRaisesRegex(ValueError, "different or modified atlas"):
            context.validate(GCA(**vars(atlas)))
        atlas.node_means[0] = 101.
        with self.assertRaisesRegex(ValueError, "different or modified atlas"):
            context.validate(atlas)
