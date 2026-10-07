"""Native-free FreeSurfer 8.2 one-feature GCSA cortical annotation."""

from __future__ import annotations

import argparse
import json
import struct
import time
from pathlib import Path

import nibabel as nib
import nibabel.freesurfer.io as fsio
import numpy as np

from .gcsa_aseg import relabel_with_aseg
from .gcsa_feature import mean_curvature_and_principal_directions
from .gcsa_finalize import apply_cortex_label, mode_filter_annotations, ordered_neighbors
from .gcsa_gibbs import GibbsModel
from .gcsa_initial import (
    InitialAtlas, initial_label, map_initial_nodes, read_ico_vertices, read_initial_atlas,
)
from .gcsa_islands import relabel_islands, vertex_areas
from .gcsa_reclassify import reclassify_gibbs


def _file_version(path: Path) -> tuple[str, int, int]:
    """Return an immutable worker-local file identity for cache validation."""
    resolved = path.resolve(strict=True)
    stat = resolved.stat()
    return str(resolved), stat.st_size, stat.st_mtime_ns


class GCSAFeatureCache:
    """Cache geometry-only GCSA inputs for the three atlases of one hemisphere.

    The private hemisphere worker processes the DK, Destrieux and DKT atlases
    serially on the same ``smoothwm``/``sphere.reg`` mesh. This object caches
    only read-only geometry, mapped ico indices, curvature directions,
    adjacency and aseg data. Atlas classifiers, Gibbs models, labels and
    output writers remain per atlas, preserving label order and seeded
    permutation semantics. The cache is local to one worker and rejects a
    different subject/hemi/device or changed input file version.
    """

    def __init__(self, subject: str | Path, hemi: str, *, device: str = "cpu") -> None:
        if hemi not in ("lh", "rh"):
            raise ValueError("hemi must be lh or rh")
        self.subject = Path(subject).resolve()
        self.hemi = hemi
        self.device = str(device)
        surf = self.subject / "surf"
        label = self.subject / "label"
        mri = self.subject / "mri"
        self.smooth_path = surf / f"{hemi}.smoothwm"
        self.sphere_path = surf / f"{hemi}.sphere.reg"
        aseg_path = mri / "aseg.presurf.mgz"
        cortex_path = label / f"{hemi}.cortex.label"
        for path in (self.smooth_path, self.sphere_path, aseg_path, cortex_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        self.signature = tuple(_file_version(path) for path in
                               (self.smooth_path, self.sphere_path, aseg_path, cortex_path))
        self.smooth, self.faces = fsio.read_geometry(str(self.smooth_path))
        self.sphere, sphere_faces = fsio.read_geometry(str(self.sphere_path))
        if not np.array_equal(self.faces, sphere_faces):
            raise ValueError("smoothwm and sphere.reg topology differs")
        self.aseg_image = nib.load(str(aseg_path))
        self.aseg = np.asanyarray(self.aseg_image.dataobj)
        self.tk_to_vox = np.linalg.inv(self.aseg_image.header.get_vox2ras_tkr())
        self.feature, self.principal = mean_curvature_and_principal_directions(
            self.smooth, self.faces, device=device)
        self.neighbors = ordered_neighbors(self.faces, len(self.smooth))
        self.vertex_area = vertex_areas(self.smooth, self.faces)
        self.cortex_vertices = fsio.read_label(str(cortex_path))
        self._mapped: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}

    def map_initial_nodes(self, ico4_file: str | Path,
                          ico7_file: str | Path) -> tuple[np.ndarray, np.ndarray]:
        """Cache sphere-to-ico classifier/prior indices by template versions."""
        ico4, ico7 = Path(ico4_file), Path(ico7_file)
        key = _file_version(ico4), _file_version(ico7)
        if key not in self._mapped:
            self._mapped[key] = map_initial_nodes(
                self.sphere, read_ico_vertices(ico4), read_ico_vertices(ico7))
        return self._mapped[key]

    def validate(self, subject: str | Path, hemi: str, device: str) -> None:
        """Reject accidental reuse across private workers or CUDA devices."""
        if Path(subject).resolve() != self.subject or hemi != self.hemi:
            raise ValueError("GCSA feature cache belongs to another subject/hemi")
        if str(device) != self.device:
            raise ValueError("GCSA feature cache belongs to another device")


def write_annotation(path: str | Path, labels: np.ndarray, atlas: InitialAtlas) -> None:
    """Write the v2 color-table annotation bytes used by ``MRISwriteAnnotation``."""
    nvertices = len(labels)
    vertex_and_label = np.empty((nvertices, 2), dtype=">i4")
    vertex_and_label[:, 0] = np.arange(nvertices)
    vertex_and_label[:, 1] = labels
    source_name = atlas.source_name.encode() + b"\0"
    with Path(path).open("wb") as stream:
        stream.write(struct.pack(">i", nvertices))
        stream.write(vertex_and_label.tobytes())
        stream.write(struct.pack(">iiii", 1, -2, max(atlas.color_table) + 1,
                                 len(source_name)))
        stream.write(source_name)
        stream.write(struct.pack(">i", len(atlas.color_table)))
        for index, (name, red, green, blue, transparency) in sorted(atlas.color_table.items()):
            encoded = name.encode() + b"\0"
            stream.write(struct.pack(">ii", index, len(encoded)))
            stream.write(encoded)
            stream.write(struct.pack(">iiii", red, green, blue, transparency))


def label_surface(subject: str | Path, hemi: str, atlas_file: str | Path,
                  ico4_file: str | Path, ico7_file: str | Path,
                  output_file: str | Path, *, device: str = "cpu",
                  prepared: GCSAFeatureCache | None = None) -> dict:
    """Run the pinned ``mris_ca_label`` sequence from fixed input files.

    ``prepared`` optionally reuses geometry-only inputs for multiple atlases
    on one hemisphere. Atlas parsing, Gibbs state and output writing remain
    per-call; omitting it preserves the original standalone behavior.
    """
    if hemi not in ("lh", "rh"):
        raise ValueError("hemi must be lh or rh")
    subject = Path(subject)
    started = time.perf_counter()
    atlas = read_initial_atlas(atlas_file, include_gibbs=True)
    cache = prepared if prepared is not None else GCSAFeatureCache(subject, hemi, device=device)
    cache.validate(subject, hemi, device)
    smooth, faces = cache.smooth, cache.faces
    classifier, prior = cache.map_initial_nodes(ico4_file, ico7_file)
    aseg, tk_to_vox = cache.aseg, cache.tk_to_vox
    feature, principal = cache.feature, cache.principal
    neighbors = cache.neighbors
    loaded = time.perf_counter()

    annotation = np.empty(len(smooth), dtype=np.int32)
    for vertex in range(len(smooth)):
        annotation[vertex] = initial_label(
            atlas.classifier_nodes[int(classifier[vertex])],
            atlas.prior_nodes[int(prior[vertex])], float(feature[vertex]))[0]
    initially_labeled = time.perf_counter()
    annotation = relabel_with_aseg(annotation, atlas, classifier, prior, feature,
                                   smooth, aseg, tk_to_vox)
    first_aseg = time.perf_counter()

    model = GibbsModel(atlas, classifier, prior, feature, smooth, neighbors,
                       principal, annotation)
    gibbs_history = reclassify_gibbs(model, atlas)
    gibbs = time.perf_counter()
    annotation = relabel_with_aseg(annotation, atlas, classifier, prior, feature,
                                   smooth, aseg, tk_to_vox)
    model.labels = annotation
    second_aseg = time.perf_counter()
    islands_history = relabel_islands(model, cache.vertex_area)
    islands = time.perf_counter()
    annotation = mode_filter_annotations(annotation, faces, atlas.color_table)
    filtered = time.perf_counter()
    annotation = apply_cortex_label(annotation, cache.cortex_vertices, atlas,
                                    classifier, prior, feature, faces)
    corrected = time.perf_counter()
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    write_annotation(output_file, annotation, atlas)
    finished = time.perf_counter()
    return {"vertices": len(annotation), "device": device,
            "gibbs_history": gibbs_history, "islands_history": islands_history,
            "seconds": {"prepare": loaded - started,
                        "initial_classifier": initially_labeled - loaded,
                        "aseg_first": first_aseg - initially_labeled,
                        "gibbs": gibbs - first_aseg,
                        "aseg_second": second_aseg - gibbs,
                        "islands": islands - second_aseg,
                        "mode_filter": filtered - islands,
                        "cortex_label": corrected - filtered,
                        "write": finished - corrected,
                        "total": finished - started}}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=Path, required=True)
    parser.add_argument("--hemi", choices=("lh", "rh"), required=True)
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--ico4", type=Path, required=True)
    parser.add_argument("--ico7", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    result = label_surface(args.subject, args.hemi, args.atlas, args.ico4,
                           args.ico7, args.output, device=args.device)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
