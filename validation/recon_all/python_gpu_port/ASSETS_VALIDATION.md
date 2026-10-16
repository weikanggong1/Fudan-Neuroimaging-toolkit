# FreeSurfer 8.2 fixed-T1 data assets

The Python [asset installer](../../../src/fnit/recon_all/assets.py)
stores verified non-model data outside the code package. The scope is the
reference single-T1 `recon-all` command in `work/reference_recon_all.cmd`,
including bilateral BA/exvivo labels. That command and its `mri_label2label`
log require 72 `fsaverage` labels and two sphere surfaces. The installed
FreeSurfer 8.2.0-1 reference sizes and hashes come from
`work/scoped_manifest.json`; the two later-discovered `mris_ca_label` ico
meshes were checked against the pinned native bundle directly.

| Data class | Files | Installed bytes | Download status |
| --- | ---: | ---: | --- |
| `average/` classifiers, priors, color tables | 18 | 284,049,476 | Verified |
| Root segmentation/statistics lookup tables | 4 | 145,484 | Verified |
| MNI152 registration targets inside atlas archive | 4 | 25,726,407 | Verified via archive |
| `fsaverage` sphere surfaces | 2 | 11,797,092 | Verified |
| `fsaverage` labels | 54 | 8,076,270 | Verified direct files |
| `fsaverage` labels | 18 | 1,323,221 | Verified via archive |
| `lib/bem/` GCS classifier and prior ico meshes | 2 | 13,575,646 | Verified direct files |
| **Fixed-profile data inventory** | **102** | **344,693,596** | **102 verified** |

The 102 verified files total **344,693,596 installed bytes (328.73 MiB)**.
Nineteen files (309,419,696 bytes) use official FreeSurfer git-annex
objects; seven (148,002 bytes) use the pinned
[FreeSurfer source tree](https://github.com/freesurfer/freesurfer/tree/d932c45b7941662ea380a05efef580568b98d41a/distribution).
The two sphere surfaces share one 5,898,546-byte annex object. Full GET,
size, and SHA-256 checks matched every registered direct source.
The additional `lib/bem/ic4.tri` and `ic7.tri` meshes are loaded by
`GCSAalloc` for all six `mris_ca_label` calls. Their direct official annex
objects matched the pinned bundle exactly: 207,432 bytes /
`95cc8e50f48df9b2a7a8558c9837becfa486f7562f2d9ae6fee4cc7aad341d13`
and 13,368,214 bytes /
`cdd2761f0921a05d4959eb5b200fc3c65c0344a575809b0b09458299d90a1e6d`.
The updated installer passed `--verify-only` for both files.

Four MNI152 files (25,726,407 installed bytes) match members of the official
`mni_icbm152_nlin_asym_09c.tar.gz` git-annex object. The complete downloaded
archive matched **514,649,342 bytes** and SHA-256
`29f8b3dec88feaa133c65ee9342fd9d875cac4e9c08e43a7537cd5d614b227d8`.
Each extracted member separately matched the reference size and SHA-256.
The archive requires a **490.81 MiB transfer** for these 24.53 MiB of used
files. Individually guessed annex member URLs returned 404; a smaller verified
official source has not been established. The installer verifies the archive
before extracting only the four allowlisted members and deletes the archive
afterward.

Fifty-four labels (8,076,270 bytes) matched complete GETs from the
[official legacy `fsaverage` directory](https://www.freesurfer.net/pub/dist/freesurfer/tutorial_versions_centos6/freesurfer/subjects/fsaverage/label/).
They match the FreeSurfer 8.2 reference byte for byte despite the older
directory. The installer requests this host with a browser user agent because
requests without one returned 403. An actual installer download followed by
`--verify-only` passed for `lh.BA1_exvivo.label`.

The remaining 18 labels (1,323,221 extracted bytes) matched members of the
official `distribution/average/fsaverage.tar.gz` from the same pinned source
commit. Its git-annex key gives **320,193,429 bytes** and SHA-256
`586cbe3513db2872ad885486a042ebbde1cb5ca66dd3255994e8736901ce147f`;
a complete GET matched both. All 18 members individually matched the installed
FreeSurfer 8.2 reference size and SHA-256. The installer extracts only these
allowlisted members, verifies each, and deletes the archive. This adds a
**305.36 MiB first-transfer cost** for 1.262 MiB of needed labels. The two
official archives together require **834,842,771 network bytes (796.17 MiB)**.
No FreeSurfer binaries, license file, or reference-installation files are
copied or redistributed. `PENDING_FILES` is empty for this fixed profile.

The old individual-file search returned 404 for the 16 `.mpm.vpnl.label`
files and 403 for the two right perirhinal files. The verified archive
provides an exact official source without repackaging these data. A smaller
standalone redistribution archive was not made; it would require a separate
license and attribution review, including any applicable third-party terms.

The original scoped native-bundle inventory counted 106 non-model files and
missed these two GCS ico meshes. Six of the original 106 are
FreeSurfer shell/runtime configuration or stamp files:
`FreeSurferEnv.csh`, `SetUpFreeSurfer.csh`, `sources.csh`, `build-stamp.txt`,
`etc/global-expert-options.v8.txt`, and `etc/recon-config.yaml`. They are not
image/template data for users to download into the Python asset directory.
Their algorithmic settings still need review before end-to-end parity.
Personal license files, native executables, and model weights are excluded.

The installer provides the complete **fixed-profile data asset set** without
a native FreeSurfer runtime. This is an asset gate, not an end-to-end
reconstruction parity claim. The focused asset suite passed **17/17** on
`control-host` (previously 13/13 before the final 18 labels), including URL routing,
hash rejection, and both archive extraction paths. Actual archive extraction
into scratch passed `--verify-only` for all four MNI members and all 18
`fsaverage` members; two sphere surfaces and one direct label also passed
after installation. The original 100 entries matched the reference manifest's
size and SHA-256; the two new ico meshes matched both their downloaded annex
objects and the pinned native bundle. Full reconstruction was not rerun.
