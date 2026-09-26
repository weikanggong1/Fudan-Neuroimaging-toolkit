# PyTorch structural connectomes

`UKBConnectome` computes four region × region matrices from one **already corrected**
4D DWI, its b-values and eddy-rotated b-vectors, and a paired T1w image. Diffusion
fitting, probabilistic tracking, SIFT2-style weighting and endpoint assignment use
PyTorch tensors on the selected device. The DWI preprocessing that precedes
[`UKB-connectomics`](https://github.com/sina-mansour/UKB-connectomics) tractography
is outside this call.

```python
from fnit.connectome import UKBConnectome

result = UKBConnectome(device="cuda:0")(
    "sub-01_desc-preproc_dwi.nii.gz", "sub-01_dwi.bval",
    "sub-01_desc-eddyRotated_dwi.bvec", "sub-01_T1w.nii.gz",
    atlas_dwi="sub-01_space-dwi_atlas.nii.gz", seed=0,
)
count = result.matrices["count"]
```

The four keys are `count`, `sift2_fbc`, `mean_length` (mm), and `mean_fa`.
Without `atlas_dwi`, the result uses a compact atlas from SynthSeg anatomical
labels, which is not the original UKB cortical plus Tian parcellation. This
implementation approximates the original FOD estimation, tracking, and SIFT2
steps; only the matrix assignment stage currently has a matched-input MRtrix
benchmark. See [API, CLI, outputs and validation](../../../docs/connectome/README.md).
