# fsLR32k single-subject MS-HBM (17 networks)

CPU-only Python adaptation of the [CBIG Kong2019 MS-HBM single-subject wrapper](https://github.com/ThomasYeoLab/CBIG/tree/master/stable_projects/brain_parcellation/Kong2019_MSHBM). It includes the HCP_40 17-network prior, fsLR32k medial mask, fs_LR_900 seeds, and mesh adjacency, derived from CBIG commit `b69b822a15e2a94f1e439606552fc44b6858cf3c` under the CBIG MIT license. It needs no MATLAB or CBIG installation at runtime. It supports only fsLR32k cortex with the supplied HCP_40 prior.

```bash
fnit-mshbm --timeseries cortical_5min.npy --output-dir out/5min
```

Input is `time × 59412` fsLR32k cortical data (standard left/right CIFTI order), `time × 64984` including medial wall, or fsLR32k `.dtseries.nii`. Vertex × time NumPy arrays are accepted. With one input, uncensored frames are split into two half-length pseudo-sessions as in the CBIG wrapper. With several inputs, each file is a session. `--censor` accepts one matching 0/1 text vector per file. The output contains 64984-vertex labels in CBIG's 1–17 order, with 0 on medial wall, plus hemisphere arrays and convergence/provenance JSON.

CBIG defaults are `--w 200 --c 50`. Each session profile uses fs_LR_900 seed vertices, Pearson correlation, a global top-10% threshold, and unit-length vertex rows. Inference alternates vMF session directions, a shared concentration, a spatial posterior with a mesh MRF, and a subject direction. CPU memory requirements are several GB for full fsLR32k input.

Validation on one held-out MSC02 five-minute sample with the original CBIG MATLAB R2018b code found 2 differing entries among 88,107,996 binarized profile values in a 50-frame pseudo-session. With identical binary profiles as input, the Python and MATLAB inference assigned the same label to all 59,412 cortical vertices. This checks one subject and duration, not universal numerical equivalence or biological validity.
