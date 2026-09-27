# BEDPOSTX validation

[Current machine-readable summary](report.public.json) contains matched gpucw1 original FSL versus the current TorchBEDPOSTX code, CPU/GPU wall times, output consistency, chain-stability controls, and FSL `probtrackx2` interoperability. The [BEDPOSTX guide](../../docs/bedpostx/README.md) explains the methods and limits. The [synthetic DWI generator and figure script](../../docs/bedpostx/synthetic_example.py) reproduces the public [example image](../../docs/bedpostx/synthetic_example.png).

Only aggregate statistics from the UK Biobank single-subject ROI are public. Raw DWI, posterior maps, and the real-data image remain on the research server. The 14-voxel real ROI is a file-contract and numerical check; it is not whole-brain validation.
