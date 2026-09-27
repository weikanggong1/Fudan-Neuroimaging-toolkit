# dMRI pipeline module

The public single-subject API is DMRIPipeline. It runs optional TOPUP, EDDY,
DTIFIT, AMICO-NODDI, then a TBSS/FNIRT or T1-plus-tensor MMORF registration
branch. Both branches write the same nine standard-space parameter maps. See
docs/dmri_pipeline/README.md for inputs, outputs, Python and CLI calls,
official UKB commands, and validation.
