# TOPUP module

The public single-subject API is `TorchTOPUP`, `prepare_ukb_topup`, and
`run_ukb_topup`. It implements the FSL 6.0.7.4 `b02b0.cnf` AP/PA path in
PyTorch and writes FSL-compatible coefficient, movement, field, corrected
image, and Jacobian files.

See [`docs/topup/README.md`](../../../docs/topup/README.md) for input formats,
CLI and Python calls, output contracts, limitations, benchmark numbers, and
the FSL comparison figure.
