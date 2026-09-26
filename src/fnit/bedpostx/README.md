# BEDPOSTX implementation

`TorchBEDPOSTX` is the single-subject Python API in [`core.py`](core.py). It reads FSL-style DWI inputs and writes posterior samples for probabilistic tractography. Use `fnit bedpostx --subject-dir SUBJECT --device cuda:0` or `fnit-bedpostx` for the CLI. Full input/output and model details: [BEDPOSTX guide](../../../docs/bedpostx/README.md).
