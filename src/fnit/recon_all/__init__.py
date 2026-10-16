"""Python reconstruction and validation against FreeSurfer 8.2."""

from .batch import run_recon_all_python_batch

def __getattr__(name):
    if name == "FNIT_recon_all":
        from .native_free import run_recon_all_python
        return run_recon_all_python
    raise AttributeError(name)


__all__ = ["run_recon_all_python_batch", "FNIT_recon_all"]
