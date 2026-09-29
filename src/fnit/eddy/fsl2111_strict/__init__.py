from .config import FSL2111Config, FSL_EDDY_VERSION, FSL_EDDY_COMMIT

__all__=['FSL2111Config','FSL_EDDY_VERSION','FSL_EDDY_COMMIT','TorchEDDYFSL2111','StrictEDDYResult']


def __getattr__(name):
    if name in ('TorchEDDYFSL2111','StrictEDDYResult'):
        from .pipeline import TorchEDDYFSL2111, StrictEDDYResult
        return {'TorchEDDYFSL2111':TorchEDDYFSL2111,'StrictEDDYResult':StrictEDDYResult}[name]
    raise AttributeError(name)
