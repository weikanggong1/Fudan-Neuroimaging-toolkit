"""The four-level HCP/sMRIPrep MSMSulc schedule supported by FNIT."""

from dataclasses import dataclass, asdict
from pathlib import Path
import math
from numbers import Integral, Real
import struct


def _source_float(value, name):
    """Match newMSM's Option<float> followed by its double-valued parameter.

    Coordinates and costs remain float64. This conversion only reproduces
    the upstream configuration parser, including the default option values.
    """
    try:
        result = struct.unpack("f", struct.pack("f", value))[0]
    except (OverflowError, struct.error) as exc:
        raise ValueError(f"{name} is outside the finite float32 option range") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} is outside the finite float32 option range")
    return result


@dataclass(frozen=True)
class MSMSulcConfig:
    """Keep the registration schedule explicit instead of shortening levels.

    The first similarity value is the rigid initialization. newMSM maps the
    historical value 3 (NMI) to 2 (Pearson). With one sulcal feature the rigid
    pairwise Pearson term is the sign of globally centered values, followed
    by spatial WLS smoothing. ``ssd_affine`` selects the
    SSD initialization used by some older newMSM benchmark configurations.
    Floating options use the official float32-parser-to-float64 conversion;
    ``to_dict`` reports the effective values used in the calculation.
    """

    simval: tuple[int, ...] = (3, 2, 2, 2)
    iterations: tuple[int, ...] = (50, 10, 15, 15)
    control_grid: tuple[int, ...] = (6, 2, 3, 4)
    sampling_grid: tuple[int, ...] = (6, 4, 5, 6)
    data_grid: tuple[int, ...] = (6, 4, 5, 6)
    regularization: tuple[float, ...] = (0.0, 10.0, 7.5, 7.5)
    affine_step_size: float = 0.01
    affine_gradient_spacing: float = 0.5
    shear_modulus: float = 0.4
    bulk_modulus: float = 1.6
    strain_exponent: float = 2.0
    regularization_exponent: float = 2.0

    def __post_init__(self):
        for name in ("simval", "iterations", "control_grid", "sampling_grid",
                     "data_grid", "regularization"):
            value = tuple(getattr(self, name))
            object.__setattr__(self, name, value)
            if len(value) != 4:
                raise ValueError(f"{name} must contain the affine and three discrete levels")
            if name != "regularization":
                if any(isinstance(v,bool) or not isinstance(v,Integral) for v in value):
                    raise ValueError(f"{name} must contain integers")
                object.__setattr__(self,name,tuple(int(v) for v in value))
            elif any(not isinstance(v,Real) or not math.isfinite(v) for v in value):
                raise ValueError("regularization must contain finite numbers")
            else:
                object.__setattr__(self,name,tuple(_source_float(v,name) for v in value))
        if self.simval[0] not in (1, 2, 3) or any(v not in (1, 2) for v in self.simval[1:]):
            raise ValueError("MSMSulc supports affine similarity 1/2/3 and discrete similarity 1/2")
        if any(v <= 0 for v in self.iterations):
            raise ValueError("iterations must be positive")
        for name in ("control_grid", "sampling_grid", "data_grid"):
            if any(v < 1 or v > 7 for v in getattr(self, name)):
                raise ValueError(f"{name} levels must be between 1 and 7")
        if any(cp>data for cp,data in zip(self.control_grid[1:],self.data_grid[1:])):
            raise ValueError("discrete control_grid must not exceed data_grid")
        if any(v < 0 for v in self.regularization) or self.regularization[0] != 0:
            raise ValueError("regularization must be nonnegative, with zero for affine")
        for name in ("affine_step_size", "affine_gradient_spacing", "shear_modulus",
                     "bulk_modulus", "strain_exponent", "regularization_exponent"):
            value=getattr(self,name)
            if not isinstance(value,Real) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive")
            value=_source_float(value,name)
            if value<=0:
                raise ValueError(f"{name} must remain positive after float32 option parsing")
            object.__setattr__(self,name,value)

    @classmethod
    def ssd_affine(cls):
        """The same schedule with the source newMSM SSD rigid initialization."""
        return cls(simval=(1, 2, 2, 2))

    @classmethod
    def from_file(cls, path):
        """Read supported official MSMSulc options and reject changed algorithms."""
        mapping = {"simval": ("simval", int), "it": ("iterations", int),
                   "CPgrid": ("control_grid", int), "SGgrid": ("sampling_grid", int),
                   "datagrid": ("data_grid", int), "lambda": ("regularization", float)}
        scalars = {"stepsize": "affine_step_size", "gradsampling": "affine_gradient_spacing",
                   "shearmod": "shear_modulus", "bulkmod": "bulk_modulus",
                   "k_exponent": "strain_exponent", "regexp": "regularization_exponent"}
        expected = {"opt": "AFFINE,DISCRETE,DISCRETE,DISCRETE", "dopt": "HOCR",
                    "regoption": "3", "sigma_in": "0,0,0,0", "sigma_ref": "0,0,0,0"}
        options = {}
        flags = set()
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            if not line.startswith("--"):
                raise ValueError(f"invalid MSMSulc option: {line}")
            key, sep, value = line[2:].partition("=")
            if key in mapping and sep:
                field, convert = mapping[key]
                options[field] = tuple(convert(item) for item in value.split(","))
            elif key in scalars and sep:
                options[scalars[key]] = float(value)
            elif key in expected and value == expected[key]:
                pass
            elif key == "opt" and value == "RIGID,DISCRETE,DISCRETE,DISCRETE":
                pass
            elif key in ("numthreads", "threads") and sep and int(value)>0:
                # Host thread count is an execution option, not a different
                # scientific schedule. FNIT uses its selected PyTorch device.
                pass
            elif not sep and key in ("VN", "rescaleL", "triclique"):
                flags.add(key)
            else:
                raise ValueError(f"unsupported MSMSulc option: {line}")
        if flags != {"VN", "rescaleL", "triclique"}:
            raise ValueError("MSMSulc requires --VN, --rescaleL and --triclique")
        return cls(**options)

    def to_dict(self):
        return asdict(self)
