"""The HCP one-level and three-level multivariate MSMAll schedules."""

from dataclasses import asdict, dataclass
from numbers import Integral, Real
from pathlib import Path
import math

from .config import _source_float


@dataclass(frozen=True)
class MSMAllConfig:
    """Discrete vector-feature registration with HCP's strain regularizer.

    There is no rigid stage: ``initial_sphere`` supplies the previous surface
    registration. Floating options retain the upstream float32-to-double
    parsing behavior. These schedules require VN, rescaled labels and the
    triangular multivariate likelihood, with zero feature smoothing.
    """

    simval: tuple[int, ...] = (2, 2, 2)
    iterations: tuple[int, ...] = (10, 15, 15)
    control_grid: tuple[int, ...] = (2, 3, 4)
    sampling_grid: tuple[int, ...] = (4, 5, 6)
    data_grid: tuple[int, ...] = (4, 5, 6)
    regularization: tuple[float, ...] = (0.00001, 0.0075, 0.01)
    shear_modulus: float = 0.4
    bulk_modulus: float = 1.6
    strain_exponent: float = 2.0
    regularization_exponent: float = 2.0

    def __post_init__(self):
        fields = ("simval", "iterations", "control_grid", "sampling_grid",
                  "data_grid", "regularization")
        for name in fields:
            object.__setattr__(self, name, tuple(getattr(self, name)))
        levels = len(self.simval)
        if levels not in (1, 3) or any(len(getattr(self, name)) != levels for name in fields):
            raise ValueError("MSMAll options must contain one or three discrete levels")
        for name in fields[:-1]:
            values = getattr(self, name)
            if any(isinstance(value, bool) or not isinstance(value, Integral) for value in values):
                raise ValueError(f"{name} must contain integers")
            object.__setattr__(self, name, tuple(int(value) for value in values))
        if any(value != 2 for value in self.simval):
            raise ValueError("HCP MSMAll supports multivariate Pearson similarity 2")
        if any(value < 1 for value in self.iterations):
            raise ValueError("iterations must be positive")
        for name in ("control_grid", "sampling_grid", "data_grid"):
            if any(value < 1 or value > 7 for value in getattr(self, name)):
                raise ValueError(f"{name} levels must be between 1 and 7")
        if any(cp > data for cp, data in zip(self.control_grid, self.data_grid)):
            raise ValueError("control_grid must not exceed data_grid")
        values = self.regularization
        if any(isinstance(value, bool) or not isinstance(value, Real) or
               not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("regularization must contain finite nonnegative numbers")
        object.__setattr__(self, "regularization", tuple(
            _source_float(value, "regularization") for value in values))
        for name in ("shear_modulus", "bulk_modulus", "strain_exponent",
                     "regularization_exponent"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
            value = _source_float(value, name)
            if value <= 0:
                raise ValueError(f"{name} must remain positive after float32 option parsing")
            object.__setattr__(self, name, value)

    @classmethod
    def coarse(cls):
        """HCP ``MSMAllStrainFinalconf1to1_1to3_1``."""
        return cls(simval=(2,), iterations=(10,), control_grid=(2,),
                   sampling_grid=(4,), data_grid=(4,), regularization=(0.00001,))

    @classmethod
    def refine(cls):
        """HCP ``MSMAllStrainFinalconf1to1_1to3_2`` (the default)."""
        return cls()

    @classmethod
    def from_file(cls, path):
        mapping = {"simval": ("simval", int), "it": ("iterations", int),
                   "CPgrid": ("control_grid", int), "SGgrid": ("sampling_grid", int),
                   "datagrid": ("data_grid", int), "lambda": ("regularization", float)}
        scalars = {"shearmod": "shear_modulus", "bulkmod": "bulk_modulus",
                   "k_exponent": "strain_exponent", "regexp": "regularization_exponent"}
        options = {}; flags = set(); checked = {}; fixed = set()
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            if not line.startswith("--"):
                raise ValueError(f"invalid MSMAll option: {line}")
            key, sep, value = line[2:].partition("=")
            if key in mapping and sep:
                field, convert = mapping[key]
                options[field] = tuple(convert(item) for item in value.split(","))
            elif key in scalars and sep:
                options[scalars[key]] = float(value)
            elif key in ("opt", "sigma_in", "sigma_ref") and sep:
                checked[key] = value
            elif (key == "dopt" and value == "HOCR") or (key == "regoption" and value == "3"):
                fixed.add(key)
            elif key in ("numthreads", "threads") and sep and int(value) > 0:
                pass
            elif not sep and key in ("VN", "rescaleL", "triclique"):
                flags.add(key)
            else:
                raise ValueError(f"unsupported MSMAll option: {line}")
        if flags != {"VN", "rescaleL", "triclique"}:
            raise ValueError("MSMAll requires --VN, --rescaleL and --triclique")
        # The official one-level file supplies every schedule field. Requiring
        # it avoids silently combining a partial file with a three-level default.
        if not set(field for field, _ in mapping.values()).issubset(options):
            raise ValueError("MSMAll config must provide every discrete schedule field")
        if fixed != {"dopt", "regoption"} or set(checked) != {"opt", "sigma_in", "sigma_ref"}:
            raise ValueError("MSMAll config must explicitly select discrete HOCR, strain and zero smoothing")
        levels = len(options["simval"])
        for name, required in {"opt": ",".join(["DISCRETE"] * levels),
                               "sigma_in": ",".join(["0"] * levels),
                               "sigma_ref": ",".join(["0"] * levels)}.items():
            if checked[name] != required:
                raise ValueError(f"unsupported MSMAll {name}: {checked[name]}")
        return cls(**options)

    def to_dict(self):
        return asdict(self)
