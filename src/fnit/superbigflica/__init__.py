"""Supervised multimodal components and continuous/categorical prediction."""

from .pipeline import apply_model, run_superbigflica
from .plotting import plot_superbigflica

__all__ = ['apply_model', 'run_superbigflica', 'plot_superbigflica']

FNIT_superbigflica = run_superbigflica
__all__.append('FNIT_superbigflica')
