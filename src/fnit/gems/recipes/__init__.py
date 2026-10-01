"""Structure-specific GEMS recipes."""

from .brainstem import BrainstemRecipe
from .thalamus import ThalamusRecipe
from .hippo_amygdala import HippoAmygdalaRecipe


def make_recipe(name, root):
    if name == "brainstem":
        return BrainstemRecipe(root / name)
    if name == "thalamus":
        return ThalamusRecipe(name, root / name)
    if name in ("hippo-amygdala-left", "hippo-amygdala-right"):
        return HippoAmygdalaRecipe(name.rsplit("-", 1)[-1], root / name)
    raise ValueError(f"Unknown subregion recipe: {name}")


__all__ = ["BrainstemRecipe", "ThalamusRecipe", "HippoAmygdalaRecipe", "make_recipe"]
