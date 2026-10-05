"""Plot the first real synthetic-label objective correction."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


root = Path(__file__).resolve().parent
figure, axes = plt.subplots(3, 2, figsize=(7.2, 9.2), constrained_layout=True)
for row, (key, title) in enumerate((("thalamus", "Thalamus"),
                                  ("ha_left", "Left hippocampus / amygdala"),
                                  ("ha_right", "Right hippocampus / amygdala"))):
    data = np.load(root / (key + "_figure_slice.public.npz"))
    report = json.loads((root / (key + "_comparison.public.json")).read_text())
    observation = data["observation"]
    correction = data["correction"]
    nonzero = np.argwhere(observation != 0)
    low = np.maximum(nonzero.min(0)-4,0)
    high = np.minimum(nonzero.max(0)+5,observation.shape)
    crop = tuple(slice(int(a),int(b)) for a,b in zip(low,high))
    observation, correction = observation[crop], correction[crop]
    for axis in axes[row]:
        axis.imshow(observation.T, origin="lower", cmap="gray", vmin=0, vmax=4)
        axis.set_axis_off()
    shown = axes[row, 1].imshow(np.ma.masked_where(correction <= .001, correction).T,
        origin="lower", cmap="inferno", vmin=0, vmax=19, interpolation="nearest")
    axes[row, 0].set_title(title+"\nFixed coarse-label observation", fontsize=10)
    axes[row, 1].set_title("First objective: old minus epsilon\n" +
        f"{report['epsilon_effect']['total_data_cost_reduction']:.2f} across all valid voxels", fontsize=10)
figure.colorbar(shown, ax=axes[:, 1], shrink=.65, label="Point negative log-likelihood reduction")
figure.suptitle("Same real mesh / image / alpha, before any mesh update", fontsize=12)
figure.savefig(root / "first_objective_epsilon.png", dpi=160)
plt.close(figure)
