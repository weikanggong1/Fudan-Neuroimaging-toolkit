"""单被试网络标签、时序和连接矩阵的文件输出。"""

from pathlib import Path

import nibabel as nib
import numpy as np


def network_timeseries(series, labels, mask, *, censor=None):
    """将 [T,cortex] 时序按 [64984] 标签平均为 [T,17]，缺失网络为 NaN。

    mask 确定 series 的完整顶点次序；censor 为每帧 0/1，保留值为 1。
    """
    series = np.asarray(series)
    mask = np.asarray(mask, dtype=bool)
    labels = np.asarray(labels)
    if series.ndim != 2 or labels.shape != mask.shape or series.shape[1] != mask.sum():
        raise ValueError("series columns must follow the masked label vertex order")
    if censor is not None:
        censor = np.asarray(censor)
        if censor.shape != (len(series),) or not np.isin(censor, [0, 1]).all():
            raise ValueError("censor must contain one 0/1 value per frame")
        series = series[censor.astype(bool)]
    cortical_labels = labels[mask]
    result = np.full((len(series), 17), np.nan, dtype=np.float64)
    for network in range(1, 18):
        selected = cortical_labels == network
        if selected.any():
            result[:, network - 1] = series[:, selected].mean(axis=1, dtype=np.float64)
    return result


def save_results(output_dir, labels, series, mask, *, censor=None):
    """保存 NPY 标签、cortex-only dlabel 和 17 网络 TSV。"""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "labels_fslr32k_64984.npy", labels)
    np.save(output / "lh_labels.npy", labels[:32492])
    np.save(output / "rh_labels.npy", labels[32492:])
    axes = nib.cifti2.cifti2_axes
    brain = (axes.BrainModelAxis.from_mask(mask[:32492], name="CortexLeft") +
             axes.BrainModelAxis.from_mask(mask[32492:], name="CortexRight"))
    # Names are HCP_40 indices; they do not imply a Yeo2011 naming correspondence.
    table = {0: ("Background", (0., 0., 0., 0.))}
    for network in range(1, 18):
        table[network] = (f"HCP40_Network{network:02d}",
                          ((network * 37 % 255) / 255., (network * 73 % 255) / 255.,
                           (network * 109 % 255) / 255., 1.))
    header = nib.Cifti2Header.from_axes((axes.LabelAxis(["MSHBM17"], [table]), brain))
    nib.save(nib.Cifti2Image(np.asarray(labels[mask], dtype=np.int32)[None], header),
             str(output / "labels_fslr32k.dlabel.nii"))
    timeseries = network_timeseries(series, labels, mask, censor=censor)
    columns = "\t".join(f"HCP40_Network{i:02d}" for i in range(1, 18))
    np.savetxt(output / "network_timeseries.tsv", timeseries, delimiter="\t",
               header=columns, comments="")
    np.savetxt(output / "network_correlation.tsv", np.corrcoef(timeseries.T),
               delimiter="\t", header=columns, comments="")
