"""Write cortical ROI statistics from Python reconstruction outputs."""

from pathlib import Path

from .anatomical_stats_global import anatomical_stats_global_lines
from .anatomical_stats_rows import anatomical_stats_rows
from .surface_stats_cache import SurfaceStatsCache


def write_anatomical_stats(subject: str | Path, hemi: str, atlas: str,
                           surface: str, brainvol_stats: dict[str, float],
                           output: str | Path, *, device: str = "cuda:0",
                           cache: SurfaceStatsCache | None = None) -> Path:
    """写出十列脑区统计与全局表头，返回实际输出 Path。

    subject 为 FNIT 被试目录；hemi 为 lh/rh；atlas 为注释文件名中图谱；
    surface 为 white/pial；brainvol_stats 是 mm³ 的全脑体积字典；output
    为 .stats 路径。device 默认 cuda:0，cache 可复用同设备/半球缓存。
    输入表面为 surface RAS/mm，morph 与注释须对应原有顶点顺序。
    路径缺失、无效半球/表面、拓扑或设备不一致抛异常。写入及父目录
    创建均包含在本函数内；调用方计时应覆盖整个调用。对应
    mris_anatomical_stats -no-th3；TH3 .volume 顶点图不作为此表输入。
    """
    if cache is None:
        with SurfaceStatsCache(device=device) as temporary:
            return write_anatomical_stats(
                subject=subject, hemi=hemi, atlas=atlas, surface=surface,
                brainvol_stats=brainvol_stats, output=output, device=device,
                cache=temporary)
    cache.check_device(device)
    if hemi not in {"lh", "rh"} or surface not in {"white", "pial"}:
        raise ValueError("Expected lh/rh hemisphere and white/pial surface")
    subject = Path(subject)
    prefix = subject / "surf" / hemi
    label = subject / "label"
    annotation = label / f"{hemi}.{atlas}.annot"
    cortex = None if atlas.startswith("BA_exvivo") else label / f"{hemi}.cortex.label"
    area = Path(str(prefix) + (".area.pial" if surface == "pial" else ".area"))
    thickness = Path(str(prefix) + ".thickness")
    global_lines = anatomical_stats_global_lines(
        area, thickness, annotation, cortex, brainvol_stats,
        subject / "mri" / "transforms" / "talairach.xfm", surface=surface,
        cache=cache)
    rows = anatomical_stats_rows(
        Path(str(prefix) + ".white"), Path(str(prefix) + ".pial"),
        Path(str(prefix) + "." + surface), area, thickness, annotation,
        cortex, device=device, cache=cache)
    lines = ["# Table of FreeSurfer cortical parcellation anatomical statistics",
             "# generating_program fnit",
             f"# subjectname {subject.name}", f"# hemi {hemi}",
             f"# AnnotationFile {annotation}", *global_lines,
             "# NTableCols 10",
             "# ColHeaders StructName NumVert SurfArea GrayVol ThickAvg ThickStd MeanCurv GausCurv FoldInd CurvInd",
             *rows]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")
    return output
