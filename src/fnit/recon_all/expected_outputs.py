"""单 T1 固定 138 项输出清单；运行完整性和比较器共用。"""

VOLUMES = (
    "mri/T1.mgz", "mri/antsdn.brain.mgz", "mri/aparc+aseg.mgz",
    "mri/aparc.DKTatlas+aseg.mgz", "mri/aparc.a2009s+aseg.mgz",
    "mri/aseg.auto.mgz", "mri/aseg.mgz", "mri/aseg.presurf.hypos.mgz",
    "mri/aseg.presurf.mgz", "mri/brain.finalsurfs.manedit.mgz",
    "mri/brain.finalsurfs.mgz", "mri/brain.mgz", "mri/brainmask.mgz",
    "mri/ctrl_pts.mgz", "mri/entowm.mgz", "mri/filled.auto.mgz",
    "mri/filled.mgz", "mri/lh.ribbon.mgz", "mri/mca-dura.mgz",
    "mri/mrisps.white.mgz", "mri/mrisps.wpa.mgz", "mri/norm.mgz",
    "mri/nu.mgz", "mri/orig.mgz", "mri/orig/001.mgz", "mri/rawavg.mgz",
    "mri/rh.ribbon.mgz", "mri/ribbon.mgz", "mri/surface.defects.mgz",
    "mri/synthseg.rca.mgz", "mri/synthstrip.mgz",
    "mri/transforms/synthmorph.1.0mm.1.0mm/test.nii.gz",
    "mri/transforms/synthmorph.1.0mm.1.0mm/warp.to.mni152.1.0mm.1.0mm.inv.nii.gz",
    "mri/transforms/synthmorph.1.0mm.1.0mm/warp.to.mni152.1.0mm.1.0mm.nii.gz",
    "mri/vsinus.mgz", "mri/wm.asegedit.mgz", "mri/wm.mgz",
    "mri/wm.seg.mgz", "mri/wmparc.mgz",
    "surf/lh.w-g.pct.mgh", "surf/rh.w-g.pct.mgh",
)
SURFACES = ("orig", "smoothwm", "inflated", "white", "white.preaparc",
            "pial", "pial.T1", "sphere", "sphere.reg")
MORPHS = ("thickness", "area", "area.pial", "area.mid", "volume",
          "curv", "curv.pial", "avg_curv", "sulc", "jacobian_white",
          "inflated.H", "inflated.K", "smoothwm.BE.crv", "smoothwm.C.crv",
          "smoothwm.FI.crv", "smoothwm.H.crv", "smoothwm.K.crv",
          "smoothwm.K1.crv", "smoothwm.K2.crv", "smoothwm.S.crv",
          "white.preaparc.H", "white.preaparc.K")
ANNOTS = ("aparc", "aparc.a2009s", "aparc.DKTatlas", "BA_exvivo", "BA_exvivo.thresh", "mpm.vpnl")
STATS = ("aseg.stats", "wmparc.stats", "brainvol.stats", "entowm.stats",
         "vsinus.stats", "synthseg.tiv.dat", "synthseg.vol.csv")

def paths() -> tuple[str, ...]:
    """返回固定 profile 的相对被试目录路径。"""
    return (
        *VOLUMES,
        *(f"surf/{hemi}.{name}" for hemi in ("lh", "rh") for name in SURFACES),
        *(f"surf/{hemi}.{name}" for hemi in ("lh", "rh") for name in MORPHS),
        *(f"label/{hemi}.{name}.annot" for hemi in ("lh", "rh") for name in ANNOTS),
        *(f"stats/{hemi}.{name}.stats" for hemi in ("lh", "rh")
          for name in ("aparc", "aparc.a2009s", "aparc.DKTatlas", "aparc.pial",
                       "BA_exvivo", "BA_exvivo.thresh", "curv", "w-g.pct")),
        *(f"stats/{name}" for name in STATS),
    )
