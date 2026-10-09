"""最终white独立BG/247标记契约；模拟输入不是公开T1 benchmark。"""
import numpy as np
import pytest

from fnit.recon_all import place_surface_final_white_rip as module


def test_annotation_gates_BG_but_not_247_and_does_not_enable_midline(monkeypatch):
    xyz = np.zeros((4,3), np.float32)
    xyz[:,0] = np.arange(4)
    normal = np.zeros_like(xyz);normal[:,2] = 1
    # 四个顶点均遇BG，但只有具名insula被冻结；247不要求注释。
    monkeypatch.setattr(module,"_nearest",lambda seg,affine,x,y,z:247 if int(x) == 3 else 12)
    table=np.zeros((3,5),np.int32);table[:,4]=[10,20,30]
    flags=module.final_white_rip_flags(xyz,normal,np.zeros((2,2,2),np.int32),
        np.eye(4),np.array([10,20,30,30]),table,
        [b"entorhinal",b"insula",b"unknown"],hemisphere="lh")
    np.testing.assert_array_equal(flags,[0,1,0,1])


def test_final_white_rip_rejects_unmatched_annotation():
    with pytest.raises(ValueError,match="dimensions differ"):
        module.final_white_rip_flags(np.zeros((2,3)),np.zeros((2,3)),np.zeros((2,2,2)),
            np.eye(4),np.array([0]),np.zeros((1,5)),[b"unknown"],hemisphere="lh")


def test_final_white_public_API_never_overwrites_preaparc_or_other_inputs(tmp_path):
    from fnit.recon_all.place_final_white_python import place_final_white

    with pytest.raises(ValueError, match="cannot overwrite inputs"):
        place_final_white(subject_dir=tmp_path, hemi="lh",
                          output=tmp_path/"surf/lh.white.preaparc")
    with pytest.raises(ValueError, match="cannot overwrite inputs"):
        place_final_white(subject_dir=tmp_path, hemi="rh", output=tmp_path/"rh.white",
                          output_volume=tmp_path/"label/rh.aparc.annot")


def test_final_white_public_API_selects_complete_white_semantics(monkeypatch,tmp_path):
    from fnit.recon_all import place_final_white_python as final

    captured = {}
    def stage(**kwargs):
        captured.update(kwargs)
        return {"white_stage":"final_white"}
    monkeypatch.setattr(final,"_place_white_preaparc",stage)
    result=final.place_final_white(subject_dir=tmp_path,hemi="rh",output=tmp_path/"rh.white",
        candidate_backend="torch_snapshot",candidate_grid_cells_per_axis=3,
        retained_mht_backend="compiled",cleanup_marking_backend="source_torch",device="cuda:0")
    assert result["white_stage"] == "final_white"
    assert captured["complete"] is True and captured["final_white"] is True
    assert captured["candidate_backend"] == "torch_snapshot" and captured["steps"] == 400
    assert captured["regularization_backend"] == "cpu" and captured["sampling_backend"] == "cpu"
