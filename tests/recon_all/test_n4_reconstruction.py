"""验证旧 N4 二进制兼容、明确线程与实际剖析状态。"""
from pathlib import Path
import json
import nibabel as nib
import numpy as np
import pytest
from fnit.recon_all.n4_itk import correct_volume

def test_legacy_binary_reports_actual_single_thread(tmp_path):
    script = tmp_path / "legacy"
    script.write_text("#!/usr/bin/env python3\nimport sys,shutil\nif len(sys.argv)!=9: sys.exit(2)\nshutil.copyfile(sys.argv[1],sys.argv[2])\n")
    script.chmod(0o755)
    source=tmp_path / "orig.mgz"
    values=np.arange(27,dtype=np.uint8).reshape(3,3,3)
    nib.save(nib.MGHImage(values,np.eye(4)),str(source))
    output=tmp_path / "nu0.mgz";profile=tmp_path / "profile.json"
    correct_volume(input_file=source,output_file=output,binary=script,
                   reconstruction_threads=4,profile_path=profile)
    np.testing.assert_array_equal(nib.load(str(output)).dataobj,values)
    info=json.loads(profile.read_text())
    assert info["reconstruction_threads"] == 1
    assert not info["native_profile_available"]
    assert info["requested_reconstruction_threads"] == 4

@pytest.mark.parametrize("threads",[0,-1,True,1.5])
def test_invalid_reconstruction_threads_fail_before_io(threads):
    with pytest.raises(ValueError):
        correct_volume(input_file="missing",output_file="missing",binary="missing",
                       reconstruction_threads=threads)
