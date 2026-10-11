"""复用已验证邻域观察器构建器，独立捕获同 candidate SFF 后字段。"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(*, native_build, base_observer, observer_tools, output):
    """固定 full-fitness builder→新 post-SFF-only benchmark observer。

    native_build/base_observer 是私有已固定源码/Conda 构建，不进生产。
    observer_tools 为既有 build_candidate_neighborhood_observer.py 所在
    目录；通过其原编译/链接逻辑构建，不改旧树。output 必须不存在。
    只移动只读字段捕获到真实 SFF call 之后，原 SFF 不重复调用。
    保留原 full-fitness 捕获/退出语义；产生独立程序/编译SHA/收据。
    """
    native_build,base_observer,observer_tools,output=map(Path,(native_build,base_observer,observer_tools,output))
    if output.exists():raise FileExistsError('fresh post-SFF workspace required')
    builder=observer_tools/'build_candidate_neighborhood_observer.py'
    sys.path.insert(0,str(observer_tools))
    spec=importlib.util.spec_from_file_location('fnit_post_sff_builder_reuse',builder)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    receipt=json.loads((base_observer/'build.private.json').read_text())
    source=base_observer/'mrisurf_defect.cpp';archive=base_observer/'libutils.a'
    if sha(source)!=receipt['patched_defect_sha256'] or sha(archive)!=receipt['archive_sha256']:
        raise ValueError('bound base source/archive required')
    anchor='    /*compute the second fundamental form */\n    computeDefectSecondFundamentalForm(mris, &dp->tp);'
    text=source.read_text()
    if text.count(anchor)!=1:raise ValueError('source SFF call anchor changed')
    captured=module.CAPTURE.replace('FNIT_TOPOLOGY_CANDIDATE_NEIGHBORHOOD_CAPTURE',
                                  'FNIT_TOPOLOGY_POST_SFF_NEIGHBORHOOD_CAPTURE')
    output.mkdir(parents=True)
    staging=output/'bound_staging';staging.mkdir()
    (staging/'mrisurf_defect.cpp').write_text(text.replace(anchor,anchor+'\n'+captured,1))
    os.link(archive,staging/'libutils.a')
    prepared=dict(receipt,patched_defect_sha256=sha(staging/'mrisurf_defect.cpp'),
        link_argv=[str(staging/'libutils.a') if arg==str(archive) else arg for arg in receipt['link_argv']])
    (staging/'build.private.json').write_text(json.dumps(prepared,indent=2)+'\n')
    module.CAPTURE=''  # Existing builder's before-call insertion becomes a no-op.
    result=module.build(native_build=native_build,base_observer=staging,output=output/'build')
    result.update(scope='first complete candidate immediately after its single SFF call; benchmark only',
        injection_after_call=True,sff_call_not_repeated=True,wrapper_sha256=sha(__file__),
        reused_builder_sha256=sha(builder),original_fullfit_receipt_sha256=sha(base_observer/'build.private.json'))
    (output/'build'/'build.private.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['native-build','base-observer','observer-tools','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    build(**vars(parser.parse_args()))


if __name__=='__main__':main()
