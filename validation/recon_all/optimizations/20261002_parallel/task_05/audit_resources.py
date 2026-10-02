"""核验MNI资源大小/SHA和所用源码/程序；仅输出元数据，不复制影像或许可证。

--weights/--assets/--native-bin为已授权本地资源目录；--source为固定FreeSurfer源码。
--output为JSON，--code-commit绑定候选代码。程序路径不用于生产调用。
"""
import argparse,hashlib,json
from pathlib import Path
from fnit.weights import WEIGHT_FILES
from fnit.recon_all.assets import ASSET_FILES
from fnit.recon_all.mni_aux_chain import TEMPLATE_DIR


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(1<<20),b''):h.update(data)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['weights','assets','native-bin','source','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--code-commit',required=True);a=p.parse_args()
    result={'code_commit':a.code_commit,'resources':{},'programs':{},'source':{},
            'source_pin':'d932c45b7941662ea380a05efef580568b98d41a',
            'redistribution':'no inputs/templates/weights/upstream sources/licenses copied into report'}
    files=[(a.weights/'synthmorph.deform.3.h5',WEIGHT_FILES['synthmorph.deform.3.h5'][1:3])]
    for suffix in ['.cropped','']:
        name=str(TEMPLATE_DIR/f'mni152.1.0mm{suffix}.nii.gz')
        files.append((a.assets/name,ASSET_FILES[name][:2]))
    for path,(size,digest) in files:
        actual=sha(path);result['resources'][str(path)]={'bytes':path.stat().st_size,'sha256':actual,
          'expected_bytes':size,'expected_sha256':digest,'match':path.stat().st_size==size and actual==digest}
    for name in ['mri_warp_convert','mri_ca_register','mri_convert']:
        path=a.native_bin/name;result['programs'][name]={'sha256':sha(path),'bytes':path.stat().st_size}
    for name in ['mri_warp_convert/mri_warp_convert.cpp','utils/gcamorph.cpp','utils/warpfield.cpp','LICENSE.txt']:
        result['source'][name]={'sha256':sha(a.source/name)}
    result['source_marker']=(a.source/'.fnit-source-commit').read_text().strip()
    result['passed']=all(x['match'] for x in result['resources'].values()) and result['source_marker']==result['source_pin']
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    if not result['passed']:raise SystemExit(1)
if __name__=='__main__':main()
