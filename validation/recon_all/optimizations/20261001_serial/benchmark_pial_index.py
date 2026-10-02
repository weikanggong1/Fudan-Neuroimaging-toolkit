"""两例四半球真实white/pial：原定义法向与完整桶的同输入CPU回归。"""
import argparse,hashlib,importlib.util,json,platform,time
from pathlib import Path
import nibabel.freesurfer.io as fs
import numpy as np
import numba,torch
from fnit.recon_all import place_surface_repulsion as new
from fnit.recon_all.place_surface_normals import FaceNormalTopology
p=argparse.ArgumentParser(description=__doc__)
p.add_argument("--subjects",type=Path,nargs=2,required=True);p.add_argument("--baseline-source",type=Path,required=True)
p.add_argument("--output",type=Path,required=True);p.add_argument("--commit",required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
torch.set_num_threads(4);numba.set_num_threads(4)
spec=importlib.util.spec_from_file_location("fnit.recon_all.frozen_repulsion",a.baseline_source)
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def measure(function,*args,**kwargs):
    tick=time.perf_counter();value=function(*args,**kwargs);return value,time.perf_counter()-tick
rows=[]
for subject in a.subjects:
 for hemi in ("lh","rh"):
    white=subject/("surf/"+hemi+".white");pial=subject/("surf/"+hemi+".pial")
    label=subject/("label/"+hemi+".cortex+hipamyg.label")
    xyz,faces=fs.read_geometry(str(white));current,_=fs.read_geometry(str(pial))
    ripped=np.ones(len(xyz),bool);ripped[fs.read_label(str(label))]=False
    index,setup=measure(new.OriginalVertexBuckets,xyz,ripped)
    topology,topology_seconds=measure(FaceNormalTopology,faces,len(xyz))
    old_normal,old_seconds=measure(old.original_vertex_normals,xyz,faces)
    new_normal,new_seconds=measure(new.original_vertex_normals,xyz,faces,topology=topology)
    np.testing.assert_array_equal(old_normal,new_normal)
    queries=[]
    for name,positions in (("white",xyz),("pial",current)):
      old_bucket,old_bucket_seconds=measure(old.vertex_buckets,positions,xyz,ripped)
      new_bucket,new_bucket_seconds=measure(index.query,positions)
      for x,y in zip(old_bucket,new_bucket):np.testing.assert_array_equal(x,y)
      queries.append({"current":name,"old_seconds":old_bucket_seconds,"new_seconds":new_bucket_seconds,"candidates":len(new_bucket[1]),"exact":True})
    rows.append({"subject":str(subject),"hemi":hemi,"input_sha256":{str(x):sha(x) for x in (white,pial,label)},
      "index_setup_seconds":setup,"topology_setup_seconds":topology_seconds,
      "normal_old_seconds":old_seconds,"normal_new_seconds":new_seconds,"normal_exact":True,"queries":queries})
(a.output/"report.json").write_text(json.dumps({"commit":a.commit,"host":platform.node(),"threads":4,
 "scope":"frozen_self_generated_geometry_component; not_full_pial_or_whole",
 "source_sha256":{"old":sha(a.baseline_source),"new":sha(new.__file__),"benchmark":sha(__file__)},"rows":rows},indent=2)+"\n")
