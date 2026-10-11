"""私有 sitecustomize 探针：记录实际守卫/线程/精度，不修改数学函数。

仅在 FNIT_GCA_AUDIT_PATH 指定新 JSON 时启用。由独立诊断脚本复制为
sitecustomize.py，随冷 CLI 导入；设置 interop=1，使4核/4线程预算明确。
系统 profile 观察注册、稳定/全样本目标与梯度；不替换计算函数或输入。
"""
import atexit,json,os,sys
from pathlib import Path

if os.environ.get('FNIT_GCA_AUDIT_PATH'):
 import torch
 torch.set_num_interop_threads(1)
 observed={};guards=[]
 def precision():
  return {'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_tf32':torch.backends.cudnn.allow_tf32,'default_dtype':str(torch.get_default_dtype()),'autocast':torch.is_autocast_enabled(),'torch_threads':torch.get_num_threads(),'torch_interop_threads':torch.get_num_interop_threads()}
 def probe(frame,event,arg):
  module=frame.f_globals.get('__name__','');name=frame.f_code.co_name
  if not module.startswith('fnit.recon_all.mri_em_register'):return
  if event=='call' and name=='register_t1' and 'registration_entry' not in observed:
   from fnit.recon_all import execution_recipe
   record=execution_recipe._guard.get();guards.append(record)
   observed['registration_entry']={'guard_enabled':bool(record and record['enabled']),'environment_guard':os.environ.get('FNIT_RECON_FORBID_ORIGINAL_STAGES'),'precision':precision()}
  if module.endswith('mri_em_register_em_candidate') and name in ('cost','gradient'):
   objective=frame.f_locals.get('self');clamp=getattr(objective,'clamp',None);key=f'{name}_clamp_{clamp}'
   if event=='return' and key not in observed:
    matrix=frame.f_locals.get('matrix');observed[key]={'matrix_dtype':str(getattr(matrix,'dtype',None)),'return_dtype':str(getattr(arg,'dtype',None)),'precision':precision()}
  if module.endswith('mri_em_register_complete') and event=='call' and name=='evaluate' and 'optimizer_vector_dtype' not in observed:
   observed['optimizer_vector_dtype']=str(frame.f_locals['vector'].dtype)
 sys.setprofile(probe)
 def save():
  sys.setprofile(None)
  import fnit,numba
  observed.update(scope='runtime_guard_and_dtype_observation_only_not_syscall_sandbox',guard_records=guards,guard_environment_at_exit=os.environ.get('FNIT_RECON_FORBID_ORIGINAL_STAGES'),numba_threads=numba.get_num_threads(),package_loaded_from_installed_target=Path(fnit.__file__).resolve().parent.parent==Path(os.environ['FNIT_GCA_INSTALLED_TARGET']).resolve(),precision_at_exit=precision())
  Path(os.environ['FNIT_GCA_AUDIT_PATH']).write_text(json.dumps(observed,indent=2)+'\n')
 atexit.register(save)
