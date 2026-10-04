"""Sequential admitted stage arms; cancel forwards to owned controller cleanup."""
import argparse,json,pathlib,signal,subprocess,sys,time,os

def main():
 a=argparse.ArgumentParser();a.add_argument('--run',required=True,type=pathlib.Path);a.add_argument('--controller',required=True,type=pathlib.Path);p=a.parse_args();state={'status':'running','pid':os.getpid(),'arms':[],'order':['A8f','B3a'],'whole_case':False};child=None;cancelled=[]
 def write():
  temp=p.run/'queue.json.tmp';temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(p.run/'queue.json')
 def cancel(sig,frame):
  cancelled.append(sig);state['cancellation_signals']=list(cancelled)
  if child is not None and child.poll() is None:child.send_signal(signal.SIGTERM)
 for sig in [signal.SIGINT,signal.SIGTERM]:signal.signal(sig,cancel)
 write()
 for arm in state['order']:
  if cancelled:break
  with (p.run/(arm+'.controller.log')).open('x') as log:
   child=subprocess.Popen([sys.executable,str(p.controller),'--config',str(p.run/(arm+'.json'))],stdout=log,stderr=subprocess.STDOUT)
   # A signal inside Popen arrived before child was assigned; forward now.
   if cancelled and child.poll() is None:
    try:child.send_signal(signal.SIGTERM)
    except ProcessLookupError:pass
   state['current_arm']=arm;state['controller_pid']=child.pid;write()
   code=child.wait();state['arms'].append({'name':arm,'controller_exit_code':code});write()
   if code!=0 or cancelled:break
 state['status']='cancelled' if cancelled else 'complete' if len(state['arms'])==2 and all(x['controller_exit_code']==0 for x in state['arms']) else 'failed';state['current_arm']=None;write()
 return 0 if state['status']=='complete' else 1
if __name__=='__main__':raise SystemExit(main())
