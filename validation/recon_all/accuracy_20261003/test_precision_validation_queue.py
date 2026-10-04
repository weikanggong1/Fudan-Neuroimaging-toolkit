"""纯 CPU 的真实子进程队列契约测试，无影像或 GPU 计算。"""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location('queue', HERE/'run_resource_replay_queue.py')
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


class PrecisionQueueTests(unittest.TestCase):
    def fixture(self, root, algorithm_ok=True, evaluator_ok=True):
        admission = root/'admission.py'
        admission.write_text('''import argparse,json,pathlib,sys
p=argparse.ArgumentParser()
for k in ['config','retry-root','report','lock','poll-seconds','query-timeout','maximum-wait-seconds']:p.add_argument('--'+k)
a=p.parse_args();r=pathlib.Path(a.retry_root);(r/'diagnostics').mkdir(parents=True)
c=json.load(open(a.config));(r/'retry_config.json').write_text(json.dumps(c))
ok=c['ok'];d={'execution_status':'complete' if ok else 'failed','exit_code':0 if ok else 1,'pipeline_status':'complete' if ok else 'failed'}
(r/'diagnostics/completion.json').write_text(json.dumps(d));pathlib.Path(a.report).write_text(json.dumps({'status':'complete' if ok else 'failed'}));sys.exit(0 if ok else 1)
''')
        evaluator = root/'evaluator.py'
        evaluator.write_text('''import json,pathlib,sys,hashlib
p=pathlib.Path(sys.argv[2]);c=json.load(open(p));actual=pathlib.Path(c['evaluated_config']);completion=json.load(open(actual.parent/'diagnostics/completion.json'))
assert completion['execution_status']=='complete'
o=pathlib.Path(c['output']);o.mkdir();(o/'entered').write_text('once')
if not c['ok']:sys.exit(1)
d={'status':'complete','case':c['case'],'evaluated_role':c['evaluated_role'],'config_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'strict_138':{'checked':138,'passed':5},'phases':{str(i):{'status':'complete'} for i in range(18)}}
(o/'checkpoint.json').write_text(json.dumps(d))
''')
        original=root/'original.json';original.write_text(json.dumps({'ok':algorithm_ok}))
        config=root/'evaluation.json';config.write_text(json.dumps({'case':'case','evaluated_role':'precision_candidate','evaluated_config':str(root/'retry/retry_config.json'),'lock':str(root/'lock'),'output':str(root/'comparison'),'ok':evaluator_ok}))
        evaluation={'script':str(evaluator),'script_sha256':queue.digest(evaluator),'config':str(config),'config_sha256':queue.digest(config)}
        plan={'execution_role':'precision_candidate','python':sys.executable,'admission_script':str(admission),'admission_script_sha256':queue.digest(admission),'shared_lock':str(root/'lock'),'cases':[{'id':'case','original_config':str(original),'retry_root':str(root/'retry'),'admission_report':str(root/'admission.json'),'post_evaluation':evaluation}]}
        return plan,config,evaluator

    def run_plan(self, root, plan):
        path=root/'plan.json';path.write_text(json.dumps(plan));report=root/'queue.json'
        proc=subprocess.run([sys.executable,str(HERE/'run_resource_replay_queue.py'),'--plan',str(path),'--report',str(report)],capture_output=True,text=True)
        return proc.returncode,json.loads(report.read_text())

    def test_raw_completion_is_persisted_while_evaluator_is_blocked(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            plan, _, evaluator = self.fixture(root)
            original = evaluator.read_text()
            barrier = "import time\nwhile not pathlib.Path(c['release']).exists():time.sleep(0.02)\n"
            evaluator.write_text(original.replace("if not c['ok']:sys.exit(1)", barrier + "if not c['ok']:sys.exit(1)"))
            config_path = Path(plan['cases'][0]['post_evaluation']['config'])
            config = json.loads(config_path.read_text())
            config['release'] = str(root/'release')
            config_path.write_text(json.dumps(config))
            evaluation = plan['cases'][0]['post_evaluation']
            evaluation.update(script_sha256=queue.digest(evaluator), config_sha256=queue.digest(config_path))
            path = root/'plan.json'; path.write_text(json.dumps(plan)); report = root/'queue.json'
            proc = subprocess.Popen([sys.executable, str(HERE/'run_resource_replay_queue.py'),
                '--plan', str(path), '--report', str(report)], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            try:
                deadline = time.monotonic() + 8
                observed = None
                while time.monotonic() < deadline:
                    self.assertIsNone(proc.poll(), 'queue exited before evaluator barrier')
                    if (root/'comparison/entered').exists():
                        try:
                            value = json.loads(report.read_text())
                        except (FileNotFoundError, json.JSONDecodeError):
                            time.sleep(0.02); continue
                        row = value['cases'][0]
                        if row.get('evaluation', {}).get('status') == 'running':
                            observed = value; break
                    time.sleep(0.02)
                self.assertIsNotNone(observed, 'running evaluation was not persisted')
                row = observed['cases'][0]
                self.assertTrue(row['execution_succeeded'])
                self.assertEqual(row['status'], 'complete')
                self.assertEqual(row['execution_completion']['pipeline_status'], 'complete')
                self.assertEqual(row['evaluation']['status'], 'running')
                self.assertIsInstance(row['evaluation']['pid'], int)
                self.assertFalse(observed['queue_finished'])
                self.assertFalse((root/'comparison/checkpoint.json').exists())
                (root/'release').write_text('continue')
                _, error = proc.communicate(timeout=8)
                self.assertEqual(proc.returncode, 0, error)
                final = json.loads(report.read_text())
                self.assertTrue(final['cases'][0]['execution_succeeded'])
                self.assertEqual(final['cases'][0]['evaluation']['status'], 'complete')
            finally:
                if proc.poll() is None:
                    (root/'release').write_text('cleanup')
                    proc.communicate(timeout=8)

    def test_default_role(self):
        self.assertEqual(queue.execution_role({}), 'baseline')
        with self.assertRaises(ValueError):queue.execution_role({'execution_role':'startup_only_candidate'})

    def test_success_keeps_strict_failure_count_nonblocking(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);plan,_,_=self.fixture(root);code,r=self.run_plan(root,plan)
            self.assertEqual(code,0);self.assertEqual(r['schema'],'fnit-precision-candidate-sequential-validation-v1')
            self.assertEqual((r['successful'],r['evaluations_succeeded']),(1,1))
            self.assertEqual(r['cases'][0]['evaluation']['strict_138']['passed'],5)

    def test_algorithm_failure_skips_evaluator(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);plan,_,_=self.fixture(root,False);code,r=self.run_plan(root,plan)
            self.assertEqual(code,1);self.assertFalse((root/'comparison').exists())
            self.assertEqual(r['evaluations_failed_or_skipped'],1)
            self.assertEqual(r['cases'][0]['evaluation']['status'],'skipped_algorithm_not_complete')

    def test_evaluation_failure_does_not_relabel_algorithm(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);plan,_,_=self.fixture(root,evaluator_ok=False);code,r=self.run_plan(root,plan)
            self.assertEqual(code,1);self.assertTrue(r['all_cases_succeeded']);self.assertTrue(r['queue_finished'])
            self.assertEqual((r['successful'],r['evaluations_succeeded'],r['evaluations_failed_or_skipped']),(1,0,1))

    def test_failed_evaluation_continues_next_case(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);plan,config,_=self.fixture(root,evaluator_ok=False)
            c=json.loads(config.read_text());c.update(case='next', ok=True,
                evaluated_config=str(root/'nextretry/retry_config.json'),output=str(root/'nextcomparison'))
            nc=root/'nextevaluation.json';nc.write_text(json.dumps(c))
            first=plan['cases'][0];second=dict(first,id='next',retry_root=str(root/'nextretry'),
                admission_report=str(root/'nextadmission.json'),post_evaluation=dict(first['post_evaluation'],
                config=str(nc),config_sha256=queue.digest(nc)));plan['cases'].append(second)
            code,r=self.run_plan(root,plan);self.assertEqual(code,1);self.assertEqual(r['successful'],2)
            self.assertEqual(r['evaluations_succeeded'],1);self.assertTrue((root/'nextcomparison/entered').is_file())

    def test_cancel_active_evaluator(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);plan,_,evaluator=self.fixture(root)
            evaluator.write_text('import os,signal,time\nos.kill(os.getppid(),signal.SIGTERM)\ntime.sleep(10)\n')
            plan['cases'][0]['post_evaluation']['script_sha256']=queue.digest(evaluator)
            code,r=self.run_plan(root,plan);self.assertEqual(code,130)
            self.assertEqual(r['status'],'interrupted');self.assertEqual(r['successful'],1)
            self.assertFalse(r['all_evaluations_succeeded'])

    def test_strict_checked_count_is_required(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);plan,_,evaluator=self.fixture(root)
            evaluator.write_text(evaluator.read_text().replace("'checked':138", "'checked':137"))
            plan['cases'][0]['post_evaluation']['script_sha256']=queue.digest(evaluator)
            code,r=self.run_plan(root,plan);self.assertEqual(code,1)
            self.assertTrue(r['all_cases_succeeded']);self.assertEqual(r['evaluations_succeeded'],0)

    def test_strict_passed_report_integrity(self):
        for replacement in ["", ", 'passed':True", ", 'passed':-1", ", 'passed':139", ", 'passed':5.0"]:
            with self.subTest(replacement=replacement),tempfile.TemporaryDirectory() as d:
                root=Path(d);plan,_,evaluator=self.fixture(root)
                evaluator.write_text(evaluator.read_text().replace(",'passed':5", replacement))
                plan['cases'][0]['post_evaluation']['script_sha256']=queue.digest(evaluator)
                code,r=self.run_plan(root,plan);self.assertEqual(code,1)
                self.assertTrue(r['all_cases_succeeded']);self.assertEqual(r['evaluations_succeeded'],0)
                self.assertIn('checkpoint incomplete',r['cases'][0]['evaluation']['error'])

    def test_wrong_role_and_actual_path_rejected_before_algorithm(self):
        for key,value in [('evaluated_role','baseline'),('evaluated_config','/wrong/retry_config.json')]:
            with self.subTest(key=key),tempfile.TemporaryDirectory() as d:
                root=Path(d);plan,config,_=self.fixture(root);c=json.loads(config.read_text());c[key]=value;config.write_text(json.dumps(c));plan['cases'][0]['post_evaluation']['config_sha256']=queue.digest(config)
                code,r=self.run_plan(root,plan);self.assertEqual(code,1);self.assertEqual(r['cases'],[])

    def test_script_and_config_drift_rejected(self):
        for which in ['script','config']:
            with self.subTest(which=which),tempfile.TemporaryDirectory() as d:
                root=Path(d);plan,config,evaluator=self.fixture(root);p=evaluator if which=='script' else config;p.write_text(p.read_text()+'\n')
                code,r=self.run_plan(root,plan);self.assertEqual(code,1);self.assertEqual(r['cases'],[])
                self.assertIn('changed',r['error'])

    def test_initial_evaluation_runs_first_without_raw_claim(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);plan,config,_=self.fixture(root);initial=root/'initial';(initial/'diagnostics').mkdir(parents=True);(initial/'retry_config.json').write_text('{}');(initial/'diagnostics/completion.json').write_text('{"execution_status":"failed"}')
            c=json.loads(config.read_text());c.update(evaluated_config=str(initial/'retry_config.json'),output=str(root/'initialcomparison'));ic=root/'initialevaluation.json';ic.write_text(json.dumps(c))
            init=dict(plan['cases'][0]['post_evaluation'],id='case',actual_config=str(initial/'retry_config.json'),config=str(ic),config_sha256=queue.digest(ic));plan['initial_evaluation']=init
            code,r=self.run_plan(root,plan);self.assertEqual(code,1);self.assertEqual(r['initial_evaluations'][0]['status'],'failed');self.assertEqual(r['successful'],1)
            self.assertEqual(r['evaluations_succeeded'],1);self.assertFalse((root/'initialcomparison').exists())


if __name__=='__main__':unittest.main()
