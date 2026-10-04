"""顺序执行原始T1资源失败重跑；队列结束与各例执行成功分别记录。"""
from pathlib import Path
import argparse, json, subprocess, datetime, os, signal, hashlib


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def execution_role(plan):
    role = plan.get('execution_role', 'baseline')
    if role not in ('baseline', 'precision_candidate'):
        raise ValueError('execution_role must be baseline or precision_candidate')
    return role


def evaluation_config(spec, case_id, actual_config, shared_lock):
    for name in ('script', 'config'):
        if digest(spec[name]) != spec[name + '_sha256']:
            raise ValueError('evaluation ' + name + ' changed')
    config = json.loads(Path(spec['config']).read_text())
    if config.get('evaluated_role') != 'precision_candidate' or config.get('case') != case_id:
        raise ValueError('evaluation role/case differs')
    if Path(config['evaluated_config']).resolve() != Path(actual_config).resolve():
        raise ValueError('evaluation must bind actual retry_config')
    if Path(config['lock']).resolve() != Path(shared_lock).resolve():
        raise ValueError('evaluation must use queue shared lock')
    output = Path(config['output']).resolve()
    retry_root = Path(actual_config).resolve().parent
    if output == retry_root or retry_root in output.parents or output in retry_root.parents:
        raise ValueError('evaluation output must be independent of algorithm output')
    return config


def evaluation_result(config, spec, exit_code):
    # Recheck the frozen evaluator/config after the actual execution.
    for name in ('script', 'config'):
        if digest(spec[name]) != spec[name + '_sha256']:
            raise ValueError('evaluation ' + name + ' changed during execution')
    checkpoint = Path(config['output']) / 'checkpoint.json'
    state = json.loads(checkpoint.read_text())
    if (exit_code != 0 or state.get('status') != 'complete'
            or state.get('evaluated_role') != 'precision_candidate'
            or state.get('case') != config['case']
            or state.get('config_sha256') != spec['config_sha256']
            or state.get('script_sha256') != spec['script_sha256']
            or type(state.get('strict_138', {}).get('checked')) is not int
            or state['strict_138']['checked'] != 138
            or type(state.get('strict_138', {}).get('passed')) is not int
            or not 0 <= state['strict_138']['passed'] <= 138
            or len(state.get('phases', {})) != 18
            or any(row.get('status') != 'complete' for row in state['phases'].values())):
        raise ValueError('evaluation checkpoint incomplete or binding differs')
    return {'status': 'complete', 'evaluation_succeeded': True,
            'checkpoint_sha256': digest(checkpoint), 'strict_138': state['strict_138'],
            'meaning': 'comparison completed; strict passed count is not an equivalence gate'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    # Reserve the receipt without replacing another queue's report.
    with args.report.open('x') as stream:
        stream.write('{}\n')
    report = {'schema': 'fnit-original-baseline-resource-replays-v1',
              'pid': os.getpid(), 'host': os.uname().nodename, 'started_utc': now(),
              'queue_script_sha256': digest(__file__), 'status': 'starting', 'cases': [],
              'queue_finished': False, 'all_cases_succeeded': False,
              'meaning': 'queue completion and per-case success are separate; original failures preserved'}
    child = None
    interrupted = False
    child_kind = None
    def save():
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    def stop(signum, frame):
        nonlocal interrupted
        interrupted = True
        report['interrupt_signal'] = signum
        # Admission owns cleanup of its descendants and independent process groups.
        if child is not None and child.poll() is None:
            try:
                if child_kind == 'evaluation':
                    os.killpg(child.pid, signal.SIGTERM)
                else:
                    child.send_signal(signal.SIGTERM)
            except ProcessLookupError:
                pass
    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    code = 1
    try:
        plan = json.loads(args.plan.read_text())
        report['plan_sha256'] = digest(args.plan)
        role = execution_role(plan)
        if role == 'precision_candidate':
            report.update(schema='fnit-precision-candidate-sequential-validation-v1',
                          execution_role=role, initial_evaluations=[])
        elif plan.get('initial_evaluation') or any(c.get('post_evaluation') for c in plan['cases']):
            raise ValueError('evaluation options require precision_candidate role')
        script = Path(plan['admission_script'])
        if digest(script) != plan['admission_script_sha256']:
            raise ValueError('admission script changed')
        cases = plan['cases']
        ids = [case['id'] for case in cases]
        if not cases or len(set(ids)) != len(ids):
            raise ValueError('queue requires distinct nonempty cases')
        evaluation_outputs = set()
        evaluations = [(c['post_evaluation'], c['id'], str(Path(c['retry_root'])/'retry_config.json'))
                       for c in cases if c.get('post_evaluation')]
        initial = plan.get('initial_evaluation')
        if initial:
            evaluations.append((initial, initial['id'], initial['actual_config']))
        for spec, case_id, actual in evaluations:
            config = evaluation_config(spec, case_id, actual, plan['shared_lock'])
            output = str(Path(config['output']).resolve())
            if output in evaluation_outputs:
                raise ValueError('evaluation outputs must be distinct')
            if Path(output).exists():
                raise FileExistsError('evaluation requires fresh output: ' + output)
            evaluation_outputs.add(output)
        def evaluate(spec, case_id, actual):
            nonlocal child, child_kind
            result = {'case': case_id, 'status': 'starting', 'evaluation_succeeded': False,
                      'started_utc': now()}
            try:
                config = evaluation_config(spec, case_id, actual, plan['shared_lock'])
                # Actual completion/source/resource verification remains in the frozen evaluator.
                command = [plan['python'], spec['script'], '--config', spec['config']]
                result['command'] = command
                log = Path(spec['config']).with_name(Path(spec['config']).name + '.queue-evaluation.log')
                result['log'] = str(log)
                with log.open('x') as stream:
                    child_kind = 'evaluation'
                    child = subprocess.Popen(command, start_new_session=True, stdout=stream,
                                             stderr=subprocess.STDOUT)
                    if interrupted:
                        stop(report['interrupt_signal'], None)
                    result['exit_code'] = child.wait()
                    child = None
                result.update(evaluation_result(config, spec, result['exit_code']))
            except Exception as error:
                result.update(status='failed', error=repr(error))
                if child is not None and child.poll() is None:
                    try:
                        os.killpg(child.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    child.wait()
                child = None
            finally:
                child_kind = None
                result['finished_utc'] = now()
            return result
        report.update(status='running', planned_cases=len(cases))
        save()
        if initial and not interrupted:
            report['initial_evaluations'].append(evaluate(initial, initial['id'], initial['actual_config']))
            save()
        for case in cases:
            if interrupted:
                break
            if digest(script) != plan['admission_script_sha256']:
                raise ValueError('admission script changed between cases')
            command = [plan['python'], str(script), '--config', case['original_config'],
                       '--retry-root', case['retry_root'], '--report', case['admission_report'],
                       '--lock', plan['shared_lock'], '--poll-seconds', '30', '--query-timeout', '5',
                       '--maximum-wait-seconds', '86400']
            row = {'case': case['id'], 'status': 'starting', 'execution_succeeded': False,
                   'started_utc': now(), 'command': command}
            report['cases'].append(row)
            report['current_case'] = case['id']
            save()
            try:
                child_kind = 'admission'
                child = subprocess.Popen(command, start_new_session=True)
                row.update(pid=child.pid, status='admission_or_execution')
                save()
                # Cover a signal delivered inside Popen before child was assigned.
                if interrupted:
                    stop(report['interrupt_signal'], None)
                row['exit_code'] = child.wait()
                child = None
                receipt = Path(case['admission_report'])
                if receipt.is_file():
                    row['admission_sha256'] = digest(receipt)
                    row['admission_status'] = json.loads(receipt.read_text()).get('status')
                completion = Path(case['retry_root']) / 'diagnostics/completion.json'
                if completion.is_file():
                    row['completion_sha256'] = digest(completion)
                    row['execution_completion'] = json.loads(completion.read_text())
                result = row.get('execution_completion', {})
                success = (row['exit_code'] == 0 and row.get('admission_status') == 'complete'
                           and result.get('execution_status') == 'complete'
                           and result.get('exit_code') == 0 and result.get('pipeline_status') == 'complete')
                row.update(execution_succeeded=success,
                           status='complete' if success else 'failed_or_not_admitted')
                if case.get('post_evaluation'):
                    row['evaluation'] = (evaluate(case['post_evaluation'], case['id'],
                        str(Path(case['retry_root'])/'retry_config.json')) if success and not interrupted
                        else {'status': 'skipped_algorithm_not_complete', 'evaluation_succeeded': False})
            except Exception as error:
                row.update(status='wrapper_failed', error=repr(error))
                if child is not None and child.poll() is None:
                    child.send_signal(signal.SIGTERM)
                    child.wait()  # Keep the cooperative lock until admission cleanup finishes.
                child = None
            finally:
                if case.get('post_evaluation') and 'evaluation' not in row:
                    row['evaluation'] = {'status': 'skipped_algorithm_not_complete', 'evaluation_succeeded': False}
                row['finished_utc'] = now()
                save()
        report['queue_finished'] = not interrupted and len(report['cases']) == len(cases)
        report['status'] = 'interrupted' if interrupted else 'all_attempts_finished'
        report['all_cases_succeeded'] = (report['queue_finished'] and
                                        all(row['execution_succeeded'] for row in report['cases']))
        if role == 'precision_candidate':
            results = report['initial_evaluations'] + [row['evaluation'] for row in report['cases'] if 'evaluation' in row]
            report.update(evaluations_succeeded=sum(x.get('evaluation_succeeded', False) for x in results),
                          evaluations_failed_or_skipped=sum(not x.get('evaluation_succeeded', False) for x in results),
                          all_evaluations_succeeded=all(x.get('evaluation_succeeded', False) for x in results))
        code = 130 if interrupted else (0 if report['all_cases_succeeded'] and report.get('all_evaluations_succeeded', True) else 1)
    except Exception as error:
        report.update(status='queue_wrapper_failed', error=repr(error))
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGTERM)
            child.wait()
        code = 130 if interrupted else 1
    finally:
        report.update(current_case=None, finished_utc=now(), exit_code=code,
                      successful=sum(row.get('execution_succeeded', False) for row in report['cases']),
                      failed_or_not_admitted=sum(not row.get('execution_succeeded', False)
                                                 for row in report['cases']))
        save()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
