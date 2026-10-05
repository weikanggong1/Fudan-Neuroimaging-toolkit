"""顺序执行原始T1资源失败重跑；队列结束与各例执行成功分别记录。"""
from pathlib import Path
import argparse, json, subprocess, datetime, os, signal, hashlib


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def execution_role(plan):
    role = plan.get('execution_role', 'baseline')
    if role not in ('baseline', 'precision_candidate', 'precision_evaluation_only'):
        raise ValueError('execution_role must be baseline, precision_candidate or precision_evaluation_only')
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


def precision_evaluation_only(args, plan):
    """只读等待已启动的 raw 整例；仅创建并管理本队列的比较子进程。"""
    import math
    import threading
    import time

    poll = plan.get('poll_seconds', 30)
    maximum_wait = plan.get('maximum_wait_seconds', 86400)
    evaluation_timeout = plan.get('evaluation_timeout_seconds', 86400)
    for name, value in [('poll_seconds', poll), ('maximum_wait_seconds', maximum_wait),
                        ('evaluation_timeout_seconds', evaluation_timeout)]:
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(name + ' must be finite and positive')
    if poll > 60:
        raise ValueError('poll_seconds must be <= 60')
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open('x') as stream:
        stream.write('{}\n')
    report = dict(schema='fnit-precision-evaluation-only-v1', execution_role='precision_evaluation_only',
                  pid=os.getpid(), host=os.uname().nodename, started_utc=now(),
                  queue_script_sha256=digest(__file__), plan_sha256=digest(args.plan),
                  status='starting', queue_finished=False, all_cases_succeeded=False,
                  all_evaluations_succeeded=False, cases=[], raw_started_by_queue=False)
    cancelled = threading.Event()
    child = None

    def save():
        temporary = args.report.with_name(args.report.name + '.writing')
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        temporary.replace(args.report)

    def stop(signum, frame):
        cancelled.set()
        report['interrupt_signal'] = signum
        if child is not None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    def cleanup_owned():
        # The group belongs to a Popen(start_new_session=True) created here.
        if child is None:
            return
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            child.poll()  # Reap the leader while checking surviving owned descendants.
            try:
                os.killpg(child.pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.05)
        else:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        child.wait(timeout=5)

    def read_receipt(path, tolerate_incomplete=False):
        path = Path(path)
        if not path.is_file():
            return None, {'path': str(path), 'status': 'missing'}
        data = path.read_bytes()
        try:
            value = json.loads(data)
        except json.JSONDecodeError:
            if tolerate_incomplete:
                return None, {'path': str(path), 'status': 'writing'}
            raise
        return value, {'path': str(path), 'sha256': hashlib.sha256(data).hexdigest(),
                       'bytes': len(data), 'status': value.get('status', value.get('execution_status'))}

    def unchanged(receipts):
        for receipt in receipts:
            hasher = hashlib.sha256()
            with Path(receipt['path']).open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    hasher.update(chunk)
            if hasher.hexdigest() != receipt['sha256']:
                raise ValueError('raw evidence changed: ' + receipt['path'])

    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    code = 1
    try:
        cases = plan['cases']
        if not cases or len({case['id'] for case in cases}) != len(cases):
            raise ValueError('queue requires distinct nonempty cases')
        if plan.get('initial_evaluation'):
            raise ValueError('evaluation-only uses cases for every existing raw run')
        outputs = set()
        for case in cases:
            actual = Path(case['actual_config'])
            if digest(actual) != case['actual_config_sha256']:
                raise ValueError('actual raw config changed: ' + str(actual))
            config = evaluation_config(case['post_evaluation'], case['id'], actual, plan['shared_lock'])
            if Path(config['evaluated_resources']).resolve() != Path(case['admission_report']).resolve():
                raise ValueError('evaluation admission path differs')
            output = str(Path(config['output']).resolve())
            if output in outputs or Path(output).exists():
                raise ValueError('evaluation requires distinct fresh output: ' + output)
            outputs.add(output)
        report.update(status='running', planned_cases=len(cases))
        save()
        for case in cases:
            if cancelled.is_set():
                break
            row = dict(case=case['id'], status='waiting_raw', rawcomplete=False, evalcomplete=False,
                       execution_succeeded=False, started_utc=now(),
                       evaluation={'status': 'not_started', 'evaluation_succeeded': False})
            report['cases'].append(row)
            report['current_case'] = case['id']
            save()
            try:
                actual_path = Path(case['actual_config'])
                actual, config_receipt = read_receipt(actual_path)
                if config_receipt['sha256'] != case['actual_config_sha256']:
                    raise ValueError('actual raw config changed')
                diagnostics = Path(actual['diagnostic_root'])
                completion_path = diagnostics/'completion.json'
                deadline = time.monotonic() + maximum_wait
                evidence = None
                while not cancelled.is_set():
                    completion, completion_receipt = read_receipt(completion_path, tolerate_incomplete=True)
                    admission, admission_receipt = read_receipt(case['admission_report'], tolerate_incomplete=True)
                    row.update(raw_config=config_receipt, admission=admission_receipt,
                               completion=completion_receipt)
                    failure_statuses = ('failed', 'child_failed', 'interrupted', 'query_or_validation_failed',
                                        'wrapper_failed', 'failed_or_not_admitted')
                    failed = (admission is not None and (admission.get('status') in failure_statuses
                              or admission.get('exit_code') not in (None, 0)))
                    failed = failed or (completion is not None and
                        (completion.get('execution_status') in failure_statuses
                         or completion.get('pipeline_status') in failure_statuses
                         or completion.get('exit_code') not in (None, 0)
                         or completion.get('child_exit_code') not in (None, 0)))
                    if failed:
                        row.update(status='raw_failed', evaluation={
                            'status': 'skipped_algorithm_not_complete', 'evaluation_succeeded': False})
                        break
                    ready = (admission is not None and admission.get('status') == 'complete'
                             and type(admission.get('exit_code')) is int and admission['exit_code'] == 0
                             and completion is not None and completion.get('execution_status') == 'complete'
                             and completion.get('pipeline_status') == 'complete'
                             and type(completion.get('exit_code')) is int and completion['exit_code'] == 0
                             and type(completion.get('child_exit_code')) is int and completion['child_exit_code'] == 0)
                    if ready:
                        # Read the potentially large pipeline JSON only once, after small terminal receipts.
                        pipeline_path = Path(actual['output'])/'fnit-native-free-run.json'
                        pipeline, pipeline_receipt = read_receipt(pipeline_path)
                        validation = pipeline.get('output_validation', {}) if pipeline else {}
                        mesh = pipeline.get('mesh_validation', {}) if pipeline else {}
                        summary = {key: validation.get(key) for key in ('status', 'expected', 'present')}
                        summary['missing_count'] = len(validation.get('missing', []))
                        pipeline_receipt.update(output_validation=summary,
                            mesh_validation={key: mesh.get(key, {}).get('status') if key != 'status'
                                             else mesh.get(key) for key in ('status', 'lh', 'rh')})
                        row['pipeline'] = pipeline_receipt
                        if (not pipeline or pipeline.get('status') != 'complete' or pipeline.get('error')
                            or pipeline.get('failed_stage') or pipeline.get('input') != actual['input']
                            or pipeline.get('subject_dir') != actual['output']
                            or validation.get('status') != 'passed'
                            or type(validation.get('expected')) is not int or validation['expected'] != 138
                            or type(validation.get('present')) is not int or validation['present'] != 138
                            or validation.get('missing') != [] or completion.get('output_validation') != validation
                            or mesh.get('status') != 'passed'
                            or any(mesh.get(hemi, {}).get('status') != 'passed' for hemi in ('lh', 'rh'))):
                            row.update(status='raw_failed', evaluation={
                                'status': 'skipped_algorithm_not_complete', 'evaluation_succeeded': False})
                        else:
                            evidence = [config_receipt, completion_receipt, admission_receipt, pipeline_receipt]
                            row.update(status='raw_complete', rawcomplete=True, execution_succeeded=True)
                        break
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        row.update(status='raw_wait_timeout', evaluation={
                            'status': 'skipped_raw_wait_timeout', 'evaluation_succeeded': False})
                        break
                    save()
                    cancelled.wait(min(poll, remaining))
                save()
                if evidence is not None and not cancelled.is_set():
                    spec = case['post_evaluation']
                    config = evaluation_config(spec, case['id'], actual_path, plan['shared_lock'])
                    unchanged(evidence)
                    if Path(config['output']).exists():
                        raise FileExistsError('evaluation requires fresh output: ' + config['output'])
                    result = row['evaluation'] = dict(status='starting', evaluation_succeeded=False,
                                                       started_utc=now())
                    command = [plan['python'], spec['script'], '--config', spec['config']]
                    log = Path(spec['config']).with_name(Path(spec['config']).name + '.queue-evaluation.log')
                    result.update(command=command, log=str(log))
                    with log.open('x') as stream:
                        if cancelled.is_set():
                            row['status'] = 'interrupted'
                        else:
                            child = subprocess.Popen(command, start_new_session=True, stdout=stream,
                                                     stderr=subprocess.STDOUT)
                            result.update(status='running', pid=child.pid)
                            row['status'] = 'evaluating'
                            save()
                            evaluation_deadline = time.monotonic() + evaluation_timeout
                            while child.poll() is None and not cancelled.is_set():
                                remaining = evaluation_deadline - time.monotonic()
                                if remaining <= 0:
                                    result['timed_out'] = True
                                    break
                                cancelled.wait(min(0.2, remaining))
                            if cancelled.is_set() or result.get('timed_out'):
                                cleanup_owned()
                            else:
                                child.wait()
                                # Clean leaked descendants on normal exit as well.
                                cleanup_owned()
                            result['exit_code'] = child.returncode
                            child = None
                            if cancelled.is_set():
                                result['status'] = 'interrupted'
                                row['status'] = 'interrupted'
                            elif result.get('timed_out'):
                                result['status'] = 'failed_timeout'
                                row['status'] = 'evaluation_failed'
                            else:
                                unchanged(evidence)
                                result.update(evaluation_result(config, spec, result['exit_code']))
                                row.update(status='evaluation_complete', evalcomplete=True)
                    result['finished_utc'] = now()
            except Exception as error:
                if child is not None:
                    cleanup_owned()
                    child = None
                row.update(status='evaluation_failed' if row['rawcomplete'] else 'raw_evidence_failed', error=repr(error))
                row['evaluation'].update(status='failed', evaluation_succeeded=False, error=repr(error))
            finally:
                if cancelled.is_set():
                    row['status'] = 'interrupted'
                    if row['evaluation']['status'] == 'not_started':
                        row['evaluation']['status'] = 'skipped_interrupted'
                row['finished_utc'] = now()
                save()
        report.update(queue_finished=not cancelled.is_set() and len(report['cases']) == len(cases),
                      status='interrupted' if cancelled.is_set() else 'all_attempts_finished')
        report['all_cases_succeeded'] = report['queue_finished'] and all(r['rawcomplete'] for r in report['cases'])
        report['all_evaluations_succeeded'] = report['queue_finished'] and all(r['evalcomplete'] for r in report['cases'])
        code = 130 if cancelled.is_set() else (0 if report['all_evaluations_succeeded'] else 1)
    except Exception as error:
        report.update(status='queue_wrapper_failed', error=repr(error))
        if child is not None:
            cleanup_owned()
            child = None
        code = 130 if cancelled.is_set() else 1
    finally:
        report.update(current_case=None, finished_utc=now(), exit_code=code,
                      successful=sum(r['rawcomplete'] for r in report['cases']),
                      evaluations_succeeded=sum(r['evalcomplete'] for r in report['cases']),
                      failed_or_not_admitted=sum(not r['rawcomplete'] for r in report['cases']),
                      evaluations_failed_or_skipped=sum(not r['evalcomplete'] for r in report['cases']))
        save()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    try:
        initial_plan = json.loads(args.plan.read_text())
    except Exception:
        initial_plan = {}  # Existing roles retain their original failure receipt path.
    if isinstance(initial_plan, dict) and initial_plan.get('execution_role') == 'precision_evaluation_only':
        return precision_evaluation_only(args, initial_plan)
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
        def evaluate(spec, case_id, actual, case_row=None):
            nonlocal child, child_kind
            result = {'case': case_id, 'status': 'starting', 'evaluation_succeeded': False,
                      'started_utc': now()}
            if case_row is not None:
                case_row['evaluation'] = result
                save()
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
                    result.update(status='running', pid=child.pid)
                    if case_row is not None:
                        save()
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
                # Persist actual raw completion before the potentially long CPU evaluation.
                save()
                if case.get('post_evaluation'):
                    row['evaluation'] = (evaluate(case['post_evaluation'], case['id'],
                        str(Path(case['retry_root'])/'retry_config.json'), case_row=row) if success and not interrupted
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
