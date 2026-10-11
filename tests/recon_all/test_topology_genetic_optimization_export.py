"""报告身份控制门；小JSON不是MRI/GA benchmark。"""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from export_recon_topology_genetic_optimization import export_metrics


class CompletedReportBindingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.run = self.root / 'candidate-run'
        self.old = self.root / 'old-run'
        self.source = self.root / 'frozen'
        for p in (self.run / 'candidate', self.old / 'candidate', self.source):
            p.mkdir(parents=True)
        self.output = self.root / 'public'
        self.old_stage = dict(status='complete_self_generated_first_defect_search')
        self._write(self.old / 'candidate/report.json', self.old_stage)
        self._write(self.old / 'process.private.json', dict(exit_code=0))
        self._write(self.old / 'candidate/COMPLETE.json', dict(status=self.old_stage['status'],
            report_sha256=self._sha(self.old / 'candidate/report.json')))
        self._write(self.source / 'SOURCE_MANIFEST.private.json', dict(payload={}))
        self._write(self.source / 'protocol.private.json', dict(
            baseline_report_sha256=self._sha(self.old / 'candidate/report.json')))
        self._write(self.run / 'candidate/report.json', dict())
        self.report = dict(status='complete_optimized_first_default_GA',
            strict_optimization_pair_passed=True,
            protocol_sha256=self._sha(self.source / 'protocol.private.json'),
            source_manifest_sha256=self._sha(self.source / 'SOURCE_MANIFEST.private.json'))
        self._write(self.run / 'candidate/optimization.json', self.report)
        self._write(self.run / 'process.private.json', dict(exit_code=0,
            source_manifest_sha256=self.report['source_manifest_sha256']))
        self.complete = dict(status=self.report['status'], complete_output_sha256={},
            optimization_report_sha256=self._sha(self.run / 'candidate/optimization.json'),
            original_report_sha256=self._sha(self.run / 'candidate/report.json'))
        self._write(self.run / 'candidate/OPTIMIZATION_COMPLETE.json', self.complete)

    @staticmethod
    def _write(path, value):
        path.write_text(json.dumps(value, indent=2)+'\n')

    @staticmethod
    def _sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _rejected(self, message):
        with self.assertRaisesRegex(ValueError, message):
            export_metrics(input_run=self.run, baseline_run=self.old,
                frozen_source=self.source, output_directory=self.output)
        self.assertFalse(self.output.exists(), 'must not publish mixed timing receipts')

    def test_alternate_baseline_report_cannot_supply_wrong_timings(self):
        self._write(self.old / 'candidate/report.json', dict(status=self.old_stage['status'],other_run=True))
        self._rejected('same completed baseline binding')

    def test_failed_baseline_process_is_not_a_timing_control(self):
        self._write(self.old / 'process.private.json', dict(exit_code=1))
        self._rejected('same completed baseline binding')

    def test_receipt_status_must_match_measured_report(self):
        self.complete['status'] = 'failed_optimized_first_default_GA'
        self._write(self.run / 'candidate/OPTIMIZATION_COMPLETE.json', self.complete)
        self._rejected('completed full first default GA')

    def test_protocol_changed_after_measurement_rejects_publication(self):
        self._write(self.source / 'protocol.private.json', dict(baseline_report_sha256='0'*64))
        self._rejected('same measured immutable protocol')


if __name__ == '__main__':
    unittest.main()
