"""缓存完成回执的身份门；小JSON不是GA或MRI benchmark。"""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from export_recon_topology_genetic_face_cache import export


class FaceCacheBindingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.run, self.source = root / 'run', root / 'frozen'
        self.candidate = self.run / 'candidate'
        self.candidate.mkdir(parents=True)
        self.source.mkdir()
        self.output = root / 'public'
        self.write(self.candidate / 'optimization.json', dict(strict_optimization_pair_passed=True))
        self.write(self.candidate / 'OPTIMIZATION_COMPLETE.json', dict(status='complete_optimized_first_default_GA'))
        self.face = dict(status='complete_default_first_GA_face_cache', strict_complete_pair_passed=True,
            inherited_complete_sha256=self.sha(self.candidate / 'OPTIMIZATION_COMPLETE.json'),
            optimization_report_sha256=self.sha(self.candidate / 'optimization.json'),
            helper_sha256='a'*64, actual_tool_sha256='b'*64)
        self.write(self.source / 'FACE_CACHE_OVERLAY_SHA.json', dict(payload={
            'src/fnit/recon_all/topology_face_detection_cache.py':dict(sha256='a'*64),
            'tools/benchmark_recon_topology_genetic_face_cache.py':dict(sha256='b'*64)}))
        self.bind()

    @staticmethod
    def write(path,value):
        path.write_text(json.dumps(value,indent=2)+'\n')

    @staticmethod
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def bind(self):
        self.write(self.candidate / 'face_cache.json',self.face)
        self.write(self.candidate / 'FACE_CACHE_COMPLETE.json',dict(status=self.face['status'],
            face_cache_report_sha256=self.sha(self.candidate / 'face_cache.json'),
            inherited_complete_sha256=self.face['inherited_complete_sha256']))

    def rejected(self,pattern):
        with self.assertRaisesRegex(ValueError,pattern):
            export(input_run=self.run,baseline_run=self.run,frozen_source=self.source,
                output_directory=self.output)
        self.assertFalse(self.output.exists())

    def test_old_inherited_complete_cannot_bind_changed_output(self):
        self.write(self.candidate / 'OPTIMIZATION_COMPLETE.json',dict(status='different_run'))
        self.rejected('same complete face-cache')

    def test_other_adapter_sha_cannot_supply_cached_claim(self):
        self.face['actual_tool_sha256']='c'*64
        self.bind()
        self.rejected('same actually measured face-cache')


if __name__ == '__main__':
    unittest.main()
