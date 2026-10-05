"""纯 JSON 发布守卫回归；账号与地址均由虚构测试片段构造。"""
import importlib.util
import json
from pathlib import Path
import unittest

HERE = Path(__file__).parent
SPEC = importlib.util.spec_from_file_location('v4_report_privacy', HERE/'build_v4_report.py')
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


def address(*octets):
    return '.'.join(map(str, octets))


class PublicReportPrivacyTests(unittest.TestCase):
    def test_existing_anonymous_numeric_report_is_unchanged(self):
        report_bytes = (HERE/'report.public.json').read_bytes()
        serialized = report_bytes.decode('utf-8')
        before = json.loads(serialized)
        self.assertIsNone(REPORT.ensure_public_report(serialized))
        self.assertEqual(json.loads(serialized), before)
        self.assertEqual((HERE/'report.public.json').read_bytes(), report_bytes)

    def test_anonymous_json_and_public_address_are_unchanged(self):
        payload = {'status': 'complete', 'wall_seconds': 12.5, 'account': 'anonymous',
                   'documentation_address': address(203, 0, 113, 17)}
        serialized = json.dumps(payload, ensure_ascii=False)
        self.assertIsNone(REPORT.ensure_public_report(serialized))
        self.assertEqual(json.loads(serialized), payload)

    def test_generic_account_pattern_is_rejected_without_echo(self):
        # 这些号码为本测试虚构，不对应服务器用户。
        for account in ('gwk_'+'999000111', 'GWK_'+'999000112', 'prefix-gwk_'+'999000113-extra'):
            serialized = json.dumps({'account': account})
            with self.assertRaises(ValueError) as caught:
                REPORT.ensure_public_report(serialized)
            self.assertNotIn(account, str(caught.exception))

    def test_all_rfc1918_ranges_are_rejected_without_echo(self):
        # 仅构造虚构测试网段；不保存真实服务器地址。
        for octets in ((10, 99, 88, 77), (172, 16, 99, 77),
                       (172, 31, 99, 77), (192, 168, 99, 77)):
            value = address(*octets)
            with self.assertRaises(ValueError) as caught:
                REPORT.ensure_public_report(json.dumps({'location': value}))
            self.assertNotIn(value, str(caught.exception))

    def test_neighbor_ranges_invalid_octets_and_versions_are_allowed(self):
        for octets in ((172, 15, 99, 77), (172, 32, 99, 77),
                       (192, 169, 99, 77), (10, 999, 88, 77)):
            self.assertIsNone(REPORT.ensure_public_report(json.dumps({'value': address(*octets)})))
        self.assertIsNone(REPORT.ensure_public_report(json.dumps({'version': '1.2.3.4.5'})))

    def test_existing_path_license_and_command_guards_are_preserved(self):
        for value in ('/cwStorage/example/file', '/public/example/file', '/home/example/file',
                      '/mnt/example/file', '/tmp/example/file', 'FS_LICENSE',
                      'BEGIN PRIVATE KEY', 'PYTHONPATH', 'job_command_basenames_only'):
            with self.assertRaises(ValueError):
                REPORT.ensure_public_report(json.dumps({'metadata': value}))


if __name__ == '__main__':
    unittest.main()
