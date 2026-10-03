"""Final export requires real complete unique tables; CPU metadata fixtures."""
import importlib.util
from pathlib import Path
import sys
import unittest
source=Path(__file__).resolve().parents[2]/'tools'
sys.path.insert(0,str(source))
import run_connectome_actual_mixed_final_cpu as chain


class ExportGate(unittest.TestCase):
    def test_pending_or_failed_waiter_never_exports(self):
        for status in ('pending_actual_v4_completed_subset','failed_metadata_or_frozen_CPU_reader'):
            with self.subTest(status=status),self.assertRaises(ValueError):chain.require_completed_tables({'status':status},{})

    def test_complete_flag_without_coverage_is_rejected(self):
        waiter={'status':'completed_actual_CPU_comparison_and_tables','matrix_checks':320,'FS_case_comparisons':10}
        summary={'status':'complete_actual_ten_case_tables','ready_for_ten_case_render':True,'completed_pairs':10,'matrix_rows':[],'anatomy_rows':[]}
        with self.assertRaises(ValueError):chain.require_completed_tables(waiter,summary)

    def test_unique_ten_by_thirtytwo_and_thirteen_required(self):
        waiter={'status':'completed_actual_CPU_comparison_and_tables','matrix_checks':320,'FS_case_comparisons':10}
        summary={'status':'complete_actual_ten_case_tables','ready_for_ten_case_render':True,'completed_pairs':10,
            'matrix_rows':[{'case_id':str(c),'atlas':str(a),'kind':str(m)} for c in range(10) for a in range(8) for m in range(4)],
            'anatomy_rows':[{'case_id':str(c),'file':str(f)} for c in range(10) for f in range(13)]}
        chain.require_completed_tables(waiter,summary)
        summary['matrix_rows'][-1]=summary['matrix_rows'][0]
        with self.assertRaises(ValueError):chain.require_completed_tables(waiter,summary)


if __name__=='__main__':unittest.main()
