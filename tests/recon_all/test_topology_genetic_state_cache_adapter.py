"""仅验证benchmark显式身份/异常恢复；真实完整GA另报。"""
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

from test_topology_candidate_cycle import fixture
from test_topology_candidate_state_cache import complete_candidate
from benchmark_recon_topology_genetic_state_cache import explicit_state_cache
from fnit.recon_all import topology_genetic_search as search
from fnit.recon_all import topology_candidate_cycle as cycle
from fnit.recon_all import topology_ga_finalize as final


class StateCacheAdapterTests(unittest.TestCase):
    def test_explicit_current_identity_and_function_restore_on_failure(self):
        initial,_,retess=fixture()
        value=SimpleNamespace(surface_state=initial,state_context=None)
        prepared=dict(evaluator=value,prepare_seconds=0.)
        result=complete_candidate(surface_state=initial,retessellation_inputs=retess)
        old=(search.prepare_genetic_defect_evaluator,cycle.materialize_candidate_state,
             final.evaluate_candidate_cycle,final.materialize_candidate_state)
        with patch.object(search,'prepare_genetic_defect_evaluator',return_value=prepared) as mock:
            mock_identity=search.prepare_genetic_defect_evaluator
            with self.assertRaisesRegex(RuntimeError,'own failed stage'):
                with explicit_state_cache(enabled=True) as audit:
                    with self.assertRaises(ValueError):
                        cycle.materialize_candidate_state(initial=initial,result=result)
                    out=search.prepare_genetic_defect_evaluator(surface_state=initial)
                    self.assertGreater(out['prepare_seconds'],0.)
                    wrong=object()
                    with self.assertRaises(ValueError):
                        cycle.materialize_candidate_state(initial=initial,result=result,state_context=wrong)
                    actual=cycle.materialize_candidate_state(initial=initial,result=result,state_context=value.state_context)
                    np.testing.assert_array_equal(actual['original_vertices'],result['geometry']['original_vertices'])
                    final.materialize_candidate_state(initial=initial,result=result)
                    self.assertEqual(len(audit['preparations']),1)
                    self.assertEqual(len(audit['materializations']),2)
                    self.assertEqual(audit['final_write_calls'],1)
                    raise RuntimeError('own failed stage')
            self.assertIs(search.prepare_genetic_defect_evaluator,mock_identity)
            mock.assert_called_once_with(surface_state=initial)
        self.assertIs(search.prepare_genetic_defect_evaluator,old[0])
        self.assertIs(cycle.materialize_candidate_state,old[1])
        self.assertIs(final.evaluate_candidate_cycle,old[2])
        self.assertIs(final.materialize_candidate_state,old[3])

    def test_disabled_is_identity_and_invalid_flag_fails(self):
        old=(search.prepare_genetic_defect_evaluator,cycle.materialize_candidate_state,
             final.evaluate_candidate_cycle,final.materialize_candidate_state)
        with explicit_state_cache(enabled=False) as audit:
            self.assertEqual(audit['preparations'],[])
            self.assertIs(search.prepare_genetic_defect_evaluator,old[0])
            self.assertIs(cycle.materialize_candidate_state,old[1])
            self.assertIs(final.evaluate_candidate_cycle,old[2])
            self.assertIs(final.materialize_candidate_state,old[3])
        with self.assertRaises(ValueError):
            with explicit_state_cache(enabled=1):pass


if __name__=='__main__':unittest.main()
