"""The installation patch is finite and rejects unverified/modified source."""
import unittest
from fnit.recon_all.mri_em_register_native_build import patch_source_text

def test_patch_rejects_unknown_source_shape():
    with unittest.TestCase().assertRaises(AssertionError):patch_source_text('unverified source')

def test_patch_keeps_unrelated_optimizer_and_limits_activation():
    source='''#include "emregisterutils.h"
// complete optimizer remains
      result = GCAcomputeLogSampleProbability( gca, gcas, mri,
\t\t\t\t\t       transform, nsamples, clamp );
// final gradient remains
'''
    patched=patch_source_text(source)
    assert '// complete optimizer remains' in patched and '// final gradient remains' in patched
    assert 'gca->ninputs == 1 && mri->type == MRI_UCHAR && mri->nframes == 1' in patched
    assert 'FNIT_GCA_SCORER' in patched and 'GCAcomputeLogSampleProbability(gca' in patched
    with unittest.TestCase().assertRaises(AssertionError):patch_source_text(patched)

if __name__=='__main__':
    test_patch_rejects_unknown_source_shape();test_patch_keeps_unrelated_optimizer_and_limits_activation();print('2 tests passed')
