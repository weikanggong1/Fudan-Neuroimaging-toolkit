"""Timing metadata contracts against fixed fMRIPrep's documented examples."""

from copy import deepcopy

import pytest

from fnit.fmri.timing import prepare_timing_parameters


@pytest.mark.parametrize("metadata, corrected, expected", [
    ({"RepetitionTime": 2}, False,
     {"RepetitionTime": 2, "SliceTimingCorrected": False}),
    ({"RepetitionTime": 2, "DelayTime": .5}, False,
     {"RepetitionTime": 2, "DelayTime": .5, "SliceTimingCorrected": False}),
    ({"VolumeTiming": [0., 1., 2., 5., 6., 7.], "AcquisitionDuration": 1.}, False,
     {"VolumeTiming": [0., 1., 2., 5., 6., 7.], "AcquisitionDuration": 1.,
      "SliceTimingCorrected": False}),
    ({"RepetitionTime": 2, "SliceTiming": [0., .2, .4, .6]}, True,
     {"RepetitionTime": 2, "DelayTime": 1.2, "SliceTimingCorrected": True, "StartTime": .3}),
    ({"RepetitionTime": 2, "SliceTiming": [0., .2, .4, .6]}, False,
     {"RepetitionTime": 2, "DelayTime": 1.2, "SliceTimingCorrected": False}),
    ({"VolumeTiming": [0., 1., 2., 5., 6., 7.], "SliceTiming": [0., .2, .4, .6, .8]}, True,
     {"VolumeTiming": [0., 1., 2., 5., 6., 7.], "AcquisitionDuration": 1.,
      "SliceTimingCorrected": True, "StartTime": .4}),
    ({"VolumeTiming": [0., 1., 2., 5., 6., 7.], "SliceTiming": [0., .2, .4, .6, .8]}, False,
     {"VolumeTiming": [0., 1., 2., 5., 6., 7.], "AcquisitionDuration": 1.,
      "SliceTimingCorrected": False}),
    ({"RepetitionTime": 2, "SliceTiming": []}, True,
     {"RepetitionTime": 2, "SliceTimingCorrected": False}),
    ({"RepetitionTime": 2, "SliceTiming": [0.]}, True,
     {"RepetitionTime": 2, "SliceTimingCorrected": False}),
])
def test_locked_fmriprep_timing_examples(metadata, corrected, expected):
    original = deepcopy(metadata)
    actual = prepare_timing_parameters(metadata, slice_timing_corrected=corrected)
    assert actual == expected
    assert metadata == original
    assert "SliceTiming" not in actual


def test_multiband_onsets_and_explicit_reference_match_upstream():
    # Upstream sorts onsets, including repeated onsets, before inferring TA.
    actual = prepare_timing_parameters(
        {"RepetitionTime": 1.5, "SliceTiming": [.8, 0., .8, 0., .4, .4]},
        slice_timing_corrected=True, reference_fraction=.25,
    )
    assert actual == {"RepetitionTime": 1.5, "DelayTime": .7,
                      "SliceTimingCorrected": True, "StartTime": .2}


def test_full_tr_does_not_fabricate_delay_and_start_is_millisecond_rounded():
    actual = prepare_timing_parameters(
        {"RepetitionTime": 1., "SliceTiming": [.05, .3, .55, .75]},
        slice_timing_corrected=True, reference_fraction=.333,
    )
    assert "DelayTime" not in actual
    assert actual["StartTime"] == .283


def test_volume_timing_output_does_not_alias_raw_metadata():
    metadata = {"VolumeTiming": [0., 2., 5.], "AcquisitionDuration": 1.}
    actual = prepare_timing_parameters(metadata, slice_timing_corrected=False)
    actual["VolumeTiming"][0] = 1.
    assert metadata["VolumeTiming"][0] == 0.


@pytest.mark.parametrize("metadata", [
    {"RepetitionTime": float("nan")},
    {"DelayTime": -1},
    {"AcquisitionDuration": float("inf")},
    {"VolumeTiming": [0., 1., 1.]},
    {"RepetitionTime": 2, "SliceTiming": [0., float("nan")]},
])
def test_invalid_timing_cannot_be_published_as_json(metadata):
    with pytest.raises(ValueError):
        prepare_timing_parameters(metadata, slice_timing_corrected=False)
