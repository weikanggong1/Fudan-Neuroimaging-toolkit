"""Official SynthSeg 2.0 volumetric cortical-parcellation label order."""

from __future__ import annotations

import numpy as np

# Byte-for-byte values of FreeSurfer v8.2.0 synthseg_parcellation_labels.npy.
PARCELLATION_LABELS = np.asarray([
    0,
    1001, 1002, 1003, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012,
    1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024,
    1025, 1026, 1027, 1028, 1029, 1030, 1031, 1032, 1033, 1034, 1035,
    2001, 2002, 2003, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012,
    2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024,
    2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035,
], dtype=np.int64)

_DK_NAMES = (
    "bankssts", "caudalanteriorcingulate", "caudalmiddlefrontal", "cuneus",
    "entorhinal", "fusiform", "inferiorparietal", "inferiortemporal",
    "isthmuscingulate", "lateraloccipital", "lateralorbitofrontal", "lingual",
    "medialorbitofrontal", "middletemporal", "parahippocampal", "paracentral",
    "parsopercularis", "parsorbitalis", "parstriangularis", "pericalcarine",
    "postcentral", "posteriorcingulate", "precentral", "precuneus",
    "rostralanteriorcingulate", "rostralmiddlefrontal", "superiorfrontal",
    "superiorparietal", "superiortemporal", "supramarginal", "frontalpole",
    "temporalpole", "transversetemporal", "insula",
)

PARCELLATION_NAMES = (
    "Background",
    *(f"ctx-lh-{name}" for name in _DK_NAMES),
    *(f"ctx-rh-{name}" for name in _DK_NAMES),
)

PARCELLATION_NAME_BY_ID = dict(zip(PARCELLATION_LABELS.tolist(), PARCELLATION_NAMES))
