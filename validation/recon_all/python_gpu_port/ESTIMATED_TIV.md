# Estimated intracranial volume

`estimated_tiv.py` reads the MNI `talairach.xfm` matrix and replays
FreeSurfer 8.2.0 `MRIestimateTIV`: `1948.106 * 1000 / determinant`.
The source parses coefficients as float32 and returns a float32 determinant;
using a conventional double-precision determinant changes the sixth-decimal
output. The Python implementation reproduces that rounding on CPU.

The frozen `fs_sub01` transform has SHA-256
`1e509f0e544614555ef77ba4df6c8f6ebfa4c73c4771d7464589250dfc607e59`.
Python returned `1310266.152503 mm³`, exactly matching the official
`lh.aparc.stats` eTIV header at its six displayed decimals. The fixture test
also pins this precision and rejects a missing transform matrix; control-host
reported 2/2 tests passing.

This covers only the eTIV global measure. Other brain-volume measures and
the complete statistics file header remain to be ported.
