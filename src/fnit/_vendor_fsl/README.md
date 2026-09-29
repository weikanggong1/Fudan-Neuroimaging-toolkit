# FSL source snapshots used by the PyTorch ports

This directory contains unmodified FSL source-code snapshots consulted while
implementing the PyTorch FLIRT, FNIRT, TOPUP, EDDY, DTIFIT, `applywarp`, BET,
BEDPOSTX and ProbtrackX paths. The snapshots are retained for licence compliance and
reproducible source provenance. They are package data: `fnit` does not compile
or import them at runtime.

FLIRT through fugue match the FSL 6.0.7.4 validation package
manifest. The EDDY, DTIFIT, BEDPOSTX, ProbtrackX, and BET rows are separate
implementation reference snapshots:

| component | tag | commit |
| --- | --- | --- |
| FLIRT | `2111.2` | `5036b4620ea97db0050f2dc132fbb331dbba060c` |
| FNIRT | `2203.0` | `27f514a182b5972094e30d8ea79f4fad89cbf03d` |
| TOPUP | `2203.2` | `3e2cb9104e834ce18c10e4b7edddbd500d0c459c` |
| basisfield | `2203.1` | `9588bbe8eb8aa0939ddefd00df756aeb80d2305b` |
| miscmaths | `2203.2` | `7824d74cdfa9fb65de178f642c3c05e57c8c8868` |
| newimage | `2203.11` | `19e3ddd10138d8ea1394fd522fb0770435c61ddd` |
| warpfns | `2203.0` | `50ea45cb0b9661adba7844444cb38649ae44892b` |
| fugue | `2201.3` | `9d815181a19c4fe1aebac74e9fa6601cde4e1ded` |
| EDDY | `2111.0` | `ecfef26151c2613d0f4e1b45dcbbe100b58db50c` |
| fdt (DTIFIT) | `2202.6` | `f0287f09f09dc34e24b95c471127239be69b4022` |
| fdt (BEDPOSTX) | `2604.0` | `03e2b6bd88423e77386356a4c75b14cca5d90c6c` |
| ptx2 (ProbtrackX) | `2608.0` | `900e72c451c556d24d2629f9c2ffff12b2ee7bfc` |
| bet2 (BET) | `2111.9` | `d6b02000500516ce7d1c0c9fa23259ca7a83f7e3` |
| meshclass (BET) | `2111.0` | `228ca8e73b86b4466e4323be10da36bd5bfdfb07` |
| avwutils (BET) | `2209.8` | `fcc335218284a2b04d5d7eb17dd4cfba264d0295` |

[`manifest.json`](manifest.json) records the upstream repository, tag, commit,
Git tree, deterministic `git archive` SHA-256, and SHA-256 of every distributed
source file. The FSL 6.0.7.4 baseline applies only to rows marked
`validation_target`. All fdt and ptx2 source files are byte-identical to
their commits. Three ptx2 NIfTI test-data LFS pointers and four avwutils
NIfTI test fixtures are omitted because binary MRI files are not distributed
inside the Python package; their paths are recorded in `manifest.json`. The
original repositories are:

- <https://git.fmrib.ox.ac.uk/fsl/flirt.git>
- <https://git.fmrib.ox.ac.uk/fsl/fnirt.git>
- <https://git.fmrib.ox.ac.uk/fsl/topup.git>
- <https://git.fmrib.ox.ac.uk/fsl/basisfield.git>
- <https://git.fmrib.ox.ac.uk/fsl/miscmaths.git>
- <https://git.fmrib.ox.ac.uk/fsl/newimage.git>
- <https://git.fmrib.ox.ac.uk/fsl/warpfns.git>
- <https://git.fmrib.ox.ac.uk/fsl/fugue.git>
- <https://git.fmrib.ox.ac.uk/fsl/eddy.git>
- <https://git.fmrib.ox.ac.uk/fsl/fdt.git>
- <https://git.fmrib.ox.ac.uk/fsl/ptx2.git>
- <https://git.fmrib.ox.ac.uk/fsl/bet2.git>
- <https://git.fmrib.ox.ac.uk/fsl/meshclass.git>
- <https://git.fmrib.ox.ac.uk/fsl/avwutils.git>

These sources and the modified Python ports are distributed under the
[FSL Software Licence, Release 6.0](../../../licenses/FSL-6.0.txt). The licence
permits redistribution without financial return when its conditions are passed
to recipients and all original and amended source code is included. It does
not permit commercial use. This project is not an official FSL release.
