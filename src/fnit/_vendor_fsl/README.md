# FSL source snapshots used by the PyTorch ports

This directory contains complete, unmodified source snapshots for the FSL
components consulted while implementing the PyTorch FLIRT, FNIRT, TOPUP, and
`applywarp` paths. The snapshots are retained for licence compliance and
reproducible source provenance. They are package data: `fnit` does
not compile or import them at runtime.

The validation target is FSL 6.0.7.4. Its Linux package manifest fixes these
component versions:

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

[`manifest.json`](manifest.json) records the upstream repository, tag, commit,
Git tree, deterministic `git archive` SHA-256, and SHA-256 of every distributed
source file. The original repositories are:

- <https://git.fmrib.ox.ac.uk/fsl/flirt.git>
- <https://git.fmrib.ox.ac.uk/fsl/fnirt.git>
- <https://git.fmrib.ox.ac.uk/fsl/topup.git>
- <https://git.fmrib.ox.ac.uk/fsl/basisfield.git>
- <https://git.fmrib.ox.ac.uk/fsl/miscmaths.git>
- <https://git.fmrib.ox.ac.uk/fsl/newimage.git>
- <https://git.fmrib.ox.ac.uk/fsl/warpfns.git>
- <https://git.fmrib.ox.ac.uk/fsl/fugue.git>

These sources and the modified Python ports are distributed under the
[FSL Software Licence, Release 6.0](../../../licenses/FSL-6.0.txt). The licence
permits redistribution without financial return when its conditions are passed
to recipients and all original and amended source code is included. It does
not permit commercial use. This project is not an official FSL release.
