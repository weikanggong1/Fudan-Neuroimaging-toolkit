# Six-target Conda C++ build on gpucw1

The current script `tools/build_recon_all_fs_cpp_conda.sh` compiled the six required/optional CPU commands from FreeSurfer 8.2 source commit `d932c45b7941662ea380a05efef580568b98d41a`, using only the active Conda C/C++/Fortran compilers, CMake, Ninja and libraries. Source was the pinned archive with verified tree SHA-256 recorded in [build provenance](build-provenance.txt); its one build-only CMake edit honors the Conda Python path. The user does not need an installed FreeSurfer runtime.

Build on gpucw1, glibc 2.17: exit 0, elapsed wall 3:08.62 (188.62 s), maximum build-driver RSS 426,020 KiB. The six resulting files are listed with SHA-256 in [bin.sha256](bin.sha256): `mri_em_register`, `mri_segment`, `mri_edit_wm_with_aseg`, `mris_fix_topology`, `mris_inflate`, `mris_place_surface`. All six have `ldd` records without links to the cluster FreeSurfer installation or external temporary ITK tree, and launch checks found no missing GLIBC/shared-library errors. The build driver alone is timed here; its timing is not a reconstruction benchmark.

The full environment YAML was solver checked under glibc 2.17; this actual build used the previously created equivalent Conda environment, rather than a newly created YAML environment. The license is supplied privately at runtime and does not appear in build outputs.
