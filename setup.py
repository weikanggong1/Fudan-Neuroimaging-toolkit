import os

from setuptools import Extension, setup


setup(
    ext_modules=[
        Extension(
            "fnit.msm._fastpd_native",
            sources=["src/fnit/msm/_fastpd_src/fastpd_module.cpp"],
            include_dirs=["src/fnit/msm/_fastpd_src"],
            language="c++",
            # Source WLS and HOCR require separately rounded products/sums.
            extra_compile_args=["/O2", "/std:c++17", "/fp:strict"] if os.name == "nt"
            else ["-O3", "-std=c++17", "-fno-fast-math", "-ffp-contract=off"],
        )
    ]
)
