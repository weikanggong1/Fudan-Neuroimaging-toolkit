"""Build the bundled GEMS extension using packages from the active conda env."""

from pathlib import Path
import os
import subprocess
import sys


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    prefix = Path(sys.prefix)
    itk = sorted((prefix / "lib" / "cmake").glob("ITK-*"))
    if not itk:
        raise RuntimeError("当前 conda 环境缺少 libitk-devel；请使用 environment-gems-native.yml")
    source = root / "third_party" / "samseg_gems"
    build = root / ".build" / "samseg_gems_conda"
    output = root / "src" / "fnit" / "gems" / "native_samseg" / "gems"
    output.mkdir(parents=True, exist_ok=True)
    compiler = prefix / "bin" / "x86_64-conda-linux-gnu-cc"
    cxx_compiler = prefix / "bin" / "x86_64-conda-linux-gnu-c++"
    if not compiler.is_file() or not cxx_compiler.is_file():
        raise RuntimeError("当前 conda 环境缺少 gcc_linux-64 或 gxx_linux-64")
    subprocess.run([
        "cmake", "-S", str(source), "-B", str(build),
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_C_COMPILER={compiler}",
        f"-DCMAKE_CXX_COMPILER={cxx_compiler}",
        f"-DCMAKE_PREFIX_PATH={prefix}",
        f"-DITK_DIR={itk[-1]}",
        f"-DPython3_EXECUTABLE={sys.executable}",
        f"-DFNIT_GEMS_OUTPUT_DIR={output}",
    ], check=True)
    subprocess.run([
        "cmake", "--build", str(build), "--target", "gemsbindings", "gems_resample", "gems_warp",
        "--parallel", str(min(os.cpu_count() or 1, 8)),
    ], check=True)
    sys.path.insert(0, str(root / "src"))
    from fnit.gems.native_samseg.gems import gemsbindings, gems_resample, gems_warp
    print(f"GEMS 扩展已编译：{gemsbindings.__file__}")
    print(f"GEMS 重采样扩展已编译：{gems_resample.__file__}")
    print(f"GEMS 配准插值扩展已编译：{gems_warp.__file__}")


if __name__ == "__main__":
    main()
