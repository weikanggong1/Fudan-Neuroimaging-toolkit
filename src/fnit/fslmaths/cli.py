"""Command line interface for the supported fslmaths subset."""

import argparse
import sys

from .core import run_fslmaths


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="fnit-fslmaths", description="PyTorch fslmaths common operations",
        usage="fnit-fslmaths [--device cpu|cuda:0] [-dt dtype] input [operations] output [-odt dtype]")
    tokens = list(sys.argv[1:] if argv is None else argv)
    if not tokens or tokens[0] in ("-h", "--help"):
        parser.print_help()
        return
    device = "cpu"
    if tokens[:1] == ["--device"]:
        if len(tokens) < 2:
            parser.error("--device requires a value")
        device, tokens = tokens[1], tokens[2:]
    input_dtype = None
    output_dtype = None
    if tokens[:1] in (["-dt"], ["-datatype"]):
        if len(tokens) < 2:
            parser.error("-dt requires a datatype")
        input_dtype, tokens = tokens[1], tokens[2:]
    if len(tokens) >= 2 and tokens[-2] in ("-odt", "-output_datatype"):
        output_dtype, tokens = tokens[-1], tokens[:-2]
    if len(tokens) < 2:
        parser.error("input and output images are required")
    print(run_fslmaths(tokens[0], tokens[1:-1], tokens[-1], device=device,
                       input_dtype=input_dtype, output_dtype=output_dtype))


if __name__ == "__main__":
    main()
