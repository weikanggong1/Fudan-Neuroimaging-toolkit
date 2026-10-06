"""Explicit experimental CLI, separate from all production FNIT entrypoints."""
import runpy

from load_candidate import load_candidate


if __name__ == '__main__':
    package = load_candidate()
    runpy.run_module(package.__name__ + '.__main__', run_name='__main__')
