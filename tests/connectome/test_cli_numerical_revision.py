"""Old complete matrices must not skip newly corrected numerical code."""
from types import SimpleNamespace

from fnit.cli import _connectome_run_options
from fnit.connectome.bids import _fingerprint, _record, _reusable


def test_complete_old_matrix_cache_requires_new_numerical_revision(tmp_path):
    input_path, matrix_path = tmp_path / "input", tmp_path / "matrix.csv"
    input_path.write_bytes(b"unchanged-input")
    matrix_path.write_text("0,1\n1,0\n")
    args = SimpleNamespace(n_seeds=100000, seed=0, device="cuda:0",
                           eddy_gp_seed=12345, shell_bvals=None, compile_arc=False)
    options = _connectome_run_options(args, ("fs-aparc",))
    old_options = {name: value for name, value in options.items() if name != "numerical_revision"}
    state = tmp_path / "run_state.json"
    _record(state, _fingerprint((input_path,), old_options))
    current_key = _fingerprint((input_path,), options)
    assert not _reusable(state, current_key, (matrix_path,))
    # File existence alone is insufficient; an explicitly current completed
    # state can reuse its unchanged inputs and outputs.
    _record(state, current_key)
    assert _reusable(state, current_key, (matrix_path,))
