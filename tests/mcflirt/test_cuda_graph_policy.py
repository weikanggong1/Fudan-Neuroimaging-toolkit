"""The flag's presence, rather than its text value, controls graph capture."""

import pytest

from fnit.mcflirt._cuda_graph import cuda_graph_capture_enabled


def test_caching_allocator_keeps_graph_capture_enabled(monkeypatch):
    monkeypatch.delenv("PYTORCH_NO_CUDA_MEMORY_CACHING", raising=False)
    assert cuda_graph_capture_enabled()


@pytest.mark.parametrize("value", ["", "0", "1"])
def test_uncached_allocator_flag_presence_disables_capture(monkeypatch, value):
    monkeypatch.setenv("PYTORCH_NO_CUDA_MEMORY_CACHING", value)
    assert not cuda_graph_capture_enabled()
