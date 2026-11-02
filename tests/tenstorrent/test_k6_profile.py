"""Guard the observed SDK accumulator collision and interval accounting."""

from pathlib import Path

from benchmarks.tenstorrent.k6_profile import instrument


def test_selected_unpack_scopes_preserve_sdk_slot_one(tmp_path):
    source = Path(__file__).resolve().parents[2] / "flex_gemm/tenstorrent/kernels/k6"
    for mode, label in (("compute", "WAIT_GATHER"), ("weight", "WAIT_WEIGHT")):
        generated = tmp_path / mode
        instrument(source, generated, mode)
        text = (generated / "k6_compute.cpp").read_text()
        scopes = [line for line in text.splitlines() if "UNPACK(DeviceZoneScopedSum" in line]
        assert len(scopes) == 1
        assert f'DeviceZoneScopedSumN2("FG_K6_{label}")' in scopes[0]
        assert "UNPACK(DeviceZoneScopedSumN1" not in text


def test_gather_barriers_share_one_scope_location(tmp_path):
    source = Path(__file__).resolve().parents[2] / "flex_gemm/tenstorrent/kernels/k6"
    generated = tmp_path / "gather"
    instrument(source, generated, "gather")
    text = (generated / "k6_gather.hpp").read_text()
    assert text.count('DeviceZoneScopedSumN2("FG_K6_GATHER_READ_BARRIERS")') == 1
    assert text.count("    fg_k6_gather_barrier();") == 2
    assert text.count("    noc_async_read_barrier();") == 1
