"""Evaluator checks exercise independent oracles and adversarial failure paths."""

import json
from pathlib import Path
import sys

import torch

from benchmarks.tenstorrent.common import correctness, neighbor_oracle, reference, work_accounting
from benchmarks.tenstorrent.run import monitor


CONTRACT = json.loads((Path(__file__).resolve().parents[2] /
    "benchmarks/tenstorrent/contract.json").read_text())


def test_oracle_against_dense_and_coordinate_permutation():
    torch.manual_seed(17)
    coords = torch.tensor([[1, 1, 0, 1], [0, 0, 0, 0], [0, 1, 1, 0], [1, 0, 1, 1]])
    features, weight = torch.randn(4, 3).bfloat16(), torch.randn(5, 3, 3, 3, 3).bfloat16()
    mapping = neighbor_oracle(coords)
    dense = torch.zeros(2, 3, 2, 2, 2)
    for (b, x, y, z), f in zip(coords, features):
        dense[b, :, x, y, z] = f.float()
    output = torch.nn.functional.conv3d(dense, weight.float().permute(0, 4, 1, 2, 3), padding=1)
    expected = torch.stack([output[b, :, x, y, z] for b, x, y, z in coords])
    torch.testing.assert_close(reference(features, weight, mapping), expected)
    permutation = torch.tensor([2, 0, 3, 1])
    torch.testing.assert_close(reference(features[permutation], weight,
        neighbor_oracle(coords[permutation])), expected[permutation])


def test_correctness_rejects_scaled_nan_stale_and_wrong_shape():
    expected = torch.arange(1., 17.).reshape(4, 4)
    assert correctness(expected.bfloat16(), expected, CONTRACT)["passed"]
    for actual in (expected * 2, expected + 1, expected * float("nan"), expected[:2]):
        assert not correctness(actual, expected, CONTRACT)["passed"]


def test_accounting_full_tile_and_empty_split():
    dense = torch.zeros(32, 1, dtype=torch.long)
    report = work_accounting(dense, 32, 32, algorithm="implicit_gemm")
    assert report["useful_flops"] == report["issued_flops"] == 2 * 32**3
    assert report["requested_feature_bytes"] == report["requested_weight_bytes"] == 2048
    assert report["requested_map_bytes"] == 128
    empty = work_accounting(torch.full((33, 3), -1), 17, 45,
                            algorithm="masked_implicit_gemm_splitk", split_k=3)
    assert empty["useful_flops"] == 0
    assert empty["dummy_tile_products"] == 2 * 2 * 3
    assert empty["issued_flops"] == 12 * 2 * 32**3
    assert empty["requested_feature_bytes"] == empty["requested_map_bytes"] == 0


def test_accounting_split_boundaries_conserve_products_and_feature_payload():
    mapping = torch.tensor([[0, -1, 1]] * 32)
    unsplit = work_accounting(mapping, 33, 35, algorithm="masked_implicit_gemm")
    split = work_accounting(mapping, 33, 35, algorithm="masked_implicit_gemm_splitk", split_k=3)
    assert split["issued_flops"] - split["dummy_tile_products"] * 2 * 32**3 == unsplit["issued_flops"]
    assert split["requested_feature_bytes"] == unsplit["requested_feature_bytes"]
    assert split["split_partial_bytes"] == 3 * 32 * 64 * 4


def test_explicit_accounting_includes_chunk_padding():
    mapping = torch.zeros(65, 3, dtype=torch.long)
    r = work_accounting(mapping, 17, 35, algorithm="explicit_gemm", row_chunk=33)
    assert r["issued_flops"] == (64 + 32) * 64 * (3 * 32) * 2
    assert r["issued_scope"] == "logical_padded_gemm"
    assert r["dram_bus_bytes"] is None


def test_runner_timeout_and_exit_failure_are_terminal(tmp_path):
    with (tmp_path / "child.log").open("w") as log:
        timeout = monitor([sys.executable, "-c", "import time; time.sleep(30)"],
            log=log, report_path=tmp_path / "missing.json", compile_timeout=.2, execution_timeout=.2)
    assert timeout["timeout"] == "compile_or_prepare_timeout"
    assert timeout["returncode"] != 0
    assert timeout["device_recovery_required"]
    with (tmp_path / "failure.log").open("w") as log:
        failure = monitor([sys.executable, "-c", "raise SystemExit(3)"],
            log=log, report_path=tmp_path / "missing.json", compile_timeout=5, execution_timeout=5)
    assert failure["returncode"] == 3 and failure["timeout"] is None
