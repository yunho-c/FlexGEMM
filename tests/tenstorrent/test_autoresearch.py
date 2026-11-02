"""Evaluator checks exercise independent oracles and adversarial failure paths."""

import json
from pathlib import Path
import sys

import torch

from benchmarks.tenstorrent.common import correctness, neighbor_oracle, reference, work_accounting
from benchmarks.tenstorrent.run import monitor
from benchmarks.tenstorrent.profile import parse_device_csv
from benchmarks.tenstorrent.sweep import comparison, summarize_comparisons
from benchmarks.tenstorrent.memory import workspace_bound


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


def test_tiled_gather_accounts_for_one_feature_read_per_edge():
    mapping = torch.tensor([[0, -1, 1]] * 65)
    implicit = work_accounting(mapping, 33, 65, algorithm="implicit_gemm")
    fused = work_accounting(mapping, 33, 65, algorithm="tiled_gather_gemm", row_chunk=64)
    assert fused["requested_feature_bytes"] * 3 == implicit["requested_feature_bytes"]
    assert fused["requested_map_bytes"] == 3 * 3 * 128
    assert fused["gather_output_bytes"] == 96 * 3 * 64 * 2
    assert fused["requested_weight_bytes"] is None  # TTNN matmul scheduling is unknown.
    assert fused["model"] == "tiled_gather_v1"


def test_digit_accounting_charges_all_four_products_and_both_gathers():
    mapping = torch.zeros(65, 3, dtype=torch.long)
    base = work_accounting(mapping, 17, 35, algorithm="explicit_gemm", row_chunk=32)
    digit = work_accounting(mapping, 17, 35, algorithm="digit_explicit_gemm", row_chunk=32,
                            digit_output_assembly="tile_copy")
    assert digit["useful_flops"] == base["useful_flops"]
    assert digit["issued_flops"] == base["issued_flops"] * 4
    assert digit["explicit_expansion_bytes"] == base["explicit_expansion_bytes"] * 2
    assert digit["model"] == "digit_explicit_v2"
    assert digit["output_assembly_copy_bytes"] == 3 * 2 * 1024 * 4
    concat = work_accounting(mapping, 17, 35, algorithm="digit_explicit_gemm", row_chunk=32)
    assert concat["model"] == "digit_explicit_v1" and concat["output_assembly_copy_bytes"] == 0


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


def test_profile_uses_critical_path_and_reports_missing_events(tmp_path):
    path = tmp_path / "device.csv"
    path.write_text("ARCH: wormhole_b0, CHIP_FREQ[MHz]: 1000\n"
        "PCIe slot,core_x,core_y,RISC processor type,timer_id,time[cycles since reset],data,run host ID,zone name,type\n"
        "0,1,1,BRISC,1,1000,0,1,BRISC-KERNEL,ZONE_START\n"
        "0,1,1,BRISC,1,5000,0,1,BRISC-KERNEL,ZONE_END\n"
        "0,1,1,TRISC_0,2,2000,0,1,TRISC-KERNEL,ZONE_START\n"
        "0,1,1,TRISC_0,2,8000,0,1,TRISC-KERNEL,ZONE_END\n"
        "0,2,1,BRISC,1,1500,0,1,BRISC-KERNEL,ZONE_START\n"
        "0,2,1,BRISC,1,6000,0,1,BRISC-KERNEL,ZONE_END\n"
        "0,1,1,BRISC,3,100,0,2,FG_MISSING,ZONE_END\n"
        "0,1,1,TRISC_0,4,9000000,1500,1,FG_INPUT_WAIT,ZONE_TOTAL\n")
    report = parse_device_csv(path)
    assert report["runs"][0]["kernel_span_us"] == 7
    assert report["runs"][0]["cores"] == 2
    assert report["unmatched_ends"] == 1
    assert report["runs"][0]["accumulated_zones"][0]["total_us"] == 1.5
    assert report["hardware_counters"] is None


def test_paired_summary_rejects_missing_failed_and_drifting_cases():
    def measured(ms, status="passed"):
        return {"status": status, "timing": {"median_ms": ms, "mad_ms": .01}}
    win = comparison(measured(10), measured(5), measured(10.1))
    assert win["eligible"] and win["speedup"] > 2
    failed = comparison(measured(10), measured(5, "numerical_or_integrity_failure"), measured(10))
    assert not failed["eligible"]
    rows = [{"candidate": "fast", "case": "a", "comparison": win} for _ in range(3)]
    report = summarize_comparisons(rows, ["a", "missing"], 3)["fast"]
    assert not report["all_requested_cases_pass"] and report["geomean_speedup"] is None
    assert report["cases"]["a"]["consistent_screening_win"]
    drift = comparison(measured(10), measured(9), measured(14))
    report = summarize_comparisons([{"candidate": "unstable", "case": "a", "comparison": drift}] * 3, ["a"], 3)
    assert not report["unstable"]["cases"]["a"]["consistent_screening_win"]


def test_workspace_trace_counts_overlapping_allocations_and_reused_addresses():
    def event(kind, address, per_bank=10):
        return {"node_type": kind, "params": {"type": "DRAM", "device_id": "0",
            "address": str(address), "max_size_per_bank": str(per_bank), "size": str(per_bank * 2)}}
    trace = [event("buffer_deallocate", 999), event("buffer_allocate", 10),
        event("buffer_allocate", 20, 30), event("buffer_deallocate", 10),
        event("buffer_allocate", 10, 15)]
    report = workspace_bound(trace, dram_banks=2, l1_capacity_bytes=50, limit_bytes=140)
    assert report["new_dram_peak_bytes"] == 90 and report["within_limit"]
    assert report["workspace_upper_bound_bytes"] == 140
    trace.append(event("buffer_allocate", 10))
    assert not workspace_bound(trace, dram_banks=2, l1_capacity_bytes=50, limit_bytes=140)["trace_valid"]


def test_workspace_excludes_only_final_output_lifetime_at_reused_address():
    def event(kind, address, per_bank, device=0):
        return {"node_type": kind, "params": {"type": "DRAM", "device_id": device,
            "address": address, "max_size_per_bank": per_bank, "size": per_bank * 2}}
    trace = [event("buffer_allocate", 10, 30), event("buffer_allocate", 20, 8),
             event("buffer_deallocate", 10, 30), event("buffer_allocate", 10, 50)]
    report = workspace_bound(trace, dram_banks=2, l1_capacity_bytes=50, limit_bytes=130,
                             output_address=10, expected_output_bytes=96)
    assert report["new_dram_peak_bytes"] == 116
    assert report["temporary_dram_peak_bytes"] == 76
    assert report["workspace_upper_bound_bytes"] == 126 and report["within_limit"]
    assert report["excluded_final_output"]["event_index"] == 3
    oversized = trace[:-1] + [event("buffer_allocate", 10, 500)]
    excess = workspace_bound(oversized, dram_banks=2, l1_capacity_bytes=50, limit_bytes=130,
                             output_address=10, expected_output_bytes=96)
    assert excess["temporary_dram_peak_bytes"] == 920 and not excess["within_limit"]
    for address, size in [(999, 96), (10, 101)]:
        invalid = workspace_bound(trace, dram_banks=2, l1_capacity_bytes=50, limit_bytes=1000,
                                  output_address=address, expected_output_bytes=size)
        assert not invalid["trace_valid"] and not invalid["within_limit"]
    trace.append(event("buffer_allocate", 10, 50, device=1))
    ambiguous = workspace_bound(trace, dram_banks=2, l1_capacity_bytes=50, limit_bytes=1000,
                                output_address=10, expected_output_bytes=96)
    assert not ambiguous["trace_valid"] and not ambiguous["within_limit"]
