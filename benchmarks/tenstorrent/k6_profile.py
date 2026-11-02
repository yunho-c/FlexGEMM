"""Diagnostic K6 wait scopes in generated copies; never a promotion score.

Invoke with the board lock held and TT_METAL_DEVICE_PROFILER=1,
TT_METAL_PROFILER_SUM=1, and a fresh TT_METAL_PROFILER_DIR. The original
kernel files remain untouched. Each measured CSV segment contains exactly one
selected product invocation, or one complete operator invocation.
"""

import argparse
import json
import os
from pathlib import Path
import time
import traceback

import torch

from .common import correctness, sha256, write_json
from .evaluate import source_state
from .profile import parse_device_csv


MODES = ("base", "compute", "weight", "gather", "multicast")


def instrument(source, destination, mode):
    """Insert timing scopes only; fail if the reviewed source anchors change."""
    assert mode in MODES
    destination.mkdir(parents=True, exist_ok=False)
    files = {p.name: p.read_text() for p in source.iterdir() if p.suffix in (".cpp", ".hpp")}

    def replace(name, old, new, count=1):
        assert files[name].count(old) == count, (name, old)
        files[name] = files[name].replace(old, new)

    for name, label in (("k6_compute.cpp", "COMPUTE"), ("k6_reader.cpp", "READER"),
                        ("k6_writer.cpp", "WRITER")):
        replace(name, '#include <cstdint>', '#include <cstdint>\n#include "tools/profiler/kernel_profiler.hpp"')
        replace(name, 'void kernel_main() {',
                'void kernel_main() {\n    DeviceZoneScopedN("FG_K6_' + label + '");')
    if mode in ("compute", "weight"):
        # SDK llk_wait_tiles owns UNPACK slot 1. Only slot 2 is free;
        # measure selected gather waits and weight waits in separate runs.
        old, label = (("cb_wait_front(cb, CIN_T);", "WAIT_GATHER") if mode == "compute" else
                      ("cb_wait_front(CB_W, WCH);", "WAIT_WEIGHT"))
        replace("k6_compute.cpp", old,
                '{ UNPACK(DeviceZoneScopedSumN2("FG_K6_' + label + '");) ' + old + ' }')
        replace("k6_compute.cpp", "tile_regs_acquire();",
                '{ MATH(DeviceZoneScopedSumN2("FG_K6_MATH_DEST_WAIT");) tile_regs_acquire(); }')
        replace("k6_compute.cpp", "                matmul_init(CB_TL, CB_W);",
                '                { MATH(DeviceZoneScopedSumN1("FG_K6_MATMUL_REGION");)\n'
                '                matmul_init(CB_TL, CB_W);')
        replace("k6_compute.cpp", "                if (!last) {", "                }\n                if (!last) {")
    elif mode == "gather":
        replace("k6_gather.hpp", '#include <cstdint>',
                '#include <cstdint>\n#include "tools/profiler/kernel_profiler.hpp"')
        # Both local zero-copy and feature/index completion use this one scope
        # location and one accumulator. Their sum is NOT a DRAM-only counter.
        replace("k6_gather.hpp", "// live (any index", '''inline void fg_k6_gather_barrier() {
    DeviceZoneScopedSumN2("FG_K6_GATHER_READ_BARRIERS");
    noc_async_read_barrier();
}

// live (any index''')
        # Exclude the helper itself from substitution.
        old = "    noc_async_read_barrier();"
        assert files["k6_gather.hpp"].count(old) == 3
        first, rest = files["k6_gather.hpp"].split(old, 1)
        files["k6_gather.hpp"] = first + old + rest.replace(old, "    fg_k6_gather_barrier();")
        replace("k6_gather.hpp", "    cb_reserve_back(cb, np * CIN_T);",
                '    { DeviceZoneScopedSumN1("FG_K6_GATHER_BUFFER_WAIT"); cb_reserve_back(cb, np * CIN_T); }')
    elif mode == "multicast":
        replace("k6_reader.cpp", "void kernel_main() {", '''inline void fg_k6_weight_ready(volatile tt_l1_ptr uint32_t* ptr, uint32_t value) {
    DeviceZoneScopedSumN2("FG_K6_WEIGHT_HANDSHAKE_WAIT");
    noc_semaphore_wait(ptr, value);
}

void kernel_main() {''')
        replace("k6_reader.cpp", "                noc_async_read_barrier();\n            }\n            if constexpr (MCAST)",
                '                { DeviceZoneScopedSumN1("FG_K6_WEIGHT_READ_BARRIER"); noc_async_read_barrier(); }\n'
                '            }\n            if constexpr (MCAST)')
        replace("k6_reader.cpp", "noc_semaphore_wait(ready_ptr, NRECV);", "fg_k6_weight_ready(ready_ptr, NRECV);")
        replace("k6_reader.cpp", "noc_semaphore_wait(valid_ptr, 1);", "fg_k6_weight_ready(valid_ptr, 1);")
    for name, content in files.items():
        (destination / name).write_text(content)
    return {name: sha256(destination / name) for name in files}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--case", required=True)
    p.add_argument("--row-chunk", type=int, required=True)
    p.add_argument("--mode", choices=MODES, required=True)
    p.add_argument("--scope", choices=("chunks", "operator"), default="chunks")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    assert os.environ.get("FLEX_GEMM_EVALUATOR_LOCKED") == "1"
    assert os.environ.get("TT_METAL_DEVICE_PROFILER") == "1"
    assert os.environ.get("TT_METAL_PROFILER_SUM") == "1"
    assert os.environ.get("TT_METAL_PROFILER_MID_RUN_DUMP") == "1"
    assert not args.output.exists()
    torch.set_num_threads(8)
    root = Path(__file__).resolve().parents[2]
    protected = {str(path): sha256(path) for path in Path(__file__).parent.glob("*.py")}
    manifest = json.loads((args.corpus / "manifest.json").read_text())
    contract = json.loads((args.corpus / "contract.json").read_text())
    assert sha256(args.corpus / "contract.json") == manifest["contract_sha256"]
    entry = next(x for x in manifest["cases"] if x["name"] == args.case)
    fixture_path = args.corpus / entry["file"]
    assert sha256(fixture_path) == entry["sha256"]
    fixture = torch.load(fixture_path, map_location="cpu", weights_only=True)
    report = dict(status="running", phase="prepare", mode=args.mode, scope=args.scope,
                  case=entry, row_chunk=args.row_chunk, candidate=source_state(root),
                  checks=[], profiles=[], selections=[], evaluator_sha256=protected,
                  scoring_eligible=False, hardware_counters=None)
    write_json(args.output, report)
    import ttnn
    import flex_gemm.tenstorrent as fg
    from flex_gemm.tenstorrent import _k6
    from flex_gemm.tenstorrent.digits import _feature_digits
    original = _k6._KDIR
    generated = args.output.with_suffix(".kernels")
    report["instrumented_source_sha256"] = instrument(Path(original), generated, args.mode)
    csv_path = Path(os.environ["TT_METAL_PROFILER_DIR"]) / ".logs/profile_log_device.csv"
    device = None
    try:
        device = ttnn.open_device(device_id=0)
        weights = fg.prepare_sparse_digit_weights(fixture["weight"], bias=fixture["bias"], device=device)
        features = [ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
                    for x in fixture["features"]]
        count = len(fixture["mapping"])
        starts = [0] if args.scope == "operator" else sorted({0, (count // 2 // args.row_chunk) * args.row_chunk,
                                                            ((count - 1) // args.row_chunk) * args.row_chunk})
        header = None

        def drain():
            nonlocal header
            ttnn.ReadDeviceProfiler(device)
            assert csv_path.exists()
            if header is None:
                with csv_path.open("rb") as stream:
                    header = stream.readline() + stream.readline()
            return csv_path.stat().st_size

        def capture(fn, label):
            before = drain()
            start = time.perf_counter()
            output = fn()
            ttnn.synchronize_device(device)
            elapsed = (time.perf_counter() - start) * 1000
            after = drain()
            segment = args.output.parent / (args.output.stem + "-" + label + ".csv")
            with csv_path.open("rb") as stream:
                stream.seek(before)
                data = stream.read(after - before)
            segment.write_bytes(header + data)
            parsed = parse_device_csv(segment)
            assert not parsed["unmatched_starts"] and not parsed["unmatched_ends"]
            report["profiles"].append(dict(label=label, wall_ms=elapsed, csv=str(segment),
                                             csv_sha256=sha256(segment), parsed=parsed))
            return output

        for start in starts:
            end = count if args.scope == "operator" else min(start + args.row_chunk, count)
            topology = fg.prepare_sparse_topology(fixture["mapping"][start:end], len(fixture["features"][0]), device=device)
            plan = fg.prepare_sparse_digit_implicit(weights, topology, row_chunk=args.row_chunk)
            assert plan.info["path"] == "digit_implicit_gemm"
            report["selections"].append(dict(start=start, end=end, policy=plan.info))
            for variant, feature in enumerate(features):
                def execute():
                    return fg.sparse_conv_device(feature, weights, topology,
                        algorithm="digit_implicit_gemm", row_chunk=args.row_chunk)
                _k6._KDIR = original
                native = ttnn.to_torch(execute())
                metrics = correctness(native.float(), fixture["expected"][variant][start:end], contract)
                metrics.update(kind="native_operator", start=start, end=end, variant=variant)
                report["checks"].append(metrics)
                _k6._KDIR = str(generated)
                actual = ttnn.to_torch(execute())
                metrics = correctness(actual.float(), fixture["expected"][variant][start:end], contract)
                metrics.update(kind="instrumented_operator", start=start, end=end, variant=variant,
                               bitwise_equal=bool(torch.equal(native, actual)))
                metrics["passed"] &= metrics["bitwise_equal"]
                report["checks"].append(metrics)
                write_json(args.output, report)
                if args.scope == "operator":
                    output = capture(execute, f"operator-v{variant}")
                    same = bool(torch.equal(ttnn.to_torch(output), native))
                    report["checks"].append(dict(kind="profiled_operator", passed=same, variant=variant))
                    del output
                else:
                    high, low, scale = _feature_digits(feature, ((entry["ci"] + 31) // 32) * 32)
                    index = plan.chunks[0][2]
                    def product():
                        return weights.implicit_layers[0](high, index, write_y=True,
                            write_table=False, dead_skip=False)[0]
                    _k6._KDIR = original
                    product_native = ttnn.to_torch(product())
                    _k6._KDIR = str(generated)
                    # Compile/cache warmup is excluded from each selected product sample.
                    warm = product()
                    ttnn.synchronize_device(device)
                    del warm
                    for repetition in range(3):
                        output = capture(product, f"r{start}-v{variant}-p{repetition}")
                        same = bool(torch.equal(ttnn.to_torch(output), product_native))
                        report["checks"].append(dict(kind="profiled_product", passed=same,
                            start=start, end=end, variant=variant, repetition=repetition))
                        del output
                    del high, low, scale, product_native
                del native, actual
                write_json(args.output, report)
            del topology, plan
        report["input_unchanged"] = all(torch.equal(ttnn.to_torch(d), h)
            for d, h in zip(features, fixture["features"]))
        report["source_unchanged"] = source_state(root)["sha256"] == report["candidate"]["sha256"]
        report["fixture_unchanged"] = sha256(fixture_path) == entry["sha256"]
        report["evaluator_unchanged"] = all(sha256(path) == digest for path, digest in protected.items())
        report["status"] = "passed" if all(c["passed"] for c in report["checks"]) and all(
            report[k] for k in ("input_unchanged", "source_unchanged", "fixture_unchanged", "evaluator_unchanged")) else "failed"
    except Exception as error:
        report.update(status="execution_failure", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
    finally:
        _k6._KDIR = original
        if device is not None:
            ttnn.close_device(device)
        report["phase"] = "complete"
        write_json(args.output, report)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
