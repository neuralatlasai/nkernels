#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from benchmark_specs import BenchmarkCase, default_registry


@dataclass(frozen=True)
class ManifestMethod:
    domain: str
    name: str
    canonical_name: str


SOURCE_BINDING_FILES = {
    "cuda": "csrc/torch_bindings.cpp",
    "cpu": "csrc/cpu/torch_bindings.cpp",
    "rocm": "csrc/rocm/torch_bindings.cpp",
    "moe": "csrc/moe/torch_bindings.cpp",
    "stable": "csrc/libtorch_stable/torch_bindings.cpp",
}

METHOD_DEF_PATTERN = re.compile(
    r'(?:ops|cache_ops|cuda_utils|custom_ar|rocm_ops|m)\.def\(\s*"([^"]+)"',
    re.DOTALL,
)


def parse_manifest(manifest_path: Path) -> List[ManifestMethod]:
    methods: List[ManifestMethod] = []
    for raw_line in manifest_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line.startswith("NKERNELS_API_METHOD("):
            continue
        parts = line.split('"')
        if len(parts) < 4:
            continue
        domain = parts[1]
        name = parts[3]
        canonical = f"nkernels.{domain}.{name}"
        methods.append(ManifestMethod(domain=domain, name=name, canonical_name=canonical))
    return methods


def parse_methods_from_source_root(source_root: Path) -> List[ManifestMethod]:
    methods: List[ManifestMethod] = []
    seen = set()
    for domain, relative_path in SOURCE_BINDING_FILES.items():
        binding_path = source_root / relative_path
        if not binding_path.exists():
            continue
        text = binding_path.read_text(encoding="utf-8")
        for match in METHOD_DEF_PATTERN.finditer(text):
            name = match.group(1)
            name = re.sub(r"\(.+$", "", name, flags=re.DOTALL)
            name = name.rstrip("(").strip()
            if not name:
                continue
            key = (domain, name)
            if key in seen:
                continue
            seen.add(key)
            methods.append(
                ManifestMethod(
                    domain=domain,
                    name=name,
                    canonical_name=f"nkernels.{domain}.{name}",
                )
            )
    methods.sort(key=lambda item: (item.domain, item.name))
    return methods


def parse_namespace_map(value: str) -> Dict[str, str]:
    path = Path(value)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return json.loads(value)


def parse_library_spec(value: str) -> Dict[str, str]:
    path = Path(value)
    if path.exists():
        return {"kind": "library", "value": str(path.resolve())}
    if value.endswith(".so") or "/" in value:
        raise FileNotFoundError(
            f"Endpoint path does not exist: {value}. Build the extension first "
            "and pass the actual .so path."
        )
    return {"kind": "module", "value": value}


def op_exists(namespace: str, op_name: str) -> bool:
    if not hasattr(torch.ops, namespace):
        return False
    ns = getattr(torch.ops, namespace)
    return hasattr(ns, op_name)


def resolve_op(namespace: str, op_name: str) -> Any:
    return getattr(getattr(torch.ops, namespace), op_name)


def synchronize_if_needed(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def benchmark_callable(
    fn: Any,
    device: str,
    warmup: int,
    iters: int,
) -> Dict[str, float]:
    synchronize_if_needed(device)
    for _ in range(warmup):
        fn()
    synchronize_if_needed(device)

    durations_ms: List[float] = []
    if device.startswith("cuda"):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for _ in range(iters):
            start.record()
            fn()
            end.record()
            end.synchronize()
            durations_ms.append(float(start.elapsed_time(end)))
    else:
        for _ in range(iters):
            start_time = time.perf_counter()
            fn()
            durations_ms.append((time.perf_counter() - start_time) * 1000.0)

    median_ms = statistics.median(durations_ms)
    mean_ms = statistics.fmean(durations_ms)
    stdev_ms = statistics.pstdev(durations_ms) if len(durations_ms) > 1 else 0.0
    min_ms = min(durations_ms)
    max_ms = max(durations_ms)
    return {
        "median_ms": median_ms,
        "mean_ms": mean_ms,
        "stdev_ms": stdev_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
    }


def load_endpoint(spec: Dict[str, str]) -> None:
    if spec["kind"] == "library":
        torch.ops.load_library(spec["value"])
        return
    __import__(spec["value"])


def worker_run(config: Dict[str, Any]) -> Dict[str, Any]:
    load_endpoint(config["endpoint"])
    device = config["device"]
    if device.startswith("cuda"):
        torch.cuda.init()

    registry = default_registry()
    namespace_map: Dict[str, str] = config["namespace_map"]
    results: List[Dict[str, Any]] = []

    for item in config["methods"]:
        domain = item["domain"]
        method_name = item["name"]
        canonical_name = item["canonical_name"]
        namespace = namespace_map.get(domain)

        if not namespace:
          results.append(
              {
                  "canonical_name": canonical_name,
                  "domain": domain,
                  "method_name": method_name,
                  "status": "missing_namespace_mapping",
              }
          )
          continue

        if not op_exists(namespace, method_name):
          results.append(
              {
                  "canonical_name": canonical_name,
                  "domain": domain,
                  "method_name": method_name,
                  "status": "missing_op",
                  "namespace": namespace,
              }
          )
          continue

        cases = registry.get(method_name)
        if not cases:
          results.append(
              {
                  "canonical_name": canonical_name,
                  "domain": domain,
                  "method_name": method_name,
                  "status": "no_spec",
                  "namespace": namespace,
              }
          )
          continue

        op = resolve_op(namespace, method_name)
        for case in cases:
            try:
                fn = case.build(op, device)
                metrics = benchmark_callable(
                    fn=fn,
                    device=device,
                    warmup=int(config["warmup"]),
                    iters=int(config["iters"]),
                )
                payload = {
                    "canonical_name": canonical_name,
                    "domain": domain,
                    "method_name": method_name,
                    "status": "benchmarked",
                    "namespace": namespace,
                    "case_label": case.label,
                }
                payload.update(metrics)
                results.append(payload)
            except Exception as exc:  # noqa: BLE001
                results.append(
                    {
                        "canonical_name": canonical_name,
                        "domain": domain,
                        "method_name": method_name,
                        "status": "error",
                        "namespace": namespace,
                        "case_label": case.label,
                        "error": repr(exc),
                    }
                )

    return {
        "endpoint_name": config["endpoint_name"],
        "device": device,
        "results": results,
    }


def run_worker_subprocess(config: Dict[str, Any]) -> Dict[str, Any]:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as handle:
        json.dump(config, handle)
        config_path = handle.name

    try:
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", config_path]
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip()
        stdout = exc.stdout.strip()
        message = (
            f"Worker subprocess failed for endpoint "
            f"{config['endpoint_name']!r}.\n"
        )
        if stdout:
            message += f"\nWorker stdout:\n{stdout}\n"
        if stderr:
            message += f"\nWorker stderr:\n{stderr}\n"
        raise RuntimeError(message) from exc
    finally:
        os.unlink(config_path)

    return json.loads(completed.stdout)


def summarize_status(rows: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    return dict(sorted(counts.items()))


def join_results(
    baseline_rows: List[Dict[str, Any]],
    candidate_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    baseline_index = {
        (row["canonical_name"], row.get("case_label", "")): row for row in baseline_rows
    }
    candidate_index = {
        (row["canonical_name"], row.get("case_label", "")): row for row in candidate_rows
    }
    keys = sorted(set(baseline_index) | set(candidate_index))

    joined: List[Dict[str, Any]] = []
    for key in keys:
        baseline = baseline_index.get(key)
        candidate = candidate_index.get(key)
        merged: Dict[str, Any] = {
            "canonical_name": key[0],
            "case_label": key[1],
            "baseline_status": baseline["status"] if baseline else "absent",
            "candidate_status": candidate["status"] if candidate else "absent",
        }

        if baseline:
            merged["baseline_median_ms"] = baseline.get("median_ms")
        if candidate:
            merged["candidate_median_ms"] = candidate.get("median_ms")

        if (
            baseline
            and candidate
            and baseline["status"] == "benchmarked"
            and candidate["status"] == "benchmarked"
        ):
            base_ms = float(baseline["median_ms"])
            cand_ms = float(candidate["median_ms"])
            merged["speedup_vs_baseline"] = (
                base_ms / cand_ms if cand_ms > 0.0 else math.inf
            )
            merged["delta_ms"] = cand_ms - base_ms

        if baseline and baseline["status"] == "error":
            merged["baseline_error"] = baseline.get("error")
        if candidate and candidate["status"] == "error":
            merged["candidate_error"] = candidate.get("error")

        joined.append(merged)
    return joined


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_summary(
    path: Path,
    manifest_methods: List[ManifestMethod],
    baseline_name: str,
    candidate_name: str,
    baseline_rows: List[Dict[str, Any]],
    candidate_rows: List[Dict[str, Any]],
    joined_rows: List[Dict[str, Any]],
    baseline_source_root: Optional[Path],
    candidate_source_root: Optional[Path],
) -> None:
    baseline_counts = summarize_status(baseline_rows)
    candidate_counts = summarize_status(candidate_rows)
    improved = [
        row
        for row in joined_rows
        if "speedup_vs_baseline" in row and row["speedup_vs_baseline"] > 1.0
    ]
    regressions = [
        row
        for row in joined_rows
        if "speedup_vs_baseline" in row and row["speedup_vs_baseline"] < 1.0
    ]
    improved.sort(key=lambda row: row["speedup_vs_baseline"], reverse=True)
    regressions.sort(key=lambda row: row["speedup_vs_baseline"])

    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Kernel Comparison Summary\n\n")
        handle.write(f"- Manifest methods tracked: `{len(manifest_methods)}`\n")
        handle.write(f"- Baseline endpoint: `{baseline_name}`\n")
        handle.write(f"- Candidate endpoint: `{candidate_name}`\n")
        if baseline_source_root:
            handle.write(f"- Baseline source root: `{baseline_source_root}`\n")
        if candidate_source_root:
            handle.write(f"- Candidate source root: `{candidate_source_root}`\n")
        handle.write(f"- Baseline status counts: `{json.dumps(baseline_counts, sort_keys=True)}`\n")
        handle.write(f"- Candidate status counts: `{json.dumps(candidate_counts, sort_keys=True)}`\n")
        handle.write(f"- Improved benchmark cases: `{len(improved)}`\n")
        handle.write(f"- Regressed benchmark cases: `{len(regressions)}`\n\n")

        handle.write("## Top Improvements\n\n")
        for row in improved[:20]:
            handle.write(
                f"- `{row['canonical_name']}` `{row['case_label']}` "
                f"speedup `{row['speedup_vs_baseline']:.3f}x`\n"
            )

        handle.write("\n## Top Regressions\n\n")
        for row in regressions[:20]:
            handle.write(
                f"- `{row['canonical_name']}` `{row['case_label']}` "
                f"speedup `{row['speedup_vs_baseline']:.3f}x`\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare legacy and nkernels custom op performance."
    )
    parser.add_argument("--worker", help="Internal worker config JSON path.")
    parser.add_argument(
        "--manifest",
        default=str(
            (SCRIPT_DIR.parent / "csrc" / "extensions" / "generated_legacy_api_manifest.inc")
        ),
        help="Path to the canonical legacy API manifest.",
    )
    parser.add_argument(
        "--baseline-source-root",
        help="Optional cloned baseline repo root used to derive the source API surface.",
    )
    parser.add_argument(
        "--candidate-source-root",
        help="Optional cloned candidate repo root used for source inventory reporting.",
    )
    parser.add_argument("--baseline", help="Baseline module name or shared library path.")
    parser.add_argument("--candidate", help="Candidate module name or shared library path.")
    parser.add_argument(
        "--baseline-namespace-map",
        help="JSON string or file mapping manifest domains to torch.ops namespaces.",
    )
    parser.add_argument(
        "--candidate-namespace-map",
        help="JSON string or file mapping manifest domains to torch.ops namespaces.",
    )
    parser.add_argument("--device", default="cuda", help="Benchmark device.")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations.")
    parser.add_argument("--iters", type=int, default=100, help="Measured iterations.")
    parser.add_argument(
        "--method-filter",
        action="append",
        default=[],
        help="Optional substring filter for canonical method names.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(SCRIPT_DIR / "results"),
        help="Directory for JSON, CSV, and Markdown reports.",
    )
    args = parser.parse_args()

    if args.worker:
        config = json.loads(Path(args.worker).read_text(encoding="utf-8"))
        print(json.dumps(worker_run(config), indent=2, sort_keys=True))
        return

    if not args.baseline or not args.candidate:
        parser.error("--baseline and --candidate are required.")
    if not args.baseline_namespace_map or not args.candidate_namespace_map:
        parser.error(
            "--baseline-namespace-map and --candidate-namespace-map are required."
        )

    baseline_source_root = (
        Path(args.baseline_source_root).resolve()
        if args.baseline_source_root
        else None
    )
    candidate_source_root = (
        Path(args.candidate_source_root).resolve()
        if args.candidate_source_root
        else None
    )

    if baseline_source_root is not None:
        manifest_methods = parse_methods_from_source_root(baseline_source_root)
    else:
        manifest_path = Path(args.manifest).resolve()
        manifest_methods = parse_manifest(manifest_path)

    if args.method_filter:
        filters = tuple(args.method_filter)
        manifest_methods = [
            item
            for item in manifest_methods
            if any(token in item.canonical_name for token in filters)
        ]

    baseline_config = {
        "endpoint_name": args.baseline,
        "endpoint": parse_library_spec(args.baseline),
        "namespace_map": parse_namespace_map(args.baseline_namespace_map),
        "device": args.device,
        "warmup": args.warmup,
        "iters": args.iters,
        "methods": [asdict(item) for item in manifest_methods],
    }
    candidate_config = {
        "endpoint_name": args.candidate,
        "endpoint": parse_library_spec(args.candidate),
        "namespace_map": parse_namespace_map(args.candidate_namespace_map),
        "device": args.device,
        "warmup": args.warmup,
        "iters": args.iters,
        "methods": [asdict(item) for item in manifest_methods],
    }

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_payload = run_worker_subprocess(baseline_config)
    candidate_payload = run_worker_subprocess(candidate_config)

    baseline_rows = baseline_payload["results"]
    candidate_rows = candidate_payload["results"]
    joined_rows = join_results(baseline_rows, candidate_rows)

    source_inventory = {
        "manifest_method_count": len(manifest_methods),
        "baseline_source_root": str(baseline_source_root) if baseline_source_root else None,
        "candidate_source_root": str(candidate_source_root) if candidate_source_root else None,
    }
    if baseline_source_root is not None:
        source_inventory["baseline_source_method_count"] = len(
            parse_methods_from_source_root(baseline_source_root)
        )
    if candidate_source_root is not None:
        source_inventory["candidate_source_method_count"] = len(
            parse_methods_from_source_root(candidate_source_root)
        )

    (output_dir / "baseline_results.json").write_text(
        json.dumps(baseline_payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "candidate_results.json").write_text(
        json.dumps(candidate_payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "joined_results.json").write_text(
        json.dumps(joined_rows, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "source_inventory.json").write_text(
        json.dumps(source_inventory, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_csv(joined_rows, output_dir / "joined_results.csv")
    write_markdown_summary(
        output_dir / "summary.md",
        manifest_methods=manifest_methods,
        baseline_name=args.baseline,
        candidate_name=args.candidate,
        baseline_rows=baseline_rows,
        candidate_rows=candidate_rows,
        joined_rows=joined_rows,
        baseline_source_root=baseline_source_root,
        candidate_source_root=candidate_source_root,
    )

    print(f"Results written to {output_dir}")


if __name__ == "__main__":
    main()
