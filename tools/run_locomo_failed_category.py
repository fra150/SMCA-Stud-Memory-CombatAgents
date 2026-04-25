import argparse
import json
import os
import re
import sys
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from locomo_benchmark_v2 import (
    LOCOMOBenchmarkV2,
    generate_entity_tests,
    generate_multi_hop_tests,
    generate_temporal_tests,
)


def _parse_test_id(test_id: str) -> Tuple[str, int, int, int]:
    m = re.match(r"^(?P<cat>[a-z_]+)_v(?P<v>\d+)_q(?P<q>\d+)_L(?P<L>\d+)$", test_id)
    if not m:
        raise ValueError(f"Unsupported test_id format: {test_id}")
    return m.group("cat"), int(m.group("v")), int(m.group("q")), int(m.group("L"))


def _get_testcase_by_id(category: str, test_id: str):
    cat, v, _, L = _parse_test_id(test_id)
    if cat != category:
        raise ValueError(f"Category mismatch for {test_id}: expected {category}, got {cat}")

    if category == "entity":
        tests = generate_entity_tests(n_variants=v + 1, lengths=[L])
    elif category == "multi_hop":
        tests = generate_multi_hop_tests(n_variants=v + 1, lengths=[L])
    elif category == "temporal":
        tests = generate_temporal_tests(n_variants=v + 1, lengths=[L])
    else:
        raise ValueError(f"Unsupported category: {category}")

    for t in tests:
        if t.test_id == test_id:
            return t
    raise RuntimeError(f"Failed to reconstruct test case: {test_id}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-file", required=True)
    parser.add_argument("--category", required=True, choices=["entity", "multi_hop", "temporal"])
    parser.add_argument("--system", default="bas")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    with open(args.results_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    results: List[Dict] = list(data.get("results") or [])
    failed = [
        r for r in results
        if r.get("system") == args.system
        and r.get("category") == args.category
        and not bool(r.get("is_correct"))
    ]

    test_ids = [r.get("test_id") for r in failed if r.get("test_id")]
    test_ids = sorted(set(test_ids))

    if not test_ids:
        print(f"No failed tests found for category={args.category} system={args.system}")
        return 0

    bench = LOCOMOBenchmarkV2(n_variants=1, lengths=[20], run_baselines=False)
    rerun_results = []
    correct = 0

    print("=" * 70)
    print(f"Re-running FAILED tests only")
    print(f"  Category: {args.category}")
    print(f"  System:   {args.system}")
    print(f"  Count:    {len(test_ids)}")
    print("=" * 70)

    for i, tid in enumerate(test_ids, start=1):
        print(f"[{i}/{len(test_ids)}] {tid}", end="", flush=True)
        test = _get_testcase_by_id(args.category, tid)
        res = bench.run_bas_test(test)
        rerun_results.append(res.to_dict())
        if res.is_correct:
            correct += 1
            print(" — OK")
        else:
            print(" — FAIL")

    acc = (correct / len(test_ids)) * 100.0
    print("=" * 70)
    print(f"Subset accuracy: {correct}/{len(test_ids)} ({acc:.1f}%)")

    if args.output:
        out = {
            "source_results_file": args.results_file,
            "category": args.category,
            "system": args.system,
            "total": len(test_ids),
            "correct": correct,
            "accuracy": acc,
            "rerun_results": rerun_results,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"Saved: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
