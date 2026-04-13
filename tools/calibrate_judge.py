import os
import json
from pathlib import Path

def load_latest_benchmark(state_dir: str) -> dict:
    p = Path(state_dir)
    files = sorted(p.glob("benchmark_*.json"))
    if not files:
        return {}
    return json.loads(files[-1].read_text(encoding="utf-8"))

def bucketize(samples, metric_key="metric"):
    buckets = [(0.0,0.2),(0.2,0.4),(0.4,0.6),(0.6,0.8),(0.8,1.0)]
    out = {}
    for lo, hi in buckets:
        mets = []
        for s in samples:
            c = float(s.get("judge_confidence", 0.0))
            if lo <= c < hi:
                m = s.get(metric_key)
                if m in ["exact_match","exact_match_f1"]:
                    mets.append(1.0 if float(s.get("em",0.0))>=1.0 else 0.0)
                else:
                    mets.append(1.0 if float(s.get("f1",0.0))>=0.8 else 0.0)
        key = f"{lo:.2f}-{hi:.2f}"
        out[key] = (sum(mets)/len(mets)) if mets else None
    return out

def main():
    state_dir = os.path.join(os.path.dirname(__file__), "..", "smca_state")
    data = load_latest_benchmark(state_dir)
    if not data:
        print("No benchmark file found")
        return
    all_samples = []
    for bl in data.get("baselines", {}).values():
        all_samples.extend(bl.get("samples", []))
    curve = bucketize(all_samples)
    print(json.dumps(curve, indent=2))

if __name__ == "__main__":
    main()
