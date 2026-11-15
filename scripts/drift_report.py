import argparse
import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = PROJECT_ROOT / "logs" / "predictions.jsonl"
FEATURES = ["estimated_days", "total_price", "total_freight"]
WARN, ALERT = 0.1, 0.25


def psi(reference, current, bins=10):
    """Population Stability Index between two samples."""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        v = edges[0]
        edges = np.array([-np.inf, v, np.nextafter(v, np.inf), np.inf])
    else:
        edges[0], edges[-1] = -np.inf, np.inf
    ref_pct = np.histogram(reference, edges)[0] / len(reference)
    cur_pct = np.histogram(current, edges)[0] / len(current)
    ref_pct = np.clip(ref_pct, 1e-4, None)
    cur_pct = np.clip(cur_pct, 1e-4, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def status(value):
    if value >= ALERT:
        return "ALERT"
    if value >= WARN:
        return "WARN"
    return "ok"


def column(records, name):
    values = (r["features"].get(name) for r in records)
    return [v for v in values if v is not None]


def main():
    parser = argparse.ArgumentParser(description="Compare recent predictions.")
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--log", type=Path, default=LOG_PATH)
    args = parser.parse_args()

    if not args.log.exists():
        raise SystemExit(f"No prediction log found: {args.log}")
    with args.log.open(encoding="utf-8") as file:
        records = [json.loads(line) for line in file if line.strip()]

    n = args.window
    if len(records) < 2 * n:
        raise SystemExit(f"Need {2 * n} logged predictions, found {len(records)}.")
    baseline, current = records[:n], records[-n:]

    rows = [
        (
            "late_probability",
            psi(
                [r["late_probability"] for r in baseline],
                [r["late_probability"] for r in current],
            ),
        )
    ]
    for name in FEATURES:
        ref, cur = column(baseline, name), column(current, name)
        if ref and cur:
            rows.append((name, psi(ref, cur)))

    rate_ref = np.mean([r["predicted_is_late"] for r in baseline])
    rate_cur = np.mean([r["predicted_is_late"] for r in current])
    print(f"Predicted-late rate: {rate_ref:.1%} (baseline) -> {rate_cur:.1%}")
    for label, value in rows:
        print(f"PSI {label:<18}{value:7.3f}  {status(value)}")
    raise SystemExit(1 if any(value >= ALERT for _, value in rows) else 0)


if __name__ == "__main__":
    main()
