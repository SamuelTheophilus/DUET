"""Offline audit of binary-reward GRPO groups in numbered rollout JSONL files.

Usage:
    python dead_group_analysis.py --inputs PATH_TO_JSONL_FOLDER --out OUTPUT_FOLDER
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def pearson(xs: list[float], ys: list[float]) -> float:
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    numerator = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denominator = math.sqrt(
        sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)
    )
    return numerator / denominator if denominator else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    files = sorted(
        (p for p in args.inputs.glob("*.jsonl") if p.stem.isdigit()),
        key=lambda p: int(p.stem),
    )
    if not files:
        parser.error("No numbered JSONL files found")
    args.out.mkdir(parents=True, exist_ok=True)

    per_batch = []
    excluded = 0
    all_group_sizes = set()
    for path in files:
        groups = defaultdict(list)
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                item = json.loads(line)
                uid = str(item.get("uid") or "")
                prompt = str(item.get("input") or "")
                if not prompt.strip() or uid.lower().startswith("pad"):
                    excluded += 1
                    continue
                prompt_id = re.sub(r"_\d+_\d+$", "", uid)
                if prompt_id == uid:
                    raise ValueError(f"Unexpected UID format: {path}:{line_number}: {uid}")
                score = float(item["score"])
                if score not in (0.0, 1.0):
                    raise ValueError(f"Nonbinary score at {path}:{line_number}: {score}")
                groups[prompt_id].append(score)

        sizes = {len(scores) for scores in groups.values()}
        all_group_sizes.update(sizes)
        if sizes != {5}:
            raise ValueError(f"Expected five rollouts/group in {path.name}, got {sizes}")
        easy = sum(all(score == 1 for score in scores) for scores in groups.values())
        hard = sum(all(score == 0 for score in scores) for scores in groups.values())
        n = len(groups)
        if not n:
            raise ValueError(f"No groups in {path}")
        per_batch.append(
            {
                "batch": int(path.stem),
                "source_file": path.name,
                "groups": n,
                "rollouts": n * 5,
                "easy": easy,
                "hard": hard,
                "productive": n - easy - hard,
                "dead": easy + hard,
                "dead_pct": round(100 * (easy + hard) / n, 6),
            }
        )

    columns = ["batch", "source_file", "groups", "rollouts", "easy", "hard", "productive", "dead", "dead_pct"]
    write_csv(args.out / "dead_group_per_batch.csv", per_batch, columns)
    total_groups = sum(row["groups"] for row in per_batch)
    total_dead = sum(row["dead"] for row in per_batch)
    early = [r for r in per_batch if r["batch"] <= 9]
    late = [r for r in per_batch if r["batch"] >= 10]

    def period_stats(rows: list[dict], label: str) -> dict:
        groups = sum(r["groups"] for r in rows)
        dead = sum(r["dead"] for r in rows)
        return {"period": label, "batches": len(rows), "groups": groups, "dead": dead,
                "dead_pct": round(100 * dead / groups, 6) if groups else ""}

    summary_rows = [period_stats(per_batch, "all")]
    if early and late:
        summary_rows += [period_stats(early, "batches_1_to_9"), period_stats(late, "batches_10_to_16")]
    summary_rows += [period_stats([r for r in per_batch if r["batch"] <= 12], "batches_1_to_12")]
    write_csv(args.out / "dead_group_summary.csv", summary_rows,
              ["period", "batches", "groups", "dead", "dead_pct"])

    corr = pearson([float(r["batch"]) for r in per_batch],
                   [float(r["dead_pct"]) for r in per_batch]) if len(per_batch) > 1 else None
    audit = {
        "input_files": [str(p) for p in files],
        "groups": total_groups,
        "dead_groups": total_dead,
        "dead_pct": 100 * total_dead / total_groups,
        "excluded_padding_rows": excluded,
        "rollouts_per_group": sorted(all_group_sizes),
        "pearson_batch_index_vs_dead_pct": corr,
        "caution": "Batch number comes from filenames; training chronology and causality are not independently verified. Historical r=-0.213 and 36.8% are not reproduced by this script.",
    }
    (args.out / "dead_group_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib absent; CSV and JSON created, figures skipped.")
    else:
        batch = [r["batch"] for r in per_batch]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(batch, [r["dead_pct"] for r in per_batch], marker="o", linewidth=2)
        ax.set(xlabel="Rollout file index", ylabel="Dead groups (%)", title="Observed dead-group prevalence")
        ax.set_xticks(batch)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=.25)
        fig.tight_layout()
        fig.savefig(args.out / "dead_group_trend.png", dpi=200)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 5))
        bottom = [0] * len(per_batch)
        for name, color in [("easy", "#298c8c"), ("productive", "#4d73ad"), ("hard", "#d78254")]:
            vals = [100 * r[name] / r["groups"] for r in per_batch]
            ax.bar(batch, vals, bottom=bottom, label=name.title(), color=color)
            bottom = [a + b for a, b in zip(bottom, vals)]
        ax.set(xlabel="Rollout file index", ylabel="Prompt groups (%)", title="GRPO reward-group composition")
        ax.set_xticks(batch)
        ax.set_ylim(0, 100)
        ax.legend(ncol=3)
        fig.tight_layout()
        fig.savefig(args.out / "dead_group_composition.png", dpi=200)
        plt.close(fig)

    print(json.dumps(audit, indent=2))
    print(f"Outputs: {args.out.resolve()}")


if __name__ == "__main__":
    main()
