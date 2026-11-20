"""Reconstruct six DUET 300-step experiments from local MLflow SQLite files.

Run from the DUET repository root:
    py -3.11 scripts/duet_six_runs_offline.py --inputs analysis_inputs --out analysis_outputs/six_runs

Uses SQLite/CSV from Python's standard library. matplotlib is optional for PNGs.
Does not contact Modal or run training. Never modifies source databases.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
from collections import defaultdict
from pathlib import Path


# (workspace, filename, first step to KEEP, last step to KEEP)
# Source precedence follows the CHECKPOINT actually used for continuation;
# earlier runs sometimes logged steps beyond their last saved checkpoint.
MANIFEST = {
    ("noop", 90): [
        ("duet", "20261011_231637_noops_200_steps.db", 1, 160),
        ("duet", "20261104_203623_noops_300_steps_from_160_steps.db", 161, 300),
    ],
    ("noop", 7): [
        ("duet1p0", "20261105_201744_noops_300_step_run_seed_7.db", 1, 200),
        ("duet1p0", "20261108_175204_noops_300_step_run_extended_from_seed_7_200_checkpoint.db", 201, 300),
    ],
        ("refill", 90): [
        ("duet", "20261010_204140_difficulty_only_200_steps.db", 1, 180),
        ("duet", "20261104_111801_difficulty_only_300_steps_extended_from_200_steps.db", 181, 220),
        ("duet", "20261110_104325_difficulty_only_refill_300_steps_extended_from_220_step_checkpoint_seed_90_w_top_up.db", 221, 300),
    ],
    ("refill", 7): [
        ("duet1p0", "20261110_103252_difficulty_only_refill_run_300_step_seed_7.db", 1, 160),
        ("duet1p0", "20261111_132534_difficulty_only_refill_run_300_step_seed_7_resuming_from_checkpoint_160.db", 161, 300),
    ],
    ("shrink", 90): [
        ("duet1p0", "20261104_114414_difficulty_only_shrink_300_steps_seed_90.db", 1, 300),
    ],
    ("shrink", 7): [
        ("duet1p0", "20261112_123508_difficulty_only_shrink_300_steps_seed_7.db", 1, 300),
    ],
}

KEYS = {
    "accuracy": ("val-core/openai/gsm8k/acc/mean_at_1", "val-core/openai/gsm8k/acc/mean@1"),
    "response_tokens": ("duet/generated_response_tokens",),
    "generation_seconds": ("timing_s/gen",),
    "batch_size": ("duet/prompts_selected", "duet/train_batch_size"),
    "prompts_topped_up": ("duet/prompts_topped_up",),
}
THRESHOLDS = (0.80, 0.825, 0.85, 0.86)


def fetch_metrics(db_path: Path):
    """Return {(step, canonical metric): value}; reject conflicting duplicates."""
    db = sqlite3.connect(f"file:{db_path.resolve().as_posix()}?mode=ro", uri=True)
    try:
        names = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "metrics" not in names:
            raise ValueError(f"Not an MLflow metrics database: {db_path}")
        aliases = {alias: name for name, variants in KEYS.items() for alias in variants}
        placeholders = ",".join("?" for _ in aliases)
        rows = db.execute(
            f"SELECT key, step, value, timestamp FROM metrics WHERE key IN ({placeholders}) ORDER BY timestamp, rowid",
            tuple(aliases),
        )
        found = {}
        for key, step, value, timestamp in rows:
            canonical = aliases[key]
            k = (int(step), canonical)
            # MLflow may have multiple run UUIDs or duplicate timestamps;
            # retain last logged value but make conflicting duplicates visible.
            if k in found and not math.isclose(found[k][0], value, rel_tol=1e-10, abs_tol=1e-10):
                raise ValueError(f"Conflicting entries for {canonical} at step {step} in {db_path}; inspect run UUIDs")
            found[k] = (float(value), int(timestamp), key)
        return {k: v[0] for k, v in found.items()}
    finally:
        db.close()


def reconstruct(inputs: Path):
    results = []
    provenance = []
    problems = []
    for (policy, seed), segments in MANIFEST.items():
        steps = {}
        for workspace, filename, start, end in segments:
            path = inputs / workspace / filename
            if not path.is_file():
                raise FileNotFoundError(f"Missing source: {path}")
            source_metrics = fetch_metrics(path)
            for (step, key), value in source_metrics.items():
                if not start <= step <= end:
                    continue
                row = steps.setdefault(step, {})
                if key in row:
                    raise ValueError(f"Overlapping source metrics: {policy} seed {seed} step {step} {key}")
                row[key] = value
                row["source_db"] = str(Path(workspace) / filename)
            provenance.append({"scheduler": policy, "seed": seed, "source_db": str(Path(workspace) / filename),
                               "keep_first_step": start, "keep_last_step": end})

        running_tokens = 0
        token_history_complete = True
        for step in range(1, 301):
            row = steps.get(step, {})
            tokens = row.get("response_tokens")
            if tokens is None or not math.isfinite(tokens) or tokens < 0:
                problems.append(f"{policy} seed {seed}: missing/invalid response tokens at step {step}")
                token_history_complete = False
                cumulative = None
            else:
                running_tokens += int(round(tokens))
                # Never publish a misleading cumulative value after a gap.
                cumulative = running_tokens if token_history_complete else None
            results.append({"scheduler": policy, "seed": seed, "step": step,
                            "accuracy": row.get("accuracy"),
                            "cumulative_response_tokens": cumulative,
                            "generated_response_tokens": tokens,
                            "generation_seconds": row.get("generation_seconds"),
                            "batch_size": row.get("batch_size") if policy == "shrink" else None,
                            "prompts_topped_up": row.get("prompts_topped_up") if policy == "refill" else None,
                            "source_db": row.get("source_db", "")})
        for field in ("generation_seconds", "batch_size" if policy == "shrink" else "generation_seconds",
                      "prompts_topped_up" if policy == "refill" else "generation_seconds"):
            missing = [r["step"] for r in results[-300:] if r[field] is None]
            if missing:
                problems.append(f"{policy} seed {seed}: {field} missing at {len(missing)} steps (first: {missing[:10]})")
        if not any(r["accuracy"] is not None for r in results[-300:]):
            problems.append(f"{policy} seed {seed}: NO validation accuracy metrics")
    return results, provenance, problems


def write_csv(path, rows, columns):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_results(rows, folder):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return "matplotlib not installed; CSV files still generated"
    for seed in (7, 90):
        subset = [r for r in rows if r["seed"] == seed]
        for field, xlabel, suffix in (
            ("step", "Training step", "accuracy_vs_step"),
            ("cumulative_response_tokens", "Cumulative generated response tokens", "accuracy_vs_tokens"),
        ):
            fig, ax = plt.subplots(figsize=(8, 5))
            for policy in ("noop", "refill", "shrink"):
                selected = [r for r in subset if r["scheduler"] == policy and r["accuracy"] is not None and r[field] is not None]
                ax.plot([r[field] for r in selected], [100*r["accuracy"] for r in selected], marker="o", label=policy)
            ax.set(xlabel=xlabel, ylabel="GSM8K validation accuracy (%)", title=f"Seed {seed}")
            ax.grid(alpha=0.25)
            ax.legend()
            fig.tight_layout()
            fig.savefig(folder / f"seed{seed}_{suffix}.png", dpi=180)
            plt.close(fig)
        for field, label, policies in (
            ("generation_seconds", "Generation seconds per step", ("noop", "refill", "shrink")),
            ("batch_size", "Shrink batch size", ("shrink",)),
            ("prompts_topped_up", "Refill prompts topped up", ("refill",)),
        ):
            fig, ax = plt.subplots(figsize=(8, 4))
            for policy in policies:
                selected = [r for r in subset if r["scheduler"] == policy and r[field] is not None]
                ax.plot([r["step"] for r in selected], [r[field] for r in selected], label=policy)
            ax.set(xlabel="Training step", ylabel=label, title=f"Seed {seed}")
            ax.grid(alpha=0.25)
            ax.legend()
            fig.tight_layout()
            fig.savefig(folder / f"seed{seed}_{field}.png", dpi=180)
            plt.close(fig)
    return "PNG figures generated"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=Path("analysis_inputs"))
    parser.add_argument("--out", type=Path, default=Path("analysis_outputs/six_runs"))
    parser.add_argument("--allow-incomplete", action="store_true", help="Generate partial files even with missing token steps")
    args = parser.parse_args()
    rows, provenance, problems = reconstruct(args.inputs)
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "per_step.csv", rows, list(rows[0]))
    write_csv(args.out / "source_manifest.csv", provenance, list(provenance[0]))
    thresholds = []
    for seed in (7, 90):
        for policy in ("noop", "refill", "shrink"):
            subset = [r for r in rows if r["seed"] == seed and r["scheduler"] == policy]
            for target in THRESHOLDS:
                crossing = next((r for r in subset if r["accuracy"] is not None and r["accuracy"] >= target), None)
                thresholds.append({"seed": seed, "scheduler": policy, "threshold_pct": target * 100,
                                   "first_eval_step": crossing["step"] if crossing else None,
                                   "cumulative_response_tokens": crossing["cumulative_response_tokens"] if crossing else None,
                                   "accuracy_at_crossing_pct": 100*crossing["accuracy"] if crossing else None})
    write_csv(args.out / "thresholds.csv", thresholds, list(thresholds[0]))
    summary = {"errors_or_warnings": problems, "source_count": len(provenance), "per_step_records": len(rows),
               "plot_status": plot_results(rows, args.out)}
    (args.out / "audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if problems and not args.allow_incomplete:
        raise SystemExit("Coverage warnings found; inspect audit.json before citing thresholds or figures")


if __name__ == "__main__":
    main()
