"""Render publication-ready DUET tables from six-run analysis CSVs.

Usage from repository root:
  uv run --no-project --python 3.11 --with matplotlib scripts/duet_render_tables.py --dir analysis_outputs/six_runs

This is offline; it reads existing CSV files and never modifies MLflow databases.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def cell_number(value, digits=1):
    return "—" if value in (None, "") else f"{float(value):,.{digits}f}"


def draw_table(path, title, columns, data, widths=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_width = 12
    fig_height = max(2.4, 1.15 + len(data) * 0.42)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=15, weight="bold", pad=18)
    table = ax.table(cellText=data, colLabels=columns, loc="center", cellLoc="center",
                     colWidths=widths, bbox=[0, 0.01, 1, 0.91])
    table.auto_set_font_size(False)
    table.set_fontsize(9.7)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#d5dce5")
        cell.set_linewidth(0.5)
        if r == 0:
            cell.set_facecolor("#23364d")
            cell.set_text_props(color="white", weight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#eff4f8")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=Path("analysis_outputs/six_runs"))
    args = parser.parse_args()
    directory = args.dir
    steps = read_csv(directory / "per_step.csv")
    thresholds = read_csv(directory / "thresholds.csv")
    grouped = defaultdict(list)
    for row in steps:
        grouped[(row["scheduler"], int(row["seed"]))].append(row)

    total_tokens = {}
    summary = []
    for (policy, seed), rows in sorted(grouped.items(), key=lambda x: (x[0][1], x[0][0])):
        last_tokens = float(rows[-1]["cumulative_response_tokens"] or "nan")
        total_tokens[(policy, seed)] = last_tokens
        generation_sum = sum(float(r["generation_seconds"]) for r in rows if r["generation_seconds"])
        batches = [float(r["batch_size"]) for r in rows if r["batch_size"]]
        topups = [float(r["prompts_topped_up"]) for r in rows if r["prompts_topped_up"]]
        summary.append({
            "seed": seed, "scheduler": policy, "training_steps": len(rows),
            "evaluation_points": sum(bool(r["accuracy"]) for r in rows),
            "final_response_tokens": int(last_tokens),
            "total_generation_seconds": round(generation_sum, 2),
            "mean_generation_seconds": round(generation_sum / len(rows), 2),
            "min_batch_size": min(batches) if batches else "",
            "max_batch_size": max(batches) if batches else "",
            "topup_metric_steps": len(topups),
            "observed_topups": int(sum(topups)) if topups else "",
        })

    keys = list(summary[0])
    with (directory / "run_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(summary)

    for seed in (7, 90):
        selected = [r for r in thresholds if int(r["seed"]) == seed]
        table_rows = [[r["scheduler"].title(), cell_number(r["threshold_pct"]),
                       r["first_eval_step"] or "Not reached",
                       cell_number(float(r["cumulative_response_tokens"]) / 1e6, 2) if r["cumulative_response_tokens"] else "—",
                       cell_number(r["accuracy_at_crossing_pct"], 2)] for r in selected]
        draw_table(directory / f"seed{seed}_thresholds_table.png",
                   f"DUET accuracy-threshold crossings — seed {seed}",
                   ["Scheduler", "Target (%)", "First eval step", "Resp. tokens (M)", "Observed accuracy (%)"],
                   table_rows, [.20, .18, .20, .21, .21])

    rows = []
    for record in summary:
        policy, seed = record["scheduler"], record["seed"]
        reference = total_tokens.get(("noop", seed), 0)
        savings = (1 - record["final_response_tokens"] / reference) * 100 if reference else float("nan")
        rows.append([str(seed), policy.title(), str(record["training_steps"]),
                     cell_number(record["final_response_tokens"] / 1e6, 2),
                     cell_number(savings, 1),
                     cell_number(record["mean_generation_seconds"], 1)])
    draw_table(directory / "six_runs_summary_table.png", "DUET — six-run comparison at 300 steps",
               ["Seed", "Scheduler", "Steps", "Resp. tokens (M)", "Savings vs NoOp (%)", "Mean gen (s/step)"],
               rows, [.09, .18, .10, .20, .23, .20])

    special_rows = []
    for record in summary:
        if record["scheduler"] not in ("refill", "shrink"):
            continue
        special_rows.append([str(record["seed"]), record["scheduler"].title(),
                             f"{cell_number(record['min_batch_size'], 0)}–{cell_number(record['max_batch_size'], 0)}" if record["min_batch_size"] != "" else "N/A",
                             f"{record['topup_metric_steps']}/300" if record["scheduler"] == "refill" else "N/A",
                             f"{record['observed_topups']:,}" if record["observed_topups"] != "" else "Not recorded"])
    draw_table(directory / "scheduler_diagnostics_table.png", "DUET scheduler diagnostics (recorded data only)",
               ["Seed", "Scheduler", "Batch-size range", "Top-up reporting", "Observed top-ups"],
               special_rows, [.10, .18, .27, .22, .23])
    print("Generated run_summary.csv and four table PNGs in", directory)
    print("Missing top-up records remain unreported (never zero-filled).")


if __name__ == "__main__":
    main()
