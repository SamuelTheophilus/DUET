"""Audit GRPO reward-group composition for checkpoint-stitched NoOp rollouts.

Example (from DUET repository root):
    python scripts/noop_512x8_dead_groups.py --inputs analysis_inputs/noop_512x8 --out analysis_outputs/noop_512x8_dead_groups

Original seed90 steps 1-160 are followed by resumed steps 161-300.
Seed7, if supplied, uses original steps 1-200 and resumed steps 201-300.
The input files are read only; no MLflow, GPU, or Modal access is required.
"""

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


SEGMENTS = {
    90: (("seed90_original", 1, 160), ("seed90_resumed", 161, 300)),
    7: (("seed7_original", 1, 200), ("seed7_resumed", 201, 300)),
}


def indexed_files(directory):
    result = {}
    for file in directory.rglob("*.jsonl"):
        if not file.stem.isdecimal():
            continue
        step = int(file.stem)
        if step in result:
            raise ValueError(f"Duplicate step {step} in {directory}: {result[step]} and {file}")
        result[step] = file
    return result


def summarize_step(seed, step, path):
    groups = defaultdict(list)
    prompt_inputs = {}
    text_groups = defaultdict(list)
    rejected = 0
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if int(row["step"]) != step:
                raise ValueError(f"Unexpected step in {path}:{line_number}: {row['step']}")
            uid = str(row.get("uid") or "")
            prompt = str(row.get("input") or "")
            if not prompt.strip() or uid.lower().startswith("pad"):
                rejected += 1
                continue
            match = re.fullmatch(r"(.+)_\d+_\d+", uid)
            if not match:
                raise ValueError(f"Unexpected UID in {path}:{line_number}: {uid}")
            score = float(row["score"])
            if score not in (0.0, 1.0):
                raise ValueError(f"Nonbinary score in {path}:{line_number}: {score}")
            groups[match.group(1)].append(score)
            prompt_inputs[match.group(1)] = prompt
            text_groups[prompt].append(score)

    sizes = Counter(map(len, groups.values()))
    if len(groups) != 512 or sizes != {8: 512}:
        raise ValueError(f"Incomplete/misaligned groups in {path}: {len(groups)} groups, sizes {sizes}")
    # Exact input text must identify one and only one group within the step
    # before it is safe to link appearances across different training steps.
    if len(text_groups) != 512 or Counter(map(len, text_groups.values())) != {8: 512}:
        raise ValueError(f"Cannot uniquely match prompt texts in {path}")
    correct_counts = Counter(sum(v) for v in groups.values())
    easy = correct_counts[8.0]
    hard = correct_counts[0.0]
    productive = 512 - easy - hard
    row = {
        "seed": seed, "step": step, "source_file": str(path),
        "groups": 512, "rollouts": 4096, "excluded_padding_rows": rejected,
        "easy_dead_8of8": easy, "hard_dead_0of8": hard,
        "productive_1to7": productive, "dead_total": easy + hard,
        "easy_dead_pct": 100 * easy / 512,
        "hard_dead_pct": 100 * hard / 512,
        "productive_pct": 100 * productive / 512,
        "dead_total_pct": 100 * (easy + hard) / 512,
    }
    prompt_status = {}
    for prompt, scores in text_groups.items():
        total = sum(scores)
        prompt_status[prompt] = "easy" if total == 8 else "hard" if total == 0 else "productive"
    return row, prompt_status


def write_csv(path, records):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def plot(records, output):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable: CSV/JSON generated, plots skipped")
        return
    for seed in sorted({r["seed"] for r in records}):
        run = [r for r in records if r["seed"] == seed]
        x = [r["step"] for r in run]
        fig, ax = plt.subplots(figsize=(10, 5))
        for field, label, color in (
            ("easy_dead_pct", "Easy dead (8/8 correct)", "#1a7d8a"),
            ("hard_dead_pct", "Hard dead (0/8 correct)", "#c15e36"),
            ("productive_pct", "Productive (1–7 correct)", "#5773b6"),
        ):
            ax.plot(x, [r[field] for r in run], label=label, linewidth=1.4, color=color)
        ax.set(xlabel="Training step", ylabel="Prompt groups (%)",
               title=f"NoOp seed {seed}: reward-group composition (512 × 8)")
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / f"seed{seed}_dead_group_composition.png", dpi=190)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(x, [r["dead_total_pct"] for r in run], color="#563c96", linewidth=1.5)
        ax.set(xlabel="Training step", ylabel="Dead groups (%)",
               title=f"NoOp seed {seed}: total zero-variance groups (512 × 8)")
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(output / f"seed{seed}_dead_group_trend.png", dpi=190)
        plt.close(fig)


def plot_recurrence(recurrence, output):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    for seed in sorted({r["seed"] for r in recurrence}):
        run = [r for r in recurrence if r["seed"] == seed and r["prior_dead_reappeared"]]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot([r["step"] for r in run], [r["dead_again_pct"] for r in run], color="#a34d70")
        ax.set(xlabel="Step of next observed appearance", ylabel="Previously-dead groups dead again (%)",
               title=f"NoOp seed {seed}: next-appearance persistence of dead groups")
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(output / f"seed{seed}_dead_group_recurrence.png", dpi=190)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    records = []
    recurrence = []
    coverage = {}
    for seed, segments in SEGMENTS.items():
        if not all((args.inputs / name).is_dir() for name, _, _ in segments):
            print(f"Seed {seed}: skipped (both original/resumed directories not present)")
            continue
        selected = {}
        for name, first, last in segments:
            files = indexed_files(args.inputs / name)
            missing = sorted(set(range(first, last + 1)) - set(files))
            if missing:
                raise ValueError(f"Seed {seed} {name}: missing steps {missing}")
            for step in range(first, last + 1):
                selected[step] = files[step]
        if sorted(selected) != list(range(1, 301)):
            raise ValueError(f"Seed {seed}: incomplete checkpoint-stitched trajectory")
        previous_status = {}
        for step, file in sorted(selected.items()):
            row, statuses = summarize_step(seed, step, file)
            records.append(row)
            previously_dead_reappeared = 0
            dead_again = 0
            prior_easy = prior_hard = easy_again = hard_again = 0
            for prompt, current in statuses.items():
                previous = previous_status.get(prompt)
                if previous in ("easy", "hard"):
                    previously_dead_reappeared += 1
                    dead_again += int(current in ("easy", "hard"))
                    if previous == "easy":
                        prior_easy += 1
                        easy_again += int(current == "easy")
                    else:
                        prior_hard += 1
                        hard_again += int(current == "hard")
                previous_status[prompt] = current
            recurrence.append({
                "seed": seed, "step": step,
                "prior_dead_reappeared": previously_dead_reappeared,
                "dead_again": dead_again,
                "dead_again_pct": (100 * dead_again / previously_dead_reappeared
                                   if previously_dead_reappeared else ""),
                "prior_easy_reappeared": prior_easy,
                "easy_again": easy_again,
                "prior_hard_reappeared": prior_hard,
                "hard_again": hard_again,
            })
        coverage[str(seed)] = {"steps": len(selected), "segments": [list(s) for s in segments]}

    if not records:
        parser.error("No complete seed trajectories found")
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "noop_dead_groups_per_step.csv", records)
    write_csv(args.out / "noop_dead_group_recurrence.csv", recurrence)
    summaries = []
    for seed in sorted({r["seed"] for r in records}):
        run = [r for r in records if r["seed"] == seed]
        summary = {"seed": seed, "steps": len(run), "groups_analyzed": sum(r["groups"] for r in run)}
        for field in ("easy_dead_pct", "hard_dead_pct", "productive_pct", "dead_total_pct"):
            summary[f"mean_{field}"] = sum(r[field] for r in run) / len(run)
        repeat_rows = [r for r in recurrence if r["seed"] == seed]
        denominator = sum(r["prior_dead_reappeared"] for r in repeat_rows)
        numerator = sum(r["dead_again"] for r in repeat_rows)
        summary["dead_reappeared_transitions"] = denominator
        summary["dead_again_transitions"] = numerator
        summary["dead_again_pct_next_appearance"] = (100 * numerator / denominator if denominator else "")
        summaries.append(summary)
    write_csv(args.out / "noop_dead_groups_summary.csv", summaries)
    audit = {"coverage": coverage,
             "method": "UID prefix identifies group; binary score; exactly 512 groups and 8 rollouts required per step",
             "recurrence_definition": "Match exact input text to its previous observed occurrence within the same seed, even across checkpoint segments. Among previously-dead prompts that reappear, count those dead again. Attribute transition to the step of reappearance. Aggregate ratio is denominator-weighted.",
             "recurrence_caveat": "Exact text matching does not merge paraphrases. Prompts without a previous appearance do not enter the denominator. Distinct prompts with identical text cannot be distinguished; within-step uniqueness is required.",
             "caveat": "Training reward-group statistics, not validation accuracy. Checkpoint cuts follow the six-run analysis manifest."}
    (args.out / "noop_dead_groups_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    plot(records, args.out)
    plot_recurrence(recurrence, args.out)
    print(json.dumps({"coverage": coverage, "summaries": summaries, "output": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
