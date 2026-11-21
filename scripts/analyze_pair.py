"""
DUET pair-run analysis.

Stage 1: validated training-log parsing.

Training records use:
    step:N - metric/key:value - metric/key:value ...

Missing metric values remain NaN.
Missing training steps are never interpolated.

No GPU, Modal, or training dependencies required.
"""

from pathlib import Path
import re

import pandas as pd


ANSI = re.compile(r"\x1b\[[0-9;]*m")

STEP_PATTERN = re.compile(
    r"\bstep:(\d+)\s+-\s+"
)

METRIC_PATTERN = re.compile(
    r"(?<!\S)([A-Za-z][A-Za-z0-9_./-]*)"
    r":(-?(?:\d+(?:\.\d*)?|\.\d+)"
    r"(?:[eE][+-]?\d+)?|nan|inf|-inf)"
    r"(?=\s|$)",
    re.IGNORECASE,
)


def parse_training_log(path):
    """
    Parse a DUET training log.

    Returns
    -------
    frame : pandas.DataFrame
        One row per observed training step.
    info : dict
        Source, row count, gaps, and metric count.
    """

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(path)

    records = []

    with path.open(
        "r", encoding="utf-8", errors="replace"
    ) as file:

        for line_number, raw_line in enumerate(file, 1):

            line = ANSI.sub("", raw_line)

            match = STEP_PATTERN.search(line)

            if match is None:
                continue

            step = int(match.group(1))

            metric_text = line[match.end():]

            metrics = {
                key: float(value)
                for key, value in METRIC_PATTERN.findall(
                    metric_text
                )
            }

            # A bare step mention without numeric metrics
            # is not a valid training metric record.
            if not metrics:
                continue

            records.append({
                "step": step,
                "line_number": line_number,
                **metrics,
            })

    if not records:
        raise ValueError(
            f"No valid training records found: {path}"
        )

    frame = pd.DataFrame(records)

    if frame["step"].duplicated().any():
        duplicates = (
            frame.loc[
                frame["step"].duplicated(False),
                "step",
            ]
            .tolist()
        )

        raise ValueError(
            f"Duplicate training steps: {duplicates}"
        )

    frame = (
        frame.sort_values("step")
        .reset_index(drop=True)
    )

    steps = set(frame["step"].astype(int))

    missing = sorted(
        set(
            range(
                min(steps),
                max(steps) + 1,
            )
        ) - steps
    )

    metrics = [
        column for column in frame.columns
        if column not in {"step", "line_number"}
    ]

    info = {
        "source": str(path),
        "records": len(frame),
        "first_step": min(steps),
        "last_step": max(steps),
        "missing_steps": missing,
        "metric_count": len(metrics),
    }

    return frame, info



# W8-M35: PAIRWISE ALIGNMENT AND VALIDATION

def align_training_runs(run_a, run_b, name_a="run_a", name_b="run_b"):
    """
    Outer-align two parsed training runs by observed step.

    The output retains missing steps as NaN. It does not
    interpolate, forward-fill, or shift either trajectory.
    """

    if name_a == name_b:
        raise ValueError("Run names must be distinct.")

    for name, frame in [(name_a, run_a), (name_b, run_b)]:
        if "step" not in frame:
            raise ValueError(f"{name}: missing step column")
        if frame["step"].duplicated().any():
            raise ValueError(f"{name}: duplicate steps")

    a = run_a.copy()
    b = run_b.copy()

    a["present_a"] = True
    b["present_b"] = True

    aligned = a.merge(
        b,
        on="step",
        how="outer",
        suffixes=(f"_{name_a}", f"_{name_b}"),
        validate="one_to_one",
    )

    aligned = aligned.sort_values("step").reset_index(drop=True)

    aligned["present_a"] = aligned["present_a"].eq(True)
    aligned["present_b"] = aligned["present_b"].eq(True)

    aligned["matched"] = (
        aligned["present_a"] & aligned["present_b"]
    )

    return aligned


def validate_cumulative_metrics(frame, run_name):
    """
    Check cumulative counters without filling missing steps.

    Negative differences between consecutive observed records
    are reported as violations. Gaps are retained in the report.
    """

    columns = [
        "duet/cumulative_response_tokens",
        "duet/cumulative_total_tokens",
        "duet/cumulative_optimizer_steps",
    ]

    ordered = frame.sort_values("step").reset_index(drop=True)

    rows = []

    for column in columns:

        if column not in ordered.columns:
            rows.append({
                "run": run_name,
                "metric": column,
                "status": "MISSING",
                "observed_values": 0,
                "decreases": None,
                "first_value": None,
                "last_value": None,
            })
            continue

        valid = ordered[
            ["step", column]
        ].dropna()

        differences = valid[column].diff()

        decreases = int((differences < -1e-9).sum())

        rows.append({
            "run": run_name,
            "metric": column,
            "status": "PASS" if decreases == 0 else "WARNING",
            "observed_values": len(valid),
            "decreases": decreases,
            "first_value": (
                float(valid[column].iloc[0])
                if len(valid) else None
            ),
            "last_value": (
                float(valid[column].iloc[-1])
                if len(valid) else None
            ),
        })

    return pd.DataFrame(rows)



# W8-M39: REUSABLE PAIR-COMPARISON FIGURES

def generate_pair_figures(run_a, run_b, output_dir,
                          name_a="NoOp", name_b="DifficultyOnly"):
    """
    Produce seven aggregate training-run comparison figures.

    Missing steps remain missing. Accumulated tokens are
    cumulative sums over observed log records only.

    Returns a dataframe summarizing recorded run metrics.
    """
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = {
        name_a: run_a.sort_values("step").copy(),
        name_b: run_b.sort_values("step").copy(),
    }

    colors = {
        name_a: "#2563EB",
        name_b: "#D97706",
    }

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "savefig.dpi": 300,
    })

    def save(fig, filename):
        fig.tight_layout()
        fig.savefig(output_dir / (filename + ".pdf"),
                    bbox_inches="tight")
        fig.savefig(output_dir / (filename + ".png"),
                    bbox_inches="tight", dpi=300)
        plt.close(fig)

    def line_plot(metric, ylabel, title, filename, scale=1.0):
        fig, ax = plt.subplots(figsize=(8, 4.5))

        for name, df in runs.items():
            if metric not in df.columns:
                raise ValueError(
                    f"{name}: required metric missing: {metric}"
                )

            steps = np.arange(
                int(df["step"].min()),
                int(df["step"].max()) + 1,
            )
            series = df.set_index("step")[metric].reindex(steps)

            ax.plot(steps, series / scale,
                    label=name, color=colors[name], linewidth=1.8)

        ax.set_title(title)
        ax.set_xlabel("Training step")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.18)
        ax.legend(frameon=False)
        save(fig, filename)

    # 1. Rollout reward
    line_plot(
        "critic/rewards/mean",
        "Mean rollout reward",
        "Training Reward Comparison",
        "01_training_reward",
    )

    # 2. Per-step response tokens
    line_plot(
        "duet/generated_response_tokens",
        "Generated response tokens (millions)",
        "Generated Response Tokens per Step",
        "02_response_tokens",
        scale=1e6,
    )

    # 3. Response length
    line_plot(
        "response_length/mean",
        "Mean response length (tokens)",
        "Response Length Comparison",
        "03_response_length",
    )

    # 4. Step duration
    line_plot(
        "timing_s/step",
        "Reported step duration (seconds)",
        "Training-Step Duration",
        "04_step_duration",
    )

    # 5. Scheduler skips
    fig, ax = plt.subplots(figsize=(8, 4.5))

    for name, df in runs.items():
        for key in [
            "duet/prompts_skipped_easy",
            "duet/prompts_skipped_hard",
        ]:
            if key not in df.columns:
                raise ValueError(f"{name}: missing {key}")

        steps = np.arange(
            int(df["step"].min()),
            int(df["step"].max()) + 1,
        )
        skipped = (
            df.set_index("step")["duet/prompts_skipped_easy"]
            + df.set_index("step")["duet/prompts_skipped_hard"]
        ).reindex(steps)

        ax.plot(
            steps, skipped, label=name,
            color=colors[name], linewidth=1.8,
        )

    ax.set_title("Logged Scheduler Skips")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Skipped prompts per step")
    ax.grid(alpha=0.18)
    ax.legend(frameon=False)
    save(fig, "05_scheduler_skips")

    # 6. Optimizer progress
    line_plot(
        "duet/cumulative_optimizer_steps",
        "Cumulative optimizer steps",
        "Optimizer Progress",
        "06_optimizer_progress",
    )

    # 7. Reward against accumulated observed tokens
    fig, ax = plt.subplots(figsize=(8, 4.5))

    for name, df in runs.items():
        for key in [
            "duet/generated_response_tokens",
            "critic/rewards/mean",
        ]:
            if key not in df.columns:
                raise ValueError(f"{name}: missing {key}")

        x = (
            df["duet/generated_response_tokens"].cumsum()
            / 1e6
        )

        ax.plot(
            x, df["critic/rewards/mean"],
            label=name, color=colors[name], linewidth=1.8,
        )

    ax.set_title("Training Reward vs Observed Response Tokens")
    ax.set_xlabel("Accumulated observed response tokens (millions)")
    ax.set_ylabel("Mean rollout reward")
    ax.grid(alpha=0.18)
    ax.legend(frameon=False)
    save(fig, "07_reward_vs_tokens")

    # Run-level numerical summary
    summary_rows = []

    for name, df in runs.items():
        summary_rows.append({
            "run": name,
            "observed_steps": len(df),
            "first_step": int(df["step"].min()),
            "last_step": int(df["step"].max()),
            "mean_rollout_reward":
                df["critic/rewards/mean"].mean(),
            "mean_response_length":
                df["response_length/mean"].mean(),
            "mean_step_duration_s":
                df["timing_s/step"].mean(),
            "observed_generated_response_tokens":
                df["duet/generated_response_tokens"].sum(),
            "observed_total_tokens":
                df["perf/total_num_tokens"].sum(),
            "observed_skipped_prompts":
                (
                    df["duet/prompts_skipped_easy"]
                    + df["duet/prompts_skipped_hard"]
                ).sum(),
        })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "run_summary.csv", index=False)

    return summary



# W8-M40: COMMAND-LINE INTERFACE

def resolve_training_log(run_path):
    """Resolve a .log file or a directory containing one log."""

    path = Path(run_path)

    if path.is_file():
        if path.suffix.lower() != ".log":
            raise ValueError(
                f"Expected a .log file: {path}"
            )
        return path

    if not path.is_dir():
        raise FileNotFoundError(path)

    candidates = sorted(path.glob("*.log"))

    if len(candidates) == 0:
        raise FileNotFoundError(
            f"No .log files found in {path}"
        )

    if len(candidates) > 1:
        raise ValueError(
            f"Multiple .log files in {path}; "
            "specify the desired log directly."
        )

    return candidates[0]


def analyze_pair(run_a, run_b, output_dir,
                 name_a="Run A", name_b="Run B"):
    """
    Execute the aggregate DUET pair-analysis workflow.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    path_a = resolve_training_log(run_a)
    path_b = resolve_training_log(run_b)

    frame_a, info_a = parse_training_log(path_a)
    frame_b, info_b = parse_training_log(path_b)

    if name_a == name_b:
        raise ValueError("Run labels must be different.")

    frame_a.to_csv(
        output_dir / "run_a_metrics.csv",
        index=False,
    )

    frame_b.to_csv(
        output_dir / "run_b_metrics.csv",
        index=False,
    )

    aligned = align_training_runs(
        frame_a, frame_b,
        name_a="a", name_b="b",
    )

    aligned.to_csv(
        output_dir / "aligned_steps.csv",
        index=False,
    )

    validation = pd.concat(
        [
            validate_cumulative_metrics(frame_a, name_a),
            validate_cumulative_metrics(frame_b, name_b),
        ],
        ignore_index=True,
    )

    validation.to_csv(
        output_dir / "cumulative_validation.csv",
        index=False,
    )

    figures_dir = output_dir / "figures"

    summary = generate_pair_figures(
        frame_a,
        frame_b,
        figures_dir,
        name_a=name_a,
        name_b=name_b,
    )

    import json

    metadata = {
        "run_a": info_a,
        "run_b": info_b,
        "matched_steps": int(aligned["matched"].sum()),
        "unmatched_steps": aligned.loc[
            ~aligned["matched"], "step"
        ].astype(int).tolist(),
        "figure_count": 7,
        "notes": [
            "Training reward is not held-out accuracy.",
            "Missing steps are not interpolated.",
            "Token sums cover observed log records only.",
            "Logged token counts are not GPU-hours.",
        ],
    }

    (output_dir / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    return summary, metadata


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Compare two DUET training runs "
            "using logged aggregate metrics."
        )
    )

    parser.add_argument(
        "--run-a", required=True,
        help="First run directory or .log file",
    )

    parser.add_argument(
        "--run-b", required=True,
        help="Second run directory or .log file",
    )

    parser.add_argument(
        "--out", required=True,
        help="Output directory",
    )

    parser.add_argument(
        "--name-a", default="Run A",
        help="Display name for the first run",
    )

    parser.add_argument(
        "--name-b", default="Run B",
        help="Display name for the second run",
    )

    args = parser.parse_args(argv)

    summary, metadata = analyze_pair(
        run_a=args.run_a,
        run_b=args.run_b,
        output_dir=args.out,
        name_a=args.name_a,
        name_b=args.name_b,
    )

    print("=" * 70)
    print("DUET - PAIR ANALYSIS")
    print("=" * 70)

    print("\nRun summary:")
    print(summary.to_string(index=False))

    print("\nMatched steps:",
          metadata["matched_steps"])

    print("Unmatched steps:",
          metadata["unmatched_steps"])

    print("\nFigures generated: 7 PDF + 7 PNG")
    print("Analysis complete.")


if __name__ == "__main__":
    main()
