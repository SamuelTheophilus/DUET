"""Profiling metrics for DUET GRPO rollouts.

Single source of truth for the numbers reported in notebooks/02_profiling.ipynb.

Two families of metric live here:

1. Rollout-time metrics computed on the per-prompt ``generate_sequences`` array
   dumped by ``duet.training.profile_patch`` (``gen_time_call*.npy``):
     - ``bubble_ratio_sweep``  : fraction of the rollout window spent as idle
                                 batch slots while faster prompts wait on
                                 stragglers (idle-slot cost).
     - ``slowest_ten_percent`` : share (%) of total time consumed by the
                                 slowest 10% of prompts (tail-mass cost).

2. Trace-level helpers for the torch profiler json (``load_traces``,
   ``wall_clock_span``) and a per-prompt (length, reward) table.

Important caveat on ``slowest_ten_percent`` when run on *durations*
-----------------------------------------------------------------
Under vLLM continuous batching with a ``max_response_length`` cap, per-request
duration understates the real tail: the cap clips long generations before they
are measured, and shared decode steps smooth per-request wall time across the
batch. The tail DUET actually targets is driven by *response length*, so run
``slowest_ten_percent`` on token length (see ``get_per_prompt_length_and_reward``)
for the true picture, and read the duration-based number as a lower bound that
the bubble ratio (idle cost) sits well above.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np

__all__ = [
    "load_traces",
    "wall_clock_span",
    "bubble_ratio_sweep",
    "slowest_pct_share",
    "slowest_ten_percent",
    "summarize",
    "get_per_prompt_length_and_reward",
    "plot_per_prompt_generation_time",
]


# --------------------------------------------------------------------------- #
# torch-profiler trace helpers
# --------------------------------------------------------------------------- #
def load_traces(path: str | Path) -> list[dict]:
    """Load complete (``ph == "X"``) duration events from a torch-profiler trace.

    Handles both plain ``.json`` and gzipped ``.json.gz`` traces.
    """
    path = Path(path)
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        data = json.load(f)
    events = data.get("traceEvents", data) if isinstance(data, dict) else data
    return [e for e in events if e.get("ph") == "X" and "dur" in e]


def wall_clock_span(events: list[dict]) -> float:
    """Wall-clock span (in microseconds) covered by a list of trace events."""
    start = min(e["ts"] for e in events)
    end = max(e["ts"] + e["dur"] for e in events)
    return end - start


# --------------------------------------------------------------------------- #
# rollout-time metrics (per-prompt generate_sequences array)
# --------------------------------------------------------------------------- #
def bubble_ratio_sweep(durations) -> float:
    r"""Bubble ratio for a batch whose requests all start at t=0.

    ``sum_k (Q - r_k) * dt_k / (T * Q)``, where ``Q`` is the number of requests,
    ``r_k`` the number still running over interval ``dt_k``, and ``T`` the max
    duration. Returns a value in ``[0, 1]``; higher means more idle batch slots
    waiting on stragglers.
    """
    d = np.asarray(durations, dtype=float)
    Q = len(d)
    T = d.max()
    bubble = 0.0
    running = Q
    prev = 0.0
    for end_t in np.sort(d):  # ascending end times
        bubble += (Q - running) * (end_t - prev)
        running -= 1
        prev = end_t
    return bubble / (Q * T)


def slowest_pct_share(values, pct: float = 10.0) -> float:
    """Share (%) of the total consumed by the slowest ``pct`` percent of items.

    Divides by the sum of the *same* array (this was the bug in the original
    notebook, which divided by a stale global). ``k`` is floored at 1 so a small
    array never triggers the ``x[-0:] == whole-array`` slicing quirk that would
    silently report 100%.
    """
    d = np.sort(np.asarray(values, dtype=float))
    k = max(1, round(len(d) * pct / 100.0))
    return float(d[-k:].sum() / d.sum() * 100.0)


def slowest_ten_percent(values) -> float:
    """Share (%) of total consumed by the slowest 10% (see ``slowest_pct_share``)."""
    return slowest_pct_share(values, 10.0)


def summarize(values, cap: float | None = None) -> dict[str, float]:
    """Scalar summary of a per-prompt array (durations or token lengths).

    ``bubble_ratio`` is only physically meaningful for durations. Pass ``cap``
    (e.g. ``max_response_length``) to also report the fraction of prompts that
    hit the cap, which is what compresses the duration tail.
    """
    d = np.asarray(values, dtype=float)
    out = {
        "count": int(d.size),
        "mean": float(d.mean()),
        "p50": float(np.percentile(d, 50)),
        "p90": float(np.percentile(d, 90)),
        "p99": float(np.percentile(d, 99)),
        "max": float(d.max()),
        "slowest_10pct_share": slowest_ten_percent(d),
        "bubble_ratio": bubble_ratio_sweep(d),
    }
    if cap is not None:
        out["cap_hit_frac"] = float((d >= cap).mean())
    return out


# --------------------------------------------------------------------------- #
# per-prompt (length, reward) table
# --------------------------------------------------------------------------- #
def get_per_prompt_length_and_reward(rollout_df, tokenizer):
    """Return a per-prompt table with tokenized response length and reward.

    ``tokenizer`` is passed explicitly (the original relied on a notebook
    global). Does not mutate the input frame.
    """
    df = rollout_df.copy()
    df["resp_len"] = df["output"].apply(lambda s: len(tokenizer(s)["input_ids"]))
    return df[["gts", "output", "resp_len", "score"]]


# --------------------------------------------------------------------------- #
# plotting
# --------------------------------------------------------------------------- #
def plot_per_prompt_generation_time(
    durations,
    *,
    ax=None,
    bins: int = 40,
    title: str = "Per-prompt generation time",
    show: bool = True,
):
    """Histogram of per-prompt generation times. Returns the matplotlib axis."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    ax.hist(np.asarray(durations, dtype=float), bins=bins, edgecolor="white")
    ax.set_xlabel("Per-prompt generation time (s)")
    ax.set_ylabel("Number of prompts")
    ax.set_title(title)
    if show:
        plt.show()
    return ax
