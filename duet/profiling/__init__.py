"""DUET rollout profiling metrics."""

from duet.profiling.metrics import (
    bubble_ratio_sweep,
    get_per_prompt_length_and_reward,
    load_traces,
    plot_per_prompt_generation_time,
    slowest_pct_share,
    slowest_ten_percent,
    summarize,
    wall_clock_span,
)

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
