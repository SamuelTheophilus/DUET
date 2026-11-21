"""Tests for DUET aggregate pair analysis."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "analyze_pair.py"
)

SPEC = importlib.util.spec_from_file_location(
    "duet_analyze_pair_test",
    SCRIPT,
)

MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def make_log(tmp_path, text):
    path = tmp_path / "train.log"
    path.write_text(text, encoding="utf-8")
    return path


def test_parse_valid_records(tmp_path):
    path = make_log(
        tmp_path,
        "step:1 - critic/rewards/mean:0.5 - "
        "duet/prompts_selected:512\n"
        "step:2 - critic/rewards/mean:0.7 - "
        "duet/prompts_selected:512\n",
    )

    frame, info = MODULE.parse_training_log(path)

    assert len(frame) == 2
    assert info["metric_count"] == 2
    assert frame["step"].tolist() == [1, 2]


def test_missing_step_preserved(tmp_path):
    path = make_log(
        tmp_path,
        "step:1 - critic/rewards/mean:0.5\n"
        "step:3 - critic/rewards/mean:0.7\n",
    )

    frame, info = MODULE.parse_training_log(path)

    assert frame["step"].tolist() == [1, 3]
    assert info["missing_steps"] == [2]


def test_duplicate_step_rejected(tmp_path):
    path = make_log(
        tmp_path,
        "step:1 - critic/rewards/mean:0.5\n"
        "step:1 - critic/rewards/mean:0.7\n",
    )

    with pytest.raises(ValueError, match="Duplicate"):
        MODULE.parse_training_log(path)


def test_configuration_line_ignored(tmp_path):
    path = make_log(
        tmp_path,
        "'total_training_steps': -1\n"
        "step:1 - critic/rewards/mean:0.5\n",
    )

    frame, _ = MODULE.parse_training_log(path)

    assert len(frame) == 1


def test_missing_file_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        MODULE.parse_training_log(
            tmp_path / "missing.log"
        )


def test_alignment_preserves_unmatched_steps():
    a = pd.DataFrame({
        "step": [1, 3],
        "reward": [0.2, 0.4],
    })

    b = pd.DataFrame({
        "step": [1, 2, 3],
        "reward": [0.2, 0.3, 0.4],
    })

    aligned = MODULE.align_training_runs(a, b)

    assert aligned["step"].tolist() == [1, 2, 3]
    assert aligned["matched"].tolist() == [
        True, False, True
    ]


def test_duplicate_alignment_rejected():
    a = pd.DataFrame({
        "step": [1, 1],
        "reward": [0.2, 0.3],
    })

    b = pd.DataFrame({
        "step": [1],
        "reward": [0.4],
    })

    with pytest.raises(ValueError, match="duplicate"):
        MODULE.align_training_runs(a, b)


def test_cumulative_counter_decrease_detected():
    frame = pd.DataFrame({
        "step": [1, 2, 3],
        "duet/cumulative_response_tokens": [
            100, 150, 120
        ],
    })

    report = MODULE.validate_cumulative_metrics(
        frame, "test"
    )

    row = report.loc[
        report["metric"]
        == "duet/cumulative_response_tokens"
    ].iloc[0]

    assert row["status"] == "WARNING"
    assert row["decreases"] == 1


def test_cumulative_counter_missing_reported():
    frame = pd.DataFrame({
        "step": [1, 2],
    })

    report = MODULE.validate_cumulative_metrics(
        frame, "test"
    )

    assert (report["status"] == "MISSING").all()


def test_resolve_direct_log(tmp_path):
    path = make_log(
        tmp_path,
        "step:1 - critic/rewards/mean:0.5\n",
    )

    assert MODULE.resolve_training_log(path) == path


def test_resolve_directory_log(tmp_path):
    path = make_log(
        tmp_path,
        "step:1 - critic/rewards/mean:0.5\n",
    )

    assert MODULE.resolve_training_log(tmp_path) == path


def test_reject_directory_multiple_logs(tmp_path):
    (tmp_path / "a.log").write_text(
        "step:1 - x:1\n",
        encoding="utf-8",
    )

    (tmp_path / "b.log").write_text(
        "step:1 - x:2\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Multiple"):
        MODULE.resolve_training_log(tmp_path)
