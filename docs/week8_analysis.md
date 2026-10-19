# Week 8 — Predictor, MoPPS, and Pair Analysis

## Implemented components

### MoPPS-inspired posterior

Source: `duet/analysis/mopps.py`

The module implements per-prompt Beta posteriors,
evidence-discounted updates, posterior means, Thompson
sampling, and posterior-predictive dead-group probabilities.

Discount convention:

- alpha_next = 1 + lambda * (alpha_previous - 1) + successes
- beta_next = 1 + lambda * (beta_previous - 1) + failures

This reproduces the Week 8 notebook implementation.
It is not a complete reproduction of the published MoPPS system.

The posterior-predictive dead probability is an additional
diagnostic, distinct from Thompson-sampling prompt selection.

### Pair-run analysis

Source: `scripts/analyze_pair.py`

The script parses DUET training logs, aligns runs by actual
training steps, validates cumulative counters, and produces
seven comparison figures in PDF and PNG formats.

Example command:

```bash
python scripts/analyze_pair.py --run-a /path/to/noop_run --run-b /path/to/difficulty_run --out /path/to/output --name-a NoOp --name-b DifficultyOnly
```

Each input may be a `.log` file or a directory containing
exactly one `.log` file.

Outputs include parsed metrics, aligned observations,
cumulative-counter validation, metadata, run summary,
and seven figures.

Pair-analysis dependencies: `pandas`, `numpy`, `matplotlib`.
These must be installed in the execution environment.
The MoPPS module itself uses only Python's standard library.

## Automated tests

Test files:

- `tests/analysis/test_mopps.py`
- `tests/analysis/test_analyze_pair.py`

Run from the repository root:

```bash
python -m pytest -q --noconftest tests/analysis/test_mopps.py tests/analysis/test_analyze_pair.py
```

The `--noconftest` option avoids loading unrelated
training-scheduler dependencies.

## Verified results

MoPPS was validated using 128 recurring prompts and
16 observations per prompt. Both discount settings
reproduced the notebook's 4,096 posterior predictions.

The pair analyzer recovered 177 NoOp and 180 DifficultyOnly
training-step records, with 177 matching step identifiers.

Unmatched steps: 70, 179, 180.
NoOp step 70 was missing from the captured log.

All 31 automated analysis tests passed.

## Interpretation and limitations

- The predictor comparisons use one logged trajectory.
- Independent training-seed replication is unavailable.
- Complete prompt-level histories are unavailable for
  the long-running NoOp and DifficultyOnly experiments.
- Bidirectional cross-run validation remains blocked.
- Training reward is not held-out evaluation accuracy.
- Skipped-prompt counts are events, not unique prompt IDs.
- DUET cumulative token counters stayed zero in both runs.
- Token analysis uses populated per-step metrics instead.
- Summed tokens cover observed training records only.
- Missing steps are not reconstructed or interpolated.
- Token counts cannot directly establish GPU-hours or cost.
- MoPPS posterior ranking and Thompson sampling are
  distinct evaluations.
- Offline selection results do not prove downstream
  training improvements.

## Research artifacts

The Week 8 notebook, bootstrap outputs, comparison CSVs,
figures, and detailed research notes remain in the separate
Fellowship Week 8 workspace.

They are not included in this repository contribution.
