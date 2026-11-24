# Corrected Seed 90 NoOp — Analysis Provenance

## Canonical training trajectory

Workspace: duet1p0
Volume: duet-volume
Scheduler: NoOp
Seed: 90

The canonical trajectory follows the checkpoint continuation specified by the team:

| Steps | Source run |
|---|---|
| 1–180 | 20261123_004046_noops_run_300_steps_seed_90_correction_run |
| 181–280 | 20261124_005539_noops_correction_run_step180_to_step300_resuming_from_20261123_004046_noops_run |
| 281–300 | 20261124_132909_noops_correction_run_step280_to_step_300_resuming_from_20261124_005539_noops_correction_run |

## MLflow provenance

The middle database contains two overlapping MLflow sessions:

- 4519e2f60ca84b7c9bfc7301e4217c38: steps 181–289
- fb585f7ccfee48308a74b3600a079004: steps 181–203

The analysis selects the first session for steps 181–280 because it covers the checkpoint-selected continuation. The original database is preserved; a filtered copy is used for analysis.

## Metrics verification

- Canonical training steps: 300/300
- Generated response tokens: 405,811,007
- Validation checkpoints: 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300
- Mean validation accuracy, steps 225–300: 86.7703%

All other five experiment trajectories retain their previous source manifests.

## Rollout sampling verification

The assembled rollout directory contains:

- 300 step files
- 7,473 distinct prompt texts
- 512 unique prompts per step
- 8 responses per prompt
- Mean adjacent-step overlap: 2.45485 prompts
- Full-overlap transitions: 0
- High-overlap transitions (>=90%): 0

## Unresolved rollout provenance

The middle source directory appears to contain rollout files from two sessions.

Modal modification times show steps 181–203 were written after the later overlapping session started, while steps 204–280 predate that session.

Consequently, the assembled rollout files may not constitute one uninterrupted checkpoint lineage.

The provisional corrected seed 90 dead-group results (75.3737% dead groups; 90.4485% dead again at next observed appearance) must NOT be represented as verified canonical results until the original rollout provenance is resolved.

The rollout sampling audit remains valid for the assembled files, but it does not establish checkpoint continuity.

## Existing limitations

Seed 90 Refill is a mixed-version experiment. Top-up accounting is unavailable before step 221.

Tokens-to-target interpolation estimates crossings between sparse validation checkpoints. These estimates are descriptive, not directly observed or statistically significant threshold crossings.

## Reproduction

Use scripts/filter_seed90_mlflow.py to prepare the filtered middle database.

Use scripts/seed90_corrected_six_runs.py to regenerate the corrected six-run analysis from local MLflow databases.

Raw MLflow databases and rollout JSONL files are not committed to this PR.
