#!/usr/bin/env bash
set -euo pipefail

RUNS_DIR="/data/runs"

read -rp "Continue from a checkpoint? (y/n): " CONTINUE

OVERRIDES=""

if [[ "$CONTINUE" =~ ^[Yy]([Ee][Ss])?$ ]]; then
    echo
    echo "All paths are relative to $RUNS_DIR/"
    echo

    read -rp "Checkpoint path: " CHECKPOINT
    read -rp "DUET resume file: " DUET_FILE
    read -rp "Total training steps: " TOTAL_STEPS
    read -rp "Checkpoint save frequency [20]: " SAVE_FREQ

    if [[ -z "$CHECKPOINT" || -z "$DUET_FILE" || ! "$TOTAL_STEPS" =~ ^[0-9]+$ ]]; then
        echo "Error: checkpoint, DUET resume file, and valid total steps are required."
        exit 1
    fi

    SAVE_FREQ=${SAVE_FREQ:-20}

    CHECKPOINT="${RUNS_DIR}/${CHECKPOINT#/data/runs/}"
    DUET_FILE="${RUNS_DIR}/${DUET_FILE#/data/runs/}"

    OVERRIDES="trainer.resume_mode=resume_path \
trainer.resume_from_path=$CHECKPOINT \
+trainer.duet.resume_from_file=$DUET_FILE \
trainer.del_local_ckpt_after_load=false \
trainer.total_training_steps=$TOTAL_STEPS \
trainer.save_freq=$SAVE_FREQ"
fi

echo
echo "========================================"
echo "       TRAINING CONFIGURATION"
echo "========================================"
echo "Mode:        $([[ -n "$OVERRIDES" ]] && echo Resume || echo New run)"

if [[ -n "$OVERRIDES" ]]; then
    echo "Checkpoint:  $CHECKPOINT"
    echo "DUET file:   $DUET_FILE"
    echo "Total steps: $TOTAL_STEPS"
    echo "Save every:  $SAVE_FREQ steps"
fi

echo "========================================"
echo

read -rp "Proceed with training? (y/n): " CONFIRM

if [[ ! "$CONFIRM" =~ ^[Yy]([Ee][Ss])?$ ]]; then
    echo "Cancelled."
    exit 0
fi

if [[ -n "$OVERRIDES" ]]; then
    modal run -i -d train.py --profile --overrides "$OVERRIDES"
else
    modal run -i -d train.py --profile
fi

