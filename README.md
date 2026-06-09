# DUET

In RL for LLM reasoning (PPO, GRPO, DAPO), the rollout phase dominates wall-clock time — accounting for up to 70% of total training time when generating long chain-of-thought trajectories. Two lines of attack exist in the literature: **length-aware scheduling** (SortedRL, RollPacker) and **selective rollout** (GRESO, SPEED-RL, HIVE). Both work. Neither has been combined.

DUET's claim: length and difficulty are correlated but not redundant. A short prompt can be uninformative (all-correct group, zero advantage). A long prompt can be highly informative (intermediate difficulty, large gradient signal). A joint predictor over both signals — plus a scheduler that uses both — should beat either signal alone on compute-per-accuracy.

**Contributions:**
1. A lightweight joint predictor (length + difficulty) trained on rollout history, costing less than 5% of one rollout step
2. A scheduler that uses joint predictions to length-pack rollout batches and filter low-information prompts
3. Evaluation on Qwen2.5-Math-1.5B and 7B with LoRA on GSM8K, MATH-500, and AIME-24, showing joint scheduling beats SortedRL-only and GRESO-only baselines on time-to-accuracy
4. An SNR decomposition showing each signal contributes independently to gradient quality

---

## Setup

### 1. Clone the repo

```bash
git clone <repo-url>
cd DUET
```

### 2. Create a virtual environment

Using `uv` (recommended):
```bash
uv venv
source .venv/bin/activate
```

Or with standard Python:
```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Run the bootstrap script

> **Important:** Run this from the root of the repo (`DUET/`). Running it from any other directory will cause failures.

```bash
bash scripts/bootstrap.sh
```

The script will walk you through:
1. Installing all dependencies
2. Creating the required config files (`config/image.yaml`, `config/infra.yaml`)
3. Setting up and logging into your Modal workspace
4. Running a verification test on a Modal GPU container

---

## Project Structure

```
DUET/
├── image.py            # Modal container image definition
├── tracking.py         # MLflow experiment tracking server
├── verify_image.py     # GPU verification script
├── duet/               # Main package
│   └── training/       # Training code and MLflow logging
├── config/
│   ├── image.example.yaml   # Template — copy to image.yaml
│   └── infra.example.yaml   # Template — copy to infra.yaml
├── scripts/
│   └── bootstrap.sh    # One-time setup script
└── docs/               # Additional documentation
```

## Requirements

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/) or `pip`
- A [Modal](https://modal.com) account
