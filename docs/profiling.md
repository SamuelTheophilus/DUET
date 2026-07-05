# Profiling the GRPO rollout-update cycle.

**Model**:  Qwen2.5-Math-1.5B
**Dataset**: GSM8K
**Algorith**: GRPO (verl, default hyperparameters)
**Hardware**: 1-A100-40GB

## What was run
The stock GRPO recipie with default hyperparameters:
  batch_size=512
  rollout=2560 sequences (512 prompts * 5 sequences generated per prompt ).
  max_response_lenght=1024 tokens

Profiled only three steps 


## 2. How it was profiled

- **PyTorch Profiler** via verl's native `global_profiler.tool=torch` (per-role, discrete mode) → Chrome traces on the Modal Volume at `/data/traces/`.
- **Per-prompt generation time** captured by hooking `AgentLoopManager._performance_metrics` (installed lazily via a Ray `worker_process_setup_hook`) → `gen_time_call*.npy`.
- **Per-prompt length + reward** via `trainer.rollout_data_dir` → `rollout_data/{step}.jsonl` (fields: `resp_len`, `output`, `gts`, `score`).


## 3. Rollout deep-dive

Per-prompt generation times (2560 sequences, step 1):

- mean **35.1s**, median **35.7s**, max **65.3s**
- **Slowest 10% of prompts = ‹17.6%›** of total per-prompt generation time
- **Bubble ratio = ‹46%›** — computed via the sweep-line integral `Σ(Q − rₖ)Δtₖ / (T·Q)`, all prompts starting together (verified: coroutines are launched with a single `asyncio.gather`, no concurrency throttle).

The per-prompt *time* distribution is nearly flat with a pile-up at the max, even though response *lengths* are heavy-tailed evidence that short prompts are held in the running batch waiting for the 1024-token stragglers.


## 4. Reproduce

```bash
modal run train.py --setup      # one-time: model + dataset
modal run train.py --train-type=baseline --profile    # 3-step profiled run
mkdir -p ./traces/duet-data
modal volume get duet-volume traces ./traces
modal volume get duet-volume duet.db ./traces/duet-data
```
Then run `notebooks/02_profiling.ipynb`

## 5. Artifacts (on `duet-volume`)

- `traces/*/` — PyTorch Profiler Chrome traces (open in ui.perfetto.dev)
- `traces/gen_time_call*.npy` — per-prompt generation times
- `traces/rollout_data/*.jsonl` — per-prompt prompt/response/reward
- `duet.db` — MLflow metrics
