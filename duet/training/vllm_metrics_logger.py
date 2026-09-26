import transfer_queue as tq
import json
import os
import time
from pathlib import Path

from vllm.v1.metrics.loggers import StatLoggerBase


def log_phase(phase: str):
    """
    Records the phase (train/validation) of the timestamps for the vllm metrics.
    This is intended to distinguish between the training and validation phases when recording the vllm-metrics.
    It's called right before the prompts are schedulled to the inference engine (engine) for each phase.
    """
    path = Path(os.environ["DUET_RUN_DIR"]) / "vllm_metrics.jsonl"

    row = {
        "timestamp_ns": time.time_ns(),
        "type": "phase",
        "phase": phase,
    }

    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")



class DuetVLLMLogger(StatLoggerBase):
    def __init__(self, vllm_config, engine_index=0):
        print(f"[DUET] DuetVLLMLogger loaded engine={engine_index}", flush=True)
        self.engine_index = engine_index

        run_dir = Path(os.environ["DUET_RUN_DIR"])
        run_dir.mkdir(parents=True, exist_ok=True)

        self.path = run_dir / "vllm_metrics.jsonl"
        if self.path.exists():
            self.path.unlink()

    def record(
        self,
        scheduler_stats,
        iteration_stats,
        mm_cache_stats=None,
        engine_idx=0,
    ):
        try:
            if scheduler_stats is None:
                return

            row = {
                "timestamp_ns": time.time_ns(),
                "engine_index": engine_idx,
                "running": scheduler_stats.num_running_reqs,
                "waiting": scheduler_stats.num_waiting_reqs,
                "kv_cache_usage": scheduler_stats.kv_cache_usage,
            }

            with self.path.open("a") as f:
                f.write(json.dumps(row) + "\n")

        except Exception as e:
            print(
                f"[DUET] vLLM metrics logger error: {e}",
                flush=True,
            )

    def log_engine_initialized(self):
        pass

