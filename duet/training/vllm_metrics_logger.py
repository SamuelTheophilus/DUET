import transfer_queue as tq
import json
import os
import time
from pathlib import Path
from dataclasses import is_dataclass, fields

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
        print(
            f"[DUET] DuetVLLMLogger loaded engine={engine_index}",
            flush=True,
        )

        self.engine_index = engine_index

        run_dir = Path(os.environ["DUET_RUN_DIR"])
        run_dir.mkdir(parents=True, exist_ok=True)

        self.path = run_dir / "vllm_metrics.jsonl"
        self.request_path = run_dir / "vllm_request_stats.jsonl"

    def record(
        self,
        scheduler_stats,
        iteration_stats,
        mm_cache_stats=None,
        engine_idx=0,
    ):
        try:
            # Log finished requests and preemptions
            if iteration_stats is not None:
                if not getattr(self, "_fields_printed", False):
                    if is_dataclass(iteration_stats):
                        print(
                            "[DUET] IterationStats fields:",
                            [f.name for f in fields(iteration_stats)],
                            flush=True,
                        )
                    else:
                        print(vars(iteration_stats), flush=True)

                    self._fields_printed = True



                with self.request_path.open("a") as f:
                    for fr in iteration_stats.finished_requests:
                        f.write(json.dumps({
                            "timestamp_ns": time.time_ns(),
                            "engine_index": engine_idx,
                            "type": "finished",
                            "queued_s": fr.queued_time,
                            "gen_tokens": fr.num_generation_tokens,
                            "prompt_tokens": fr.num_prompt_tokens,
                            "e2e_s": fr.e2e_latency,
                        }) + "\n")

                    n_pre = getattr(
                        iteration_stats,
                        "num_preempted_reqs",
                        None,
                    )

                    if n_pre:
                        f.write(json.dumps({
                            "timestamp_ns": time.time_ns(),
                            "engine_index": engine_idx,
                            "type": "preempted",
                            "n": n_pre,
                        }) + "\n")

            # Log existing scheduler metrics
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
