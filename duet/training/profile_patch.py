import os
import sys


from ..scheduler.trainer import install_duet_trainer
from .vllm_metrics_logger import log_phase


trace_dir = "/data/test_traces"


def get_trace_dir():
    run_dir = os.environ["DUET_RUN_DIR"]
    return os.path.join(run_dir, "gen_times")


def get_worker_gen_time_dir():
    run_dir = os.environ["DUET_RUN_DIR"]
    return os.path.join(run_dir, "worker_generation_time")


def apply(module):
    try:
        import numpy as np

        AgentLoopManager = module.AgentLoopManager
    except Exception as e:
        print(f"[duet] gen-time patch skipped: {e}")
        return
    # trace_dir = get_trace_dir()

    os.makedirs(trace_dir, exist_ok=True)
    orig = AgentLoopManager._performance_metrics
    state = {"call": 0}

    def new_performance_metrics(self, metrics, output):
        print("*" * 80)
        print("Saving generation_times")
        print("*" * 80)

        arr = np.array([m["generate_sequences"] for chunk in metrics for m in chunk])
        np.save(f"{trace_dir}/_gen_time_call{state['call']:03d}.npy", arr)
        state["call"] += 1
        return orig(self, metrics, output)

    AgentLoopManager._performance_metrics = new_performance_metrics
    print("[duet] AgentLoopManager._performance_metrics patched")

    install_duet_trainer()


def patch_agent_loop_worker_timing(module):
    import json
    import time
    import os
    import asyncio
    import torch

    try:
        actor_class = module.AgentLoopWorkerTQ
        AgentLoopWorkerTQ = actor_class.__ray_metadata__.modified_class

        if getattr(AgentLoopWorkerTQ, "_duet_timing_patched", False):
            return

        original = AgentLoopWorkerTQ.generate_sequences

        async def wrapped(self, batch):
            # validate = batch[0]["extra_info"].get("validate", False)
            validate = batch["validate"] if "validate" in batch else False
            # batch.pop("validate", None)

            log_phase("validate" if validate else "train")

            print(
                f"[DUET_WORKER_TIMING] entered worker={os.getpid()} "
                f"prompts={len(batch)}"
            )

            start = time.perf_counter()
            worker_id = os.getpid()

            # Tasks already running before this batch arrived
            before = set(self.background_tasks)

            # Original method spawns _run_prompt() tasks and returns
            await original(self, batch)

            # Tasks created for this batch
            batch_tasks = [task for task in self.background_tasks if task not in before]

            print(
                f"[DUET_WORKER_TIMING] worker={worker_id} "
                f"captured_tasks={len(batch_tasks)}"
            )

            async def watch_completion():
                await asyncio.gather(*batch_tasks, return_exceptions=True)

                if validate:
                    return

                elapsed = time.perf_counter() - start

                step = batch["global_steps"]
                if isinstance(step, torch.Tensor):
                    step = step.flatten()[0].item()

                worker_gen_time_dir = get_worker_gen_time_dir()
                os.makedirs(worker_gen_time_dir, exist_ok=True)

                record = {
                    "step": step,
                    "worker_id": worker_id,
                    "num_prompts": len(batch),
                    "wall_time": elapsed,
                }

                write_path = os.path.join(
                    worker_gen_time_dir, f"worker_{worker_id}.jsonl"
                )

                with open(write_path, "a") as f:
                    f.write(json.dumps(record) + "\n")

                print(
                    f"[DUET_WORKER_TIME] "
                    f"worker={worker_id} "
                    f"prompts={len(batch)} "
                    f"wall_time={elapsed:.4f}"
                )

            # Important: don't await this here.
            watcher = asyncio.create_task(watch_completion())
            self.background_tasks.add(watcher)
            watcher.add_done_callback(self.background_tasks.discard)

        AgentLoopWorkerTQ.generate_sequences = wrapped
        AgentLoopWorkerTQ._duet_timing_patched = True

    except Exception as error:
        print(f"[duet] patching agent loop worker failed.\n Error: {str(error)}")

    print("*" * 80)
    print("[duet] patching agent loop worker success.")
    print("*" * 80)


def patch_llm_request_timing(module):
    import json
    import os
    import time

    LLMServerClient = module.LLMServerClient

    if getattr(LLMServerClient, "_duet_request_timing_patched", False):
        return

    original = LLMServerClient.generate

    async def wrapped(self, request_id, *args, **kwargs):
        start_time_ns = time.time_ns()
        start_perf_ns = time.perf_counter_ns()

        output = None

        try:
            output = await original(
                self,
                request_id,
                *args,
                **kwargs,
            )
            return output

        finally:
            end_time_ns = time.time_ns()
            end_perf_ns = time.perf_counter_ns()

            timing_dir = os.path.join(
                os.environ["DUET_RUN_DIR"],
                "request_timings",
            )
            os.makedirs(timing_dir, exist_ok=True)

            global_step = None
            response_tokens = None

            if output is not None:
                global_step = output.extra_fields.get("global_steps")
                response_tokens = len(output.token_ids)

            sampling_params = kwargs.get("sampling_params", {})

            record = {
                "global_step": global_step,
                "request_id": request_id,
                "worker_id": os.getpid(),
                "start_time_ns": start_time_ns,
                "end_time_ns": end_time_ns,
                "duration_s": (end_perf_ns - start_perf_ns) / 1e9,
                "prompt_tokens": len(kwargs.get("prompt_ids", [])),
                "response_tokens": response_tokens,
                "temperature": sampling_params.get("temperature"),
                "top_p": sampling_params.get("top_p"),
                "top_k": sampling_params.get("top_k"),
            }

            path = os.path.join(
                timing_dir,
                f"worker_{os.getpid()}.jsonl",
            )

            with open(path, "a") as f:
                f.write(json.dumps(record) + "\n")

    LLMServerClient.generate = wrapped
    LLMServerClient._duet_request_timing_patched = True

    print("[duet] LLMServerClient.generate timing patched")


def patch_vllm_stat_logger(module):
    # from vllm.v1.engine.async_llm import AsyncLLM
    from duet.training.vllm_metrics_logger import DuetVLLMLogger

    AsyncLLM = module.AsyncLLM

    if getattr(
        AsyncLLM,
        "_duet_stat_logger_patched",
        False,
    ):
        return

    original_init = AsyncLLM.__init__

    def patched_init(self, *args, **kwargs):
        stat_loggers = list(kwargs.get("stat_loggers") or [])

        if DuetVLLMLogger not in stat_loggers:
            stat_loggers.append(DuetVLLMLogger)

        kwargs["stat_loggers"] = stat_loggers

        return original_init(
            self,
            *args,
            **kwargs,
        )

    AsyncLLM.__init__ = patched_init
    AsyncLLM._duet_stat_logger_patched = True

    print(
        "[duet] AsyncLLM custom stat logger patched",
        flush=True,
    )


# =================================================================================
# Core monkey patching function.
# =================================================================================


def patch_verl_for_profiling():
    """Install lazy hooks for VERL profiling patches after the relevant
    modules are imported, avoiding premature CUDA initialization."""

    targets = {
        "verl.experimental.agent_loop.agent_loop": apply,
        "verl.trainer.ppo.v1.agent_loop_tq": patch_agent_loop_worker_timing,
        "verl.workers.rollout.llm_server": patch_llm_request_timing,
        "vllm.v1.engine.async_llm": patch_vllm_stat_logger,
    }
    patched = set()

    # already imported? patch immediately
    for module, patch_func in targets.items():
        if module in sys.modules:
            patch_func(sys.modules[module])
            patched.add(module)

    if len(patched) == len(targets):
        return

    # otherwise defer until it's imported (after CUDA is set up)
    import importlib.util
    from importlib.abc import MetaPathFinder

    class _LazyPatchFinder(MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name not in targets:
                return None

            sys.meta_path.remove(self)  # avoid recursion
            spec = importlib.util.find_spec(name)
            sys.meta_path.insert(0, self)
            if spec is None:
                return None
            real_exec = spec.loader.exec_module

            def exec_module(module):
                real_exec(module)

                targets[name](module)

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _LazyPatchFinder())
    print("[duet] lazy profiling patches armed")
