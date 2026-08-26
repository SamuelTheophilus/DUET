import os
import sys


from ..scheduler.trainer import install_duet_trainer


trace_dir = "/data/test_traces"
def get_trace_dir():
    run_dir = os.environ["DUET_RUN_DIR"]
    return os.path.join(run_dir, "gen_times")



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


def patch_verl_for_profiling():
    """Install a lazy hook that patches AgentLoopManager._performance_metrics
    the moment verl's agent_loop module is imported -- NOT now, to avoid
    triggering transformer_engine's GPU probe before Ray assigns devices."""
    TARGET = "verl.experimental.agent_loop.agent_loop"

    # already imported? patch immediately
    if TARGET in sys.modules:
        apply(sys.modules[TARGET])
        return

    # otherwise defer until it's imported (after CUDA is set up)
    import importlib.util
    from importlib.abc import MetaPathFinder

    class _LazyPatchFinder(MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name != TARGET:
                return None
            sys.meta_path.remove(self)  # avoid recursion
            spec = importlib.util.find_spec(name)
            sys.meta_path.insert(0, self)
            if spec is None:
                return None
            real_exec = spec.loader.exec_module

            def exec_module(module):
                real_exec(module)
                apply(module)  # patch right after real import

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _LazyPatchFinder())
    print("[duet] lazy gen-time patch armed")
