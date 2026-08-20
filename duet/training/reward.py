import logging
import re

logger = logging.getLogger(__name__)

_DEBUG_COUNT = 0

def custom_reward_function(data_source, solution_str, ground_truth, extra_info=None):
    matches = re.findall(r"\\boxed\{([^}]*)\}", solution_str)

    global _DEBUG_COUNT
    if _DEBUG_COUNT < 10:
        print("[custom_reward_function] Computing the custom reward..", flush=True)
        print(
            "[CUSTOM REWARD]",
            repr(solution_str[-200:]),
            "GT:",
            repr(ground_truth),
            flush=True

        )
        print("MATCHES:", matches, flush=True)
        _DEBUG_COUNT += 1

    if not matches:
        return 0.0

    def is_number(s):
        return bool(re.fullmatch(r"-?\d+(\.\d+)?", s))

    sol_val = matches[-1].strip().replace(",", "")
    if not is_number(sol_val):  # To capture values in \boxed{} when they're not valid integers
        return 0.0

    gt = ground_truth.strip()
    if float(sol_val) == float(gt):
        return 1.0

    return 0.0 


