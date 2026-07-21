import re


def custom_reward_function(data_source, solution_str, ground_truth, extra_info=None):
    matches = re.findall(r"\\boxed\{([^}]*)\}", solution_str)
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


