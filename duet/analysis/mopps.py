"""
DUET MoPPS-inspired offline analysis.

Implements a Beta posterior for each prompt, evidence-discounted
updates, Thompson sampling, and posterior-predictive dead-group
probabilities.

Discount convention:
    alpha_new = alpha0 + discount * (alpha_old - alpha0) + successes
    beta_new  = beta0  + discount * (beta_old  - beta0)  + failures

This module reproduces the Week 8 offline evaluation convention.
It is not a complete reproduction of the published MoPPS
training system.

Standard-library only.
"""

import math
import random


class MoPPSPosterior:
    """Independent Beta posterior for each prompt."""

    def __init__(self, discount=1.0, seed=42):
        if not (0 < discount <= 1):
            raise ValueError("discount must be in (0, 1].")

        self.discount = float(discount)
        self.alpha0 = 1.0
        self.beta0 = 1.0
        self.state = {}
        self.rng = random.Random(seed)

    def get_params(self, prompt_id):
        return self.state.get(
            prompt_id,
            (self.alpha0, self.beta0),
        )

    def posterior_mean(self, prompt_id):
        alpha, beta = self.get_params(prompt_id)
        return alpha / (alpha + beta)

    def thompson_sample(self, prompt_id):
        alpha, beta = self.get_params(prompt_id)
        return self.rng.betavariate(alpha, beta)

    def update(self, prompt_id, successes, failures):
        if (
            isinstance(successes, bool)
            or isinstance(failures, bool)
            or int(successes) != successes
            or int(failures) != failures
        ):
            raise ValueError("Counts must be integers.")

        successes = int(successes)
        failures = int(failures)

        if successes < 0 or failures < 0:
            raise ValueError("Counts must be nonnegative.")

        if successes + failures == 0:
            raise ValueError("At least one observation is required.")

        alpha, beta = self.get_params(prompt_id)

        new_alpha = (
            self.alpha0
            + self.discount * (alpha - self.alpha0)
            + successes
        )

        new_beta = (
            self.beta0
            + self.discount * (beta - self.beta0)
            + failures
        )

        self.state[prompt_id] = (new_alpha, new_beta)

    def reset(self):
        self.state.clear()

    @staticmethod
    def dead_probability(alpha, beta, n=5):
        """
        Posterior-predictive probability of n all-pass
        or n all-fail Bernoulli outcomes.

        Conditional independence given latent pass rate
        is an explicit modeling assumption.
        """
        if alpha <= 0 or beta <= 0:
            raise ValueError("Beta parameters must be positive.")

        if not isinstance(n, int) or isinstance(n, bool) or n < 2:
            raise ValueError("n must be an integer >= 2.")

        def log_beta(a, b):
            return (
                math.lgamma(a)
                + math.lgamma(b)
                - math.lgamma(a + b)
            )

        base = log_beta(alpha, beta)

        all_pass = math.exp(log_beta(alpha + n, beta) - base)
        all_fail = math.exp(log_beta(alpha, beta + n) - base)

        return all_pass + all_fail


def chronological_replay(rows, discount=1.0, seed=42):
    """
    Replay observations without target leakage.

    Input:
        Iterable of dictionaries containing:
        prompt_id, step, successes, failures.

    Output:
        List of dictionaries with predictions produced
        BEFORE each current observation is incorporated.

    Input rows are sorted by (step, prompt_id).
    Duplicate prompt-step observations are rejected.
    """

    model = MoPPSPosterior(discount=discount, seed=seed)

    observations = sorted(
        rows,
        key=lambda row: (row["step"], str(row["prompt_id"])),
    )

    seen = set()
    output = []
    previous_step = {}

    for row in observations:
        prompt_id = row["prompt_id"]
        step = int(row["step"])

        key = (prompt_id, step)

        if key in seen:
            raise ValueError(f"Duplicate observation: {key}")

        seen.add(key)

        if (
            prompt_id in previous_step
            and step <= previous_step[prompt_id]
        ):
            raise ValueError("Non-increasing prompt step.")

        previous_step[prompt_id] = step

        alpha, beta = model.get_params(prompt_id)

        output.append({
            "prompt_id": prompt_id,
            "step": step,
            "alpha_pre": alpha,
            "beta_pre": beta,
            "posterior_mean": model.posterior_mean(prompt_id),
            "thompson_sample": model.thompson_sample(prompt_id),
            "mopps_dead_prob": model.dead_probability(
                alpha, beta, n=5
            ),
        })

        model.update(
            prompt_id,
            row["successes"],
            row["failures"],
        )

    return output
