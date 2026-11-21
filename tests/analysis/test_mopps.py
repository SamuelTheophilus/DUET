"""Regression tests for DUET's MoPPS offline baseline."""

import math
import pytest

from duet.analysis.mopps import (
    MoPPSPosterior,
    chronological_replay,
)


def test_initial_prior():
    model = MoPPSPosterior()
    assert model.get_params("A") == (1.0, 1.0)
    assert model.posterior_mean("A") == 0.5


def test_undiscounted_updates():
    model = MoPPSPosterior(discount=1.0)

    model.update("A", 4, 1)
    assert model.get_params("A") == (5.0, 2.0)

    model.update("A", 2, 3)
    assert model.get_params("A") == (7.0, 5.0)


def test_discounted_updates():
    model = MoPPSPosterior(discount=0.5)

    model.update("A", 4, 1)
    model.update("A", 2, 3)

    assert model.get_params("A") == (5.0, 4.5)


def test_independent_prompts():
    model = MoPPSPosterior()

    model.update("A", 5, 0)
    model.update("B", 0, 5)

    assert model.get_params("A") == (6.0, 1.0)
    assert model.get_params("B") == (1.0, 6.0)
    assert model.get_params("C") == (1.0, 1.0)


def test_reset():
    model = MoPPSPosterior()
    model.update("A", 4, 1)

    model.reset()

    assert model.get_params("A") == (1.0, 1.0)


def test_dead_probability_uniform_prior():
    result = MoPPSPosterior.dead_probability(1, 1, 5)
    assert math.isclose(result, 1 / 3, abs_tol=1e-12)


def test_dead_probability_symmetry():
    left = MoPPSPosterior.dead_probability(5, 2, 5)
    right = MoPPSPosterior.dead_probability(2, 5, 5)

    assert math.isclose(left, right, abs_tol=1e-12)


def test_dead_probability_known_value():
    # Beta(2, 2), n=2:
    # P(all pass) = 2*3/(4*5) = 0.3
    # P(all fail) = 2*3/(4*5) = 0.3
    result = MoPPSPosterior.dead_probability(2, 2, 2)

    assert math.isclose(result, 0.6, abs_tol=1e-12)


def test_thompson_seed_reproducibility():
    a = MoPPSPosterior(seed=42)
    b = MoPPSPosterior(seed=42)

    assert a.thompson_sample("A") == b.thompson_sample("A")


def test_chronological_replay_pre_observation():
    rows = [
        {"prompt_id": "A", "step": 2,
         "successes": 2, "failures": 3},
        {"prompt_id": "A", "step": 1,
         "successes": 4, "failures": 1},
    ]

    output = chronological_replay(rows, discount=0.5)

    assert len(output) == 2
    assert output[0]["step"] == 1
    assert output[0]["alpha_pre"] == 1.0
    assert output[0]["beta_pre"] == 1.0

    assert output[1]["step"] == 2
    assert output[1]["alpha_pre"] == 5.0
    assert output[1]["beta_pre"] == 2.0


def test_duplicate_prompt_step_rejected():
    row = {
        "prompt_id": "A",
        "step": 1,
        "successes": 4,
        "failures": 1,
    }

    with pytest.raises(ValueError, match="Duplicate"):
        chronological_replay([row, row])


@pytest.mark.parametrize(
    "discount", [0.0, -0.1, 1.1]
)
def test_invalid_discount(discount):
    with pytest.raises(ValueError):
        MoPPSPosterior(discount=discount)


@pytest.mark.parametrize(
    "successes,failures",
    [
        (-1, 3),
        (2, -1),
        (0, 0),
        (1.5, 2),
    ],
)
def test_invalid_reward_counts(successes, failures):
    model = MoPPSPosterior()

    with pytest.raises(ValueError):
        model.update("A", successes, failures)


def test_historical_posterior_fixture():
    # Known 5-rollout sequence from synthetic
    # Week 8 regression validation.
    model = MoPPSPosterior(discount=0.5)

    model.update("A", 4, 1)
    model.update("A", 2, 3)

    alpha, beta = model.get_params("A")

    assert alpha == 5.0
    assert beta == 4.5

    assert math.isclose(
        model.posterior_mean("A"),
        5.0 / 9.5,
        abs_tol=1e-12,
    )
