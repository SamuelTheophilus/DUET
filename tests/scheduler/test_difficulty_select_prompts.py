from tests.scheduler.helpers import (
    add_reward_history,
    make_prompt_pool,
    snapshot_reward_history,
)


def test_selects_unseen_prompts_without_skipping(difficulty_scheduler):
    pool = make_prompt_pool(10)

    selected = difficulty_scheduler.select_prompts(pool, batch_size=5)

    assert len(selected) == 5
    assert list(selected["prompt_id"]) == [f"train_{i}" for i in range(5)]


def test_can_return_partial_batch_when_some_prompts_are_skipped(
    difficulty_scheduler,
    monkeypatch,
):
    """
    Checkpoint from Objective 1 in the cookbook
    """
    pool = make_prompt_pool(50)
    add_reward_history(
        difficulty_scheduler,
        num_prompts=50,
        reward=1.0,
        history_length=20,
    )
    before = snapshot_reward_history(difficulty_scheduler)

    calls = 0

    def skip_first_40(prompt, base_prob):
        nonlocal calls
        calls += 1
        return calls <= 40

    monkeypatch.setattr(
        difficulty_scheduler,
        "_skip_easy_prompts",
        skip_first_40,
    )
    monkeypatch.setattr(
        difficulty_scheduler,
        "_skip_hard_prompts",
        lambda prompt, base_prob: False,
    )

    selected = difficulty_scheduler.select_prompts(pool, batch_size=50)

    assert calls == 50
    assert len(selected) == 10
    assert list(selected["prompt_id"]) == [f"train_{i}" for i in range(40, 50)]
    assert snapshot_reward_history(difficulty_scheduler) == before


def test_returns_empty_batch_when_every_prompt_is_skipped(
    difficulty_scheduler,
    monkeypatch,
):
    pool = make_prompt_pool(50)
    add_reward_history(
        difficulty_scheduler,
        num_prompts=50,
        reward=1.0,
        history_length=20,
    )
    before = snapshot_reward_history(difficulty_scheduler)

    monkeypatch.setattr(
        difficulty_scheduler,
        "_skip_easy_prompts",
        lambda prompt, base_prob: True,
    )
    monkeypatch.setattr(
        difficulty_scheduler,
        "_skip_hard_prompts",
        lambda prompt, base_prob: False,
    )

    selected = difficulty_scheduler.select_prompts(pool, batch_size=50)

    assert len(selected) == 0
    assert snapshot_reward_history(difficulty_scheduler) == before


def test_does_not_skip_unseen_prompts_after_skipping_seen_prompts(
    difficulty_scheduler,
    monkeypatch,
):
    pool = make_prompt_pool(10)

    add_reward_history(
        difficulty_scheduler,
        num_prompts=5,
        reward=1.0,
        history_length=20,
    )

    monkeypatch.setattr(
        difficulty_scheduler,
        "_skip_easy_prompts",
        lambda prompt, base_prob: True,
    )
    monkeypatch.setattr(
        difficulty_scheduler,
        "_skip_hard_prompts",
        lambda prompt, base_prob: False,
    )

    selected = difficulty_scheduler.select_prompts(pool, batch_size=5)

    assert len(selected) == 5
    assert list(selected["prompt_id"]) == [f"train_{i}" for i in range(5, 10)]
