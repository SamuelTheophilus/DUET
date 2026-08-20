from tensordict import TensorDict

from duet.scheduler.core import DifficultyOnlyScheduler


def make_prompt_pool(num_prompts: int) -> TensorDict:
    return TensorDict(
        {"prompt_id": [f"train_{i}" for i in range(num_prompts)]},
        batch_size=[num_prompts],
    )


def add_reward_history(
    scheduler: DifficultyOnlyScheduler,
    num_prompts: int,
    *,
    reward: float,
    history_length: int,
) -> None:
    for i in range(num_prompts):
        prompt_id = f"train_{i}"
        for step in range(history_length):
            scheduler.update(prompt_id, step, reward)


def snapshot_reward_history(scheduler: DifficultyOnlyScheduler):
    return {
        prompt_id: list(history)
        for prompt_id, history in scheduler.data.items()
    }
