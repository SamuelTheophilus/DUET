import json
import os
import torch
from pathlib import Path
from tensordict import TensorDict
from abc import ABC, abstractmethod
from collections import defaultdict
from .utils import SelectPromptStats

import logging

logger = logging.getLogger(__name__)


class Scheduler(ABC):
    """
    This is an abstract interface for prompt schedulling strategies (NoOP, LengthOnly, DifficultyOnly, Joint)

    A Scheduler controls how a pool of prompts will be selected, reorderd, and transformed into batches before
    being passed to the rollout.
    The schedulling pipeline: prompt_pool -> select_prompts() -> prompts -> reorder() -> reordered_prompts -> pack() -> List[Batch]
    Implementations of each scheduler's core functions (select_prompts, reorder, and pack) may differ.
    Implementations of each scheduler never mutates the original pool.
    """

    last_stats: SelectPromptStats | None = None
    name: str = ""

    @abstractmethod
    def select_prompts(self, pool, batch_size, **kwargs):
        """
        Selects a list of prompts from the general pool for the next training step.

        Args:
             pool: list[str]
                 Candidate prompts that are available for selection.
             batch_size: int
                 Number of prompts to be selected.

        Returns:
             A list of selected prompts from the entire pool.
        """
        return pool[:batch_size]

    @abstractmethod
    def reorder(self, prompts):
        """
        Reorder selected prompts before batching.
        This may include arranging by difficulty, length, or both.

        Args:
             prompts:
                    Prompts from the `select_prompts` method

        Returns:
              An ordered list of prompts.
        """
        return prompts

    @abstractmethod
    def pack(self, prompts, pack_size: int) -> list:
        """
        Organises the re-ordered prompts into training batches

        Args:
            prompts:
                   Prompts from the `reorder` method

        Returns:
            A list of training batches.
        """
        return [prompts[i : i + pack_size] for i in range(0, len(prompts), pack_size)]

    def update(self, prompt_id, step, observed_reward):
        """
        Updates the streak history of every prompt at each step.
        Used primarily by the difficulty only scheduler
        """
        pass

    def reset_stats(self):
        pass

    @abstractmethod
    def assign(
        self, buckets: list[TensorDict], num_workers: int, schedulling_type: str = "lpt"
    ) -> TensorDict:
        return torch.cat(buckets, dim=0)


class NoOpScheduler(Scheduler):
    name: str = "noops"

    def __init__(self) -> None:
        self.mini_batch_size = 8  # This is passed in the config of the user.
        self.last_stats: SelectPromptStats | None = SelectPromptStats()
        self.data: dict  = defaultdict(list)

    def select_prompts(self, pool, batch_size, **kwargs):
        print("[NoOpScheduler]: selecting prompts")
        return super().select_prompts(pool, batch_size, **kwargs)

    def reorder(self, prompts):
        print("[NoOpScheduler]: reordering prompts")
        return super().reorder(prompts)

    def pack(self, prompts, pack_size: int):
        print("[NoOpScheduler]: packing prompts")
        return super().pack(prompts, pack_size)

    # def update(self, prompt_id, step, observed_reward):
    #     super().update(prompt_id, step, observed_reward)
    #

    def update(self, prompt_id, step, observed_reward):
            """
            Records the step and reward for each prompt after rewards have been generated
            (i.e in the reward generation step).
            """
            self.data[prompt_id].append((step, observed_reward))




    def reset_stats(self):
        return super().reset_stats()

    def assign(self, buckets, num_workers, schedulling_type: str = "lpt"):
        return super().assign(buckets, num_workers, schedulling_type)


    def save_history(self, step: int) -> None:

        streak_dir_str = os.environ.get("PROMPT_STREAK_DIR")
        if not streak_dir_str:
            return

        streak_dir = Path(streak_dir_str) 
        if not streak_dir.is_dir():
            return 

        with (streak_dir/ f"{step}.json").open("w") as f:
            json.dump(self.data, f)



class LengthOnlyScheduler(Scheduler):
    name: str = "length_only"

    def __init__(self):
        self.last_stats: SelectPromptStats | None = SelectPromptStats()

    def select_prompts(self, pool, batch_size, **kwargs):
        print(f"[{self.name}] filtering prompts. (Fall through).")
        return super().select_prompts(pool, batch_size, **kwargs)

    def reorder(self, prompts: TensorDict) -> TensorDict:
        """
        Reorder the prompts based on the predicted lengths.
        Reordering returns a list of 'mini-batches', where each batch in this list will complete
        an entire step (rollout, reward-calc, advantage-calc, training) in the GRPO step.
        This introduces some off-policy-ness and uses KL-penalty adjustment.
        """
        print(f"[{self.name}] reordering prompts")
        indices = sorted(
            # Sorting this way returns the position of the indicies which can be used to "reshuffle"
            # the prompts in this manner: prompts[indices]. Doing it this way to preserve the output as a
            # TensorDict and not a list.
            range(len(prompts)),
            key=lambda i: prompts[i]["extra_info"].get("predicted_response_length", 0),
        )

        return prompts[indices]

    def pack(self, prompts, pack_size):
        print(f"[{self.name}] packing prompts into mini-batches of size {pack_size}")
        return super().pack(prompts, pack_size)

    def update(self, prompt_id, step, observed_reward):
        pass

    def assign(
        self, buckets: list[TensorDict], num_workers: int, schedulling_type: str = "lpt"
    ):
        """
        Assigning buckets to workers using LPT/Round-robin schedulling.
        """

        weight_per_worker = [0 for _ in range(num_workers)]

        bucket_weights = [(bucket, self._bucket_weight(bucket)) for bucket in buckets]

        bucket_weights.sort(key=lambda x: x[1], reverse=True)

        assigned_buckets = []

        if schedulling_type == "lpt":
            for bucket, weight in bucket_weights:
                min_idx = weight_per_worker.index(min(weight_per_worker))

                bucket["assign_to_worker"] = torch.full(
                    bucket.batch_size, min_idx, dtype=torch.long, device=bucket.device
                )
                assigned_buckets.append(bucket)

                weight_per_worker[min_idx] = weight_per_worker[min_idx] + weight
        else:
            # Default round robin schedulling
            for idx, (bucket, _) in enumerate(bucket_weights):
                worker_id = idx % num_workers
                bucket["assign_to_worker"] = torch.full(
                    bucket.batch_size, worker_id, dtype=torch.long, device=bucket.device
                )
                assigned_buckets.append(bucket)

        return torch.cat(assigned_buckets, dim=0)

    def _bucket_weight(self, bucket: TensorDict) -> int:

        return sum(
            prompt["extra_info"].get("predicted_response_length", 0)
            for prompt in bucket
        )


class DifficultyOnlyScheduler(Scheduler):
    name: str = "difficulty_only"

    def __init__(self):
        super().__init__()
        self.data: dict = defaultdict(list)
        # These should prolly be loaded from config.
        self.EASY_SKIP_PROBABILITY_THRESHOLD = 0.98
        self.HARD_SKIP_PROBABILITY_THRESHOLD = 0.11
        self.BASELINE_PROBABILITY = 0.01

        self.last_stats: SelectPromptStats = SelectPromptStats()

    def reset_stats(self):
        self.last_stats = SelectPromptStats()

    def select_prompts(
        self,
        # -> Assuming this is the entire pool and will be reduced by batch_size
        pool: TensorDict,
        batch_size: int,
        **kwargs,
    ) -> TensorDict:
        """
        Removes all prompts that have been predicted to be zero variance (i.e All their rollout responses will return 0/1)
        """
        print(f"[{self.name}] filtering prompts")

        selected_indices = []

        skip_easy_count = 0
        skip_hard_count = 0

        easy_base_prob = kwargs.get("easy_base_prob", 0.75)
        hard_base_prob = kwargs.get("hard_base_prob", 0.5)

        prompt_pt = 0
        while len(selected_indices) < batch_size and prompt_pt < len(pool):
            ## Collect prompts into the selected prompts from the larger pool until the batch size is reached.
            prompt = pool[prompt_pt]
            prompt_id = prompt["prompt_id"]

            idx = prompt_pt
            prompt_pt += 1

            history = self.data[prompt_id]

            if len(history) == 0:  # First observation of the prompt
                selected_indices.append(idx)
                continue

            if self._skip_easy_prompts(prompt, base_prob=easy_base_prob):
                skip_easy_count += 1
                continue

            if self._skip_hard_prompts(prompt, base_prob=hard_base_prob):
                skip_hard_count += 1
                continue

            selected_indices.append(idx)

        print(f"INFO:[Selected Prompts]-> Easy prompts skipped {skip_easy_count}")
        print(f"INFO:[Selected Prompts]-> Hard prompts skipped {skip_hard_count}")

        if not selected_indices:
            selected_indices = [pt for pt in range(batch_size)]

        # Record the last stats
        self.last_stats.selected += len(selected_indices)
        self.last_stats.skipped_easy += skip_easy_count
        self.last_stats.skipped_hard += skip_hard_count
        self.last_stats.pool_size += len(pool)
        self.last_stats.considered += prompt_pt

        return pool[selected_indices]

    def reorder(self, prompts):
        print(f"[{self.name}] reordering prompts (Fall through).")
        return super().reorder(prompts)

    def pack(self, prompts, pack_size):
        print(f"[{self.name}] packing prompts")

        return super().pack(prompts, pack_size)

    def _skip_easy_prompts(
        self,
        prompt: dict,
        base_prob: float,
    ) -> bool:
        """
        Skips prompts which are relatively easy to answer.
        Compares the most recent reward in against the threshold for demarcating a prompt as easy.
        And returns the appropriate value for when the reward is greater, equal to or lesser than the threshold.

        Args:
            prompt: dict

        Returns:
            skip_prompt: bool
        """
        prompt_id = prompt["prompt_id"]
        reward_history = self.data[prompt_id]

        count = 0

        for _, reward in reversed(reward_history):
            if reward < self.EASY_SKIP_PROBABILITY_THRESHOLD:
                break

            count += 1

        skip_probability = 1 - max(
            base_prob**count,
            self.BASELINE_PROBABILITY,
        )

        return torch.rand(()).item() < skip_probability

    def _skip_hard_prompts(
        self,
        prompt: dict,
        base_prob: float,
    ) -> bool:
        """
        Skips prompts which are relatively difficult to answer.
        Compares the most recent reward in against the threshold for demarcating a prompt as difficult.
        And returns the appropriate value for when the reward is greater, equal to or lesser than the threshold.

        Args:
            prompt: dict

        Returns:
            skip_prompt: bool
        """

        prompt_id = prompt["prompt_id"]
        reward_history = self.data[prompt_id]

        latest_reward = reward_history[-1][1]

        # The current result is not hard.
        if latest_reward > self.HARD_SKIP_PROBABILITY_THRESHOLD:
            return False

        count = 0

        for _, reward in reversed(reward_history):
            if reward > self.HARD_SKIP_PROBABILITY_THRESHOLD:
                break

            count += 1

        skip_probability = 1 - max(
            base_prob**count,
            self.BASELINE_PROBABILITY,
        )

        return torch.rand(()).item() < skip_probability

    def update(self, prompt_id, step, observed_reward):
        """
        Records the step and reward for each prompt after rewards have been generated
        (i.e in the reward generation step).
        """
        self.data[prompt_id].append((step, observed_reward))


    def save_history(self, step: int) -> None:

        streak_dir_str = os.environ.get("PROMPT_STREAK_DIR")
        if not streak_dir_str:
            return

        streak_dir = Path(streak_dir_str) 
        if not streak_dir.is_dir():
            return 

        with (streak_dir/ f"{step}.json").open("w") as f:
            json.dump(self.data, f)

    def assign(
        self, buckets: list[TensorDict], num_workers: int, schedulling_type: str = "lpt"
    ) -> TensorDict:
        return torch.cat(buckets, dim=0)
        # return super().assign(buckets, num_workers)


class JointScheduler(Scheduler):
    name: str = "joint"

    def __init__(self) -> None:
        super().__init__()
        self.difficulty_scheduler = DifficultyOnlyScheduler()
        self.length_scheduler = LengthOnlyScheduler()

    def select_prompts(self, pool, batch_size, **kwargs):
        return self.difficulty_scheduler.select_prompts(pool, batch_size, **kwargs)

    def reorder(self, prompts):
        return self.length_scheduler.reorder(prompts)

    def pack(self, prompts, pack_size):
        return super().pack(prompts, pack_size)

    def update(self, prompt_id, step, observed_reward):
        self.difficulty_scheduler.update(prompt_id, step, observed_reward)

    def assign(
        self, buckets: list[TensorDict], num_workers: int, schedulling_type: str = "lpt"
    ) -> TensorDict:
        return self.length_scheduler.assign(buckets, num_workers)


SCHEDULERS = {
    "noops": NoOpScheduler,
    "length_only": LengthOnlyScheduler,
    "difficulty_only": DifficultyOnlyScheduler,
    "joint": JointScheduler,
}


def get_scheduler(key: str):
    return SCHEDULERS.get(key, None)
