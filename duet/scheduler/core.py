import torch
from abc import ABC, abstractmethod
from collections import defaultdict

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
    def reorder(self, prompts, group_size_n):
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
    def pack(self, prompts) -> list:
        """
        Organises the re-ordered prompts into training batches

        Args:
            prompts:
                   Prompts from the `reorder` method

        Returns:
            A list of training batches.
        """
        pass


class NoOpScheduler(Scheduler):
    def __init__(self) -> None:
        self.mini_batch_size = 8  # This is passed in the config of the user.

    def select_prompts(self, pool, batch_size, **kwargs):
        return super().select_prompts(pool, batch_size, **kwargs)

    def reorder(self, prompts, group_size_n):
        return super().reorder(prompts, group_size_n)

    def pack(self, prompts):
        return [
            prompts[i : i + self.mini_batch_size]
            for i in range(len(prompts), self.mini_batch_size)
        ]


class LengthOnlyScheduler(Scheduler):
    def select_prompts(self, pool, batch_size, **kwargs):
        return super().select_prompts(pool, batch_size, **kwargs)

    def reorder(
        self,
        prompts: list[dict],  # -> This is an assumption and maybe subject to modification
        group_size_n: int = 2,
    ) -> list[list[dict]]:
        """
        Reorder the prompts based on the predicted lengths.
        Reordering returns a list of 'mini-batches', where each batch in this list will complete
        an entire step (rollout, reward-calc, advantage-calc, training) in the GRPO step.
        This introduces some off-policy-ness and uses KL-penalty adjustment.
        """
        # Over here I'm assuming the prompts is a list[dict].
        # The data structure may be subject to change and this function will need to be updated
        # if/when the data structure changes.
        sorted_prompts_by_predicted_length: list[dict] = sorted(
            prompts, key=lambda x: x["predicted_response_length"]
        )
        # Group the sorted prompts in to mini batches by `group_size_n`
        return [
            sorted_prompts_by_predicted_length[i : i + group_size_n]
            for i in range(len(sorted_prompts_by_predicted_length), group_size_n)
        ]

    def pack(self, prompts):
        return super().pack(prompts)


class DifficultyOnlyScheduler(Scheduler):

    def __init__(self):
        super().__init__()
        self.data: dict = defaultdict(list)
        self.EASY_SKIP_PROBABILITY_THRESHOLD = 0.98 # These should prolly be loaded from config.
        self.HARD_SKIP_PROBABILITY_THRESHOLD = 0.11 # These should prolly be loaded from config.
        self.BASELINE_PROBABILITY = 0.01

    def select_prompts(
        self,
        pool: list[dict], # -> Assuming this is the entire pool and will be reduced by batch_size
        batch_size: int,
        **kwargs,
    ) -> list[dict]:
        """
        Removes all prompts that have been predicted to be zero variance (i.e All their rollout responses will return 0/1)
        """

        selected_prompts = []

        skip_easy_count = 0
        skip_hard_count = 0

        easy_base_prob = kwargs.get("easy_base_prob", 0.75)
        hard_base_prob = kwargs.get("hard_base_prob", 0.5)


        for prompt in pool:
            # Assumption here is the prompt is dictionary containing prompt_id, epoch, and predicted_reward.
            prompt_id = prompt["prompt_id"]
            epoch = prompt["epoch"]
            predicted_reward = prompt["predicted_reward"]



            history = self.data[prompt_id]
            is_first_observation = len(history) == 0

            history.append((epoch, predicted_reward))

            if is_first_observation:
                selected_prompts.append(prompt)
                continue

            if self._skip_easy_prompts(prompt, base_prob=easy_base_prob):
                skip_easy_count += 1
                continue

            if self._skip_hard_prompts(prompt, base_prob=hard_base_prob):
                skip_hard_count += 1
                continue

            selected_prompts.append(prompt)

        print(f"INFO:[Selected Prompts] -> Easy prompts skipped {skip_easy_count}")
        print(f"INFO:[Selected Prompts]-> Easy prompts skipped {skip_hard_count}")
        return selected_prompts[:batch_size]


    def reorder(self, prompts, group_size_n):
        return super().reorder(prompts, group_size_n)


    def pack(self, prompts):
        return super().pack(prompts)


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
    

class JointScheduler(Scheduler):
    def __init__(self) -> None:
        super().__init__()
        self.difficulty_scheduler = DifficultyOnlyScheduler()
        self.length_scheduler = LengthOnlyScheduler()

    def select_prompts(self, pool, batch_size, **kwargs):
        return self.difficulty_scheduler.select_prompts(pool, batch_size, **kwargs)

    def reorder(self, prompts, group_size_n):
        return self.length_scheduler.reorder(prompts, group_size_n)

    def pack(self, prompts):
        return super().pack(prompts)
