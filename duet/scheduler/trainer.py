from verl.utils.debug import marked_timer
from verl.trainer.ppo.v1  import  PPOTrainerSync
from verl.trainer.ppo.v1.utils import MetricsAggregator
from verl.trainer.ppo.v1.trainer_base import TRAINER_REGISTRY
import transfer_queue as tq
from transfer_queue import KVBatchMeta
from collections import defaultdict, Counter

from .core import Scheduler, get_scheduler


class DuetPPOTrainerSync(PPOTrainerSync):

    def __init__(self, config) -> None:
        super().__init__(config)
        _scheduler_cls = get_scheduler(self.config.trainer.duet.scheduler) 
        self.scheduler: Scheduler | None = _scheduler_cls() if _scheduler_cls else None
        self.selected_count: int | None = None
        self.underfill_policy: str = self.config.trainer.duet.underfill_policy or "short"

    def _add_batch_to_generate(self):
        print("=" * 80)
        print("[DUET] DuetPPOTrainerSync._add_batch_to_generate called ")
        print(f"[DUET] Scheduler selected. {self.scheduler}")
        print("=" * 80)
        # Resetting the selected_count to make sure it's not carried over from the previous call
        self.selected_count = None

        batch = self._next_train_batch()
        candidate_batch = len(batch)
        if self.scheduler:
            self.scheduler.reset_stats()
            
            print("[DUET] calling schedulers")
            batch = self.scheduler.select_prompts(batch, batch_size=candidate_batch)
            batch = self.scheduler.reorder(batch)
            self.selected_count = len(batch)
        else:
            self.selected_count = len(batch)

        if self.underfill_policy == "refill":
            # TODO: pull more candidates until the selected_count reaches data.train_batch_size
            raise NotImplementedError(
                "Refill policy hasn't been implemented yet"
            )
            self.selected_count = len(batch)

        self._submit_batch_to_rollout(batch)

    def _record_reward_history(self, batch) -> None:
        """
        Records the reward (group average) for each prompt in every step.
        This builds up the streak for each prompt during the training and is
        used to filter out easy/hard prompts later on.
        Args:
            batch: TensorDict of the most recent rollout data
        Returns: 
            None
        """

        real_keys = [
            key
            for key, tag in zip(batch.keys, batch.tags)
            if not tag.get("is_padding", False)
        ]

        data = tq.kv_batch_get(
            keys=real_keys,
            partition_id=batch.partition_id,
            select_fields=["prompt_id", "rm_scores"]
        )

        print("[DUET] recording reward streak for prompt_ids.")
        print("[DUET] data type:", type(data))
        print("[DUET] prompt_id type:", type(data["prompt_id"]))
        print("[DUET] prompt_ids:", data["prompt_id"][:5])
        print("[DUET] rm_scores type:", type(data["rm_scores"]))
        print("[DUET] rm_scores shape:", data["rm_scores"].shape)
        prompt_ids = list(data["prompt_id"])
        counts = Counter(prompt_ids)

        print("total:", len(prompt_ids))
        print("unique:", len(counts))
        print("count distribution:", Counter(counts.values()))

        for prompt_id, count in counts.items():
            if count != self.config.actor_rollout_ref.rollout.n:
                print("ANOMALY:", repr(prompt_id), count)

        grouped_rewards = self._compute_group_rewards(
            prompt_ids,
            data["rm_scores"],
        )

        for p_id, avg_reward in grouped_rewards.items():
            if self.scheduler:
                self.scheduler.update(
                    p_id,
                    self.global_steps,
                    avg_reward,
                )

    def _compute_group_rewards(
        self,
        prompt_ids,
        rm_scores,
    ) -> dict[str, float]:

        # rewards = rm_scores.sum(dim=1).tolist()

        rewards = [
                float(score.sum().item())
                for score in rm_scores.unbind()
        ]

        grouped = defaultdict(list)

        for p_id, reward in zip(prompt_ids, rewards):
            grouped[p_id].append(float(reward))

        return {
            p_id: sum(values) / len(values)
            for p_id, values in grouped.items()
        }

    def _step_once(self, metrics: dict, timing_raw: dict, sample_batch_size: int) -> KVBatchMeta:
        batch = super()._step_once(metrics, timing_raw, sample_batch_size)
        try:
            self._record_reward_history(batch)
        except Exception as error:
            print(f"[DUET] Error => {str(error)}")
            raise
        return batch

    def step(self, metrics: dict, timing_raw: dict) -> KVBatchMeta:
        normal_batch_size = self.config.data.train_batch_size

        assert normal_batch_size % self.parameter_sync_step == 0, (
            f"train_batch_size ({normal_batch_size}) must be divisible by "
            f"parameter_sync_step ({self.parameter_sync_step})"
        )
        # regular feed: stream one train batch worth of prompts for this step
        with marked_timer("feed", timing_raw):
            self._add_batch_to_generate()

        # Take a snapshot of the scheduler stats
        # if self.scheduler:
        scheduler_stats_snapshot = self.scheduler.last_stats if self.scheduler else None

        if self.selected_count is None:
            raise RuntimeError(
                "DUET failed to record selected_count during feed."
            )

        if self.underfill_policy == "short":
            effective_batch_size = self.selected_count
        elif self.underfill_policy == "refill":
            assert normal_batch_size == self.selected_count
            effective_batch_size = normal_batch_size
        else:
            raise ValueError(
                f"Unknown `underfill_policy` {self.underfill_policy}"
            )

        assert effective_batch_size % self.parameter_sync_step == 0, (
            f"effective_batch_size ({effective_batch_size}) must be divisible by "
            f"parameter_sync_step ({self.parameter_sync_step})"
        )
        sample_batch_size = effective_batch_size // self.parameter_sync_step


        metrics_aggregator = MetricsAggregator()
        combined_keys: list = []
        combined_tags: list = []
        combined_partition_id = "train"
        for trigger_idx in range(self.parameter_sync_step):
            self.local_trigger_step = trigger_idx

            iter_metrics: dict = {}
            batch = self._step_once(iter_metrics, timing_raw, sample_batch_size)
            sample_count = sum(not tag.get("is_padding", False) for tag in batch.tags)
            metrics_aggregator.add_step_metrics(iter_metrics, sample_count=sample_count)
            combined_keys.extend(batch.keys)
            combined_tags.extend(batch.tags)
            combined_partition_id = batch.partition_id

        metrics.update(metrics_aggregator.get_aggregated_metrics())
        if scheduler_stats_snapshot is not None:
            metrics.update({
                "duet/prompts_selected": scheduler_stats_snapshot.selected,
                "duet/prompts_skipped_easy": scheduler_stats_snapshot.skipped_easy,
                "duet/prompts_skipped_hard": scheduler_stats_snapshot.skipped_hard,
                # "duet/prompts_considered": scheduler_stats_snapshot.considered,

            })
        return KVBatchMeta(partition_id=combined_partition_id, keys=combined_keys, tags=combined_tags)









def install_duet_trainer():
    print(f"[DUET] replacing trainer: {TRAINER_REGISTRY['sync']} -> DuetPPOTrainerSync")
    TRAINER_REGISTRY["sync"] = DuetPPOTrainerSync

