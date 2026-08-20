from verl.trainer.ppo.v1  import  PPOTrainerSync
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

    def _add_batch_to_generate(self):
        print("=" * 80)
        print("[DUET] DuetPPOTrainerSync._add_batch_to_generate called ")
        print(f"[DUET] Scheduler selected. {self.scheduler}")
        print("=" * 80)

        batch = self._next_train_batch()
        if self.scheduler:
            print("[DUET] calling schedulers")
            batch = self.scheduler.select_prompts(batch, batch_size=len(batch))
            batch = self.scheduler.reorder(batch)
        return self._submit_batch_to_rollout(batch)

    def _record_reward_history(self, batch) -> None:

        data = tq.kv_batch_get(
            keys=batch.keys,
            partition_id=batch.partition_id,
            select_fields=["prompt_id", "rm_scores"]
        )

        print("[DUET] recording reward streak for prompt_ids.")
        print("[DUET] data type:", type(data))
        print("[DUET] prompt_id type:", type(data["prompt_id"]))
        print("[DUET] prompt_ids:", data["prompt_id"][:5])
        print("[DUET] rm_scores type:", type(data["rm_scores"]))
        print("[DUET] rm_scores shape:", data["rm_scores"].shape)
        print()

        prompt_ids = list(data["prompt_id"])
        counts = Counter(prompt_ids)

        print("total:", len(prompt_ids))
        print("unique:", len(counts))
        print("count distribution:", Counter(counts.values()))

        for prompt_id, count in counts.items():
            if count != 5:
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

        rewards = rm_scores.sum(dim=1).tolist()

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



def install_duet_trainer():
    print(f"[DUET] replacing trainer: {TRAINER_REGISTRY['sync']} -> DuetPPOTrainerSync")
    TRAINER_REGISTRY["sync"] = DuetPPOTrainerSync

