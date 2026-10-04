import torch
import math
from verl.utils import tensordict_utils as tu
from verl.utils.debug import marked_timer
from verl.trainer.ppo.v1  import  PPOTrainerSync
from verl.trainer.ppo.v1.utils import MetricsAggregator
from verl.trainer.ppo.v1.trainer_base import TRAINER_REGISTRY
import transfer_queue as tq
from transfer_queue import KVBatchMeta
from collections import defaultdict, Counter

from .core import Scheduler, get_scheduler


default_schedulers = {
    "noops": 0,
    "difficulty_only": 1,
    "length_only": 2,
    "joint": 3
}


class DuetPPOTrainerSync(PPOTrainerSync):

    def __init__(self, config) -> None:
        super().__init__(config)
        # _scheduler_cls = get_scheduler(self.config.trainer.duet.scheduler) 
        # self.scheduler: Scheduler | None = _scheduler_cls() if _scheduler_cls else None

        self.candidate_buffer = None
        self.selected_count: int | None = None
        self.scheduler_tracker: int = 0
        self.scheduler: Scheduler | None = None
        self.underfill_policy: str = self.config.trainer.duet.underfill_policy or "short"
        self.num_workers: int = self.config.actor_rollout_ref.rollout.agent.num_workers or 8
        self.cumulative_total_tokens: int = 0
        self.cumulative_response_tokens: int = 0

        self.interleave_schedulers: list[Scheduler] = []
        for s in self.config.trainer.duet.scheduler_options:
            assert s in default_schedulers.keys(), f"Invalid Scheduler: Scheduler ({s})"

            _scheduler_cls = get_scheduler(s)
            if _scheduler_cls:
                valid_scheduler = _scheduler_cls()
                self.interleave_schedulers.append(valid_scheduler)
                print("=" * 80)
                print(f"Appending Scheduler: {valid_scheduler}")
                print("=" * 80)
                print()


        if self.scheduler and not self.underfill_policy:
            raise RuntimeError(
                "Scheduler available but no underfill_policy was selected"
            )

        if self.underfill_policy not in {"short" , "refill" }:
            raise RuntimeError(
                f"Invalid arg for underfill_policy passed: {self.underfill_policy}.\n",
                "Valid options: short, refill"
            )


    def scheduler_switch(self) -> Scheduler | None:
        if not (self.scheduler_tracker < len(self.interleave_schedulers)):
            # Snap back to the begining of the list if scheduler hits the end of the list
            self.scheduler_tracker = 0

        self.scheduler = self.interleave_schedulers[self.scheduler_tracker]
        self.scheduler_tracker += 1 

        print("=" * 80)
        print(f"[DUET] Scheduler selected. {self.scheduler}")
        print("=" * 80)



    def _add_batch_to_generate(self):
        """
        Overrides the original _add_batch_to_generate function.
        Injects scheduler calls (select_prompts and reorder) to 
        mutate the prompts before rollout.
        """
        #================================================
        # Core scheduler calls prior to rollout.
        #================================================
        self.selected_count = None
        target_size = self.config.data.train_batch_size

        self.scheduler_switch()

        if self.scheduler is None:
            batch = self._take_candidates(target_size)
            self.selected_count = len(batch)
            tu.assign_non_tensor_data(
                batch,
                "global_steps",
                self.global_steps,
            )
            self._submit_batch_to_rollout(batch)
            return


        self.scheduler.reset_stats()

        if self.underfill_policy == "short":
            batch = self._take_candidates(target_size)
            batch = self.scheduler.select_prompts(
                pool=batch,
                batch_size=target_size
            )
            batch = self.scheduler.reorder(batch)

            batch = self.scheduler.pack(batch, pack_size=64)
            batch = self.scheduler.assign(batch, self.num_workers)

            self.selected_count = len(batch)
            tu.assign_non_tensor_data(
                batch,
                "global_steps",
                self.global_steps,
            )
            self._submit_batch_to_rollout(batch)
            return

        if self.underfill_policy == "refill":
            batch = self._generate_refill_batch(target_size)
            batch = self.scheduler.reorder(batch)

            batch = self.scheduler.pack(batch, pack_size=64)
            batch = self.scheduler.assign(batch, self.num_workers)

            self.selected_count = len(batch)
            # ==========================================
            # This is to stamp all items in the batch in 
            # the current global steps
            # ==========================================
            tu.assign_non_tensor_data(
                batch,
                "global_steps",
                self.global_steps,
            )
            self._submit_batch_to_rollout(batch)
            return



    def _take_candidates(self, count: int): 
        total_prompt_buffer = []
        remaining = count

        while remaining > 0:

            if self.candidate_buffer is None or len(self.candidate_buffer) == 0:
                self.candidate_buffer = self._next_train_batch()

            take = min(remaining, len(self.candidate_buffer))

            total_prompt_buffer.append(self.candidate_buffer[:take])


            # =======================================================================
            # We don't throw away any prompt that has been read from the dataloader 
            # in self._next_train_batch(). Keep prompts in the self.candidate buffer
            # if they exceed the count needed and re-use in the next _take_candidates
            # call.
            # =======================================================================

            if take == len(self.candidate_buffer):
                self.candidate_buffer = None
            else:
                self.candidate_buffer = self.candidate_buffer[take:]

            remaining -= take

        if len(total_prompt_buffer) == 1:
            return total_prompt_buffer[0]

        return torch.cat(total_prompt_buffer, dim=0)

    def _prepend_candidate_buffer(self, batch):
        if len(batch) == 0:
            return
        if self.candidate_buffer is None or len(self.candidate_buffer) == 0:
            self.candidate_buffer = batch
        else:
            self.candidate_buffer = torch.cat([
                batch, self.candidate_buffer
            ], dim=0)


    def _generate_refill_batch(self, target_size: int):
        assert self.scheduler is not None, "Scheduler required when generating refill_batch"
        initial_pool_size: int = math.ceil(target_size * 1.3)
        refill_size: int = math.ceil(target_size * 0.3)



        candidates = self._take_candidates(initial_pool_size)
        selected_chunks = []
        selected_count = 0


        max_rounds = 10


        for _ in range(max_rounds):
            needed = target_size - selected_count

            if needed <= 0:
                break

            considered_before = self.scheduler.last_stats.considered
            selected = self.scheduler.select_prompts(
                pool=candidates,
                batch_size=needed
            )
            considered_after = self.scheduler.last_stats.considered
            considered_this_call = considered_after - considered_before


            if considered_this_call < len(candidates):
                unconsidered = candidates[considered_this_call:]
                self._prepend_candidate_buffer(unconsidered)


            if len(selected) > 0:
                selected_chunks.append(selected)
                selected_count += len(selected)

            if selected_count >= target_size:
                break

            candidates = self._take_candidates(
                max(refill_size, target_size - selected_count )
            )

        if selected_count < target_size:
           raise RuntimeError(
                f"DUET refill failed: selected {selected_count}/"
                f"{target_size} after {max_rounds} refill rounds"
            )
 
        if len(selected_chunks) == 1:
            return selected_chunks[0]

        return torch.cat(selected_chunks, dim=0)


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

        prompt_ids = list(data["prompt_id"])
        counts = Counter(prompt_ids)

        for prompt_id, count in counts.items():
            if count != self.config.actor_rollout_ref.rollout.n:
                print("ANOMALY:", repr(prompt_id), count)

        grouped_rewards = self._compute_group_rewards(
            prompt_ids,
            data["rm_scores"],
        )

        for p_id, avg_reward in grouped_rewards.items():
            for scheduler in self.interleave_schedulers:
                scheduler.update(p_id, self.global_steps, avg_reward)

                # if hasattr(scheduler, 'save_history'):
                #     scheduler.save_history(self.global_steps)
                #
        # Save reward history to disk
        # _s is any scheduler.
        _s = self.interleave_schedulers[0]
        if _s and hasattr(_s, "save_history"):
            _s.save_history(self.global_steps)



    def _compute_group_rewards(
        self,
        prompt_ids,
        rm_scores,
    ) -> dict[str, float]:

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
        """
        Inject the functionality to record the reward history..
        The original _step_once function is not modified.
        """
        batch = super()._step_once(metrics, timing_raw, sample_batch_size)
        try:
            self._record_reward_history(batch)

            #=========================================================
            # Record the generated tokens after each rollout
            #==========================================================
            gen_time = metrics.get("timing_s/gen")
            gen_time_per_token = metrics.get("timing_per_token_ms/gen")

            if gen_time is not None and gen_time_per_token is not None and gen_time_per_token > 0:
                generated_tokens = gen_time * 1000 / gen_time_per_token
                metrics["duet/generated_total_tokens"] = round(generated_tokens)

        except Exception as error:
            print(f"[DUET] Error => {str(error)}")
            raise
        return batch


    def step(self, metrics: dict, timing_raw: dict) -> KVBatchMeta:
        """
        Completely override the original step method.
        This step functoin replaces the batch_size needed in each step once function.
        The original batch_size is no longer static and can alter based on the underfill_policy.
        Additionally new metrics (stats from filtering the prompt) are included with the metrics
        that are saved.
        """

        normal_batch_size = self.config.data.train_batch_size

        assert normal_batch_size % self.parameter_sync_step == 0, (
            f"train_batch_size ({normal_batch_size}) must be divisible by "
            f"parameter_sync_step ({self.parameter_sync_step})"
        )
        # regular feed: stream one train batch worth of prompts for this step
        with marked_timer("feed", timing_raw):
            self._add_batch_to_generate()

        # Take a snapshot of the scheduler stats
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
                "duet/prompts_selected": self.selected_count,
                "duet/prompts_skipped_easy": scheduler_stats_snapshot.skipped_easy,
                "duet/prompts_skipped_hard": scheduler_stats_snapshot.skipped_hard,
                "duet/scheduler_selected": default_schedulers.get(self.scheduler.name, -1) if self.scheduler else -1
                # "duet/prompts_considered": scheduler_stats_snapshot.considered,

            })

        step_total_tokens = int(metrics.get("perf/total_num_tokens", 0) or 0)
        step_response_tokens = int(metrics.get("duet/generated_response_tokens", 0) or 0)
        self.cumulative_total_tokens += step_total_tokens
        self.cumulative_response_tokens += step_response_tokens
        metrics["duet/cumulative_total_tokens"] = self.cumulative_total_tokens
        metrics["duet/cumulative_response_tokens"] = self.cumulative_response_tokens
        metrics["duet/cumulative_optimizer_steps"] = self.global_steps

        return KVBatchMeta(partition_id=combined_partition_id, keys=combined_keys, tags=combined_tags)

    def _compute_metrics(
        self,
        batch,
        metrics,
        timing_raw,
        global_steps,
        epoch,
    ):
        """
        """
        super()._compute_metrics(
            batch,
            metrics,
            timing_raw,
            global_steps,
            epoch,
        )

        try:

            self._save_generation_times(
                batch,
                metrics,
                global_steps,
            )
        except Exception as error:
            print("Saving generation times failed. See error below")
            print(f"Error: {str(error)}")

    

    def _save_generation_times(self, batch, metrics, global_steps):
        """
        Saving all the rollout durations into a .npy file for each step
        """
        import os
        import numpy as np
        from verl.utils import tensordict_utils as tu

        real_keys = [
            key
            for key, tag in zip(batch.keys, batch.tags)
            if not tag.get("is_padding", False)
        ]

        data = tq.kv_batch_get(
            keys=real_keys,
            partition_id=batch.partition_id,
            select_fields=["metrics", "responses"],
        )




        # ===========================================================
        # Save generation durations (raw second counts) for all rollouts. 
        # ===========================================================


        agent_metrics = tu.get(data, "metrics")

        generation_times = np.asarray(
            [
                float(m["generate_sequences"])
                for m in agent_metrics
            ],
            dtype=np.float64,
        )

        trace_dir = os.path.join(os.environ["DUET_RUN_DIR"], "gen_times",)
        os.makedirs(trace_dir, exist_ok=True)

        path = os.path.join(trace_dir, f"gen_time_call{global_steps - 1:03d}.npy")
        np.save(path, generation_times)

        print(
            f"[DUET] saved {len(generation_times)} generation times "
            f"to {path}"
        )

        # ==============================================================
        # Save p95 for response lengths and the total generated response
        # tokens.
        # ==============================================================

        _responses = data.get("responses")
        if _responses is None:
            print("[DUET] no parameter `responses` found the data retrieved from TQ")
            return

        response_lengths = _responses.offsets().diff()
        metrics["response_length/p95"] = torch.quantile(
            response_lengths.float(), 0.95
            ).item()
        metrics["response_length/max_from_tq"] = int(
            response_lengths.max().item()
        )
        metrics["duet/generated_response_tokens"] = int(
            response_lengths.sum().item()
        )
        metrics["response_length/at_cap"] = int(
            (response_lengths == self.config.data.max_response_length).sum().item()
        )


def install_duet_trainer():
    print(f"[DUET] replacing trainer: {TRAINER_REGISTRY['sync']} -> DuetPPOTrainerSync")
    TRAINER_REGISTRY["sync"] = DuetPPOTrainerSync
