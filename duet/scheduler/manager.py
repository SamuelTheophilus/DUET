# import os
import ray
from tensordict import TensorDict
# import torch
# from .core import  Scheduler, get_scheduler

from verl.trainer.ppo.v1 import AgentLoopManagerTQ
from verl.utils.ray_utils import auto_await


class MyAgentLoopManager:
    def __init__(self, inner_manager, config):
        self.inner_manager = inner_manager
        self.config = config
        # _schdeduler_cls = get_scheduler(self.config.trainer.duet.scheduler)
        # self.scheduler: Scheduler | None  = _schdeduler_cls() if _schdeduler_cls else None

    @classmethod
    @auto_await
    async def create(
        cls,
        config,
        llm_client,
        teacher_client=None,
        reward_loop_worker_handles=None,
    ):
        print("[DUET] Creating MyAgentLoopManager")

        inner_manager = await AgentLoopManagerTQ.create(
            config=config,
            llm_client=llm_client,
            teacher_client=teacher_client,
            reward_loop_worker_handles=reward_loop_worker_handles,
        )

        print("[DUET] Normal AgentLoopManagerTQ created")

        #TODO:Select the scheduler from here via the loaded config.
        # For now test with the manually created schedulers.
        # scheduler = LengthOnlyScheduler()

        return cls(
            inner_manager=inner_manager,
            config=config,
        )

    def _generate_sequences(self, prompts: TensorDict) -> None:
        """
        Custom wrapper over the original generate_sequences.
        Modifies prompts by filtering, reordering or packing before
        forwarding the modified batch into generate_sequences.
        NB: this function is called during the rollout and the update steps.
        Ergo, it's configured to run only during the rollout phase.


        Args:
        prompts: TensorDict

        Returns:
        - TensorDict
        """
        print("=" * 80)
        print("[DUET] MyAgentLoopManager.generate_sequences called")
        print(f"[DUET] Batch size: {len(prompts)}")
        print("=" * 80)

        # batch_size: int = len(prompts)

        # save_dir = "/data/debug"
        # os.makedirs(save_dir, exist_ok=True)
        # save_path = os.path.join(save_dir, "prompts_before_generation.pt")
        # torch.save(prompts, save_path)
        # print(f"[DUET] Saved prompts to {save_path}")
        # TODO: add a check to only call the scheduler during the rollout phase and not the updating phase.
        # if self.scheduler:
        #     prompts = self.scheduler.select_prompts(prompts, batch_size=batch_size)
        #     prompts = self.scheduler.reorder(prompts)
        #
        return self.inner_manager.generate_sequences(prompts)

    def generate_sequences(self, prompts: TensorDict) -> None:

        print(f"[duet] Batch size: {len(prompts)}")

        if "assign_to_worker" not in prompts:

            print("*" * 80)
            print("[duet] generating prompt without assigning to specific workers")
            print("*" * 80)

            return self.inner_manager.generate_sequences(prompts)

        worker_ids = prompts["assign_to_worker"]

        refs = []

        for worker_idx, worker in enumerate(
            self.inner_manager.agent_loop_workers
        ):
            mask = worker_ids == worker_idx

            if not mask.any():
                continue

            worker_batch = prompts[mask]
            worker_batch.pop("assign_to_worker")

            refs.append(
                worker.generate_sequences.remote(worker_batch)
            )

        print("*" * 80)
        print("[duet] generating sequences with assigning to specific workers")
        print("*" * 80)


        ray.get(refs)
