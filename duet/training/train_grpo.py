import os
import re

VOLUME_MOUNT = "/data"
MODEL_ID = "Qwen/Qwen2.5-Math-1.5B"
GSM8K_DIR = f"{VOLUME_MOUNT}/gsm8k"
MODEL_DIR = f"{VOLUME_MOUNT}/models/Qwen2.5-Math-1.5B"


def setup():
    import datasets
    from huggingface_hub import snapshot_download

    os.makedirs(GSM8K_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    train_parquet = os.path.join(GSM8K_DIR, "train.parquet")
    test_parquet = os.path.join(GSM8K_DIR, "test.parquet")

    if not os.path.exists(train_parquet):
        print("Downloading and preprocessing GSM8K...")
        dataset = datasets.load_dataset("openai/gsm8k", "main")

        def extract_answer(solution_str):
            match = re.search(r"#### (-?[0-9.,]+)", solution_str)
            assert match, f"No answer found in: {solution_str}"
            return match.group(1).replace(",", "")

        instruction = (
            "Please reason step by step, and put your final answer within \\boxed{}. "
            "for example: \\boxed{8} "
        )

        def process(split):
            def fn(example, idx):
                question = example["question"]
                return {
                    "data_source": "openai/gsm8k",
                    "prompt": [
                        {"role": "system", "content": instruction},
                        {
                            "role": "user",
                            "content": question,
                        },
                    ],
                    "ability": "math",
                    "reward_model": {
                        "style": "rule",
                        "ground_truth": extract_answer(example["answer"]),
                    },
                    "extra_info": {
                        "split": split,
                        "index": idx,
                        "answer": example["answer"],
                        "question": question,
                    },
                }

            return fn

        dataset["train"].map(process("train"), with_indices=True).to_parquet(train_parquet)
        dataset["test"].map(process("test"), with_indices=True).to_parquet(test_parquet)
        print(f"GSM8K saved to {GSM8K_DIR}")
    else:
        print(f"GSM8K already present at {GSM8K_DIR}, skipping.")

    model_config = os.path.join(MODEL_DIR, "config.json")
    if not os.path.exists(model_config):
        print(f"Downloading {MODEL_ID}...")
        snapshot_download(repo_id=MODEL_ID, local_dir=MODEL_DIR)
        print(f"Model saved to {MODEL_DIR}")
    else:
        print(f"Model already present at {MODEL_DIR}, skipping.")


def profile_cmd() -> list[str]:
    cmd = train_cmd()

    cmd = [c for c in cmd if not c.startswith("trainer.total_training_steps")]
    cmd += [
        "trainer.total_training_steps=3",
        "global_profiler.tool=torch",
        "global_profiler.steps=[1,2]",
        "global_profiler.save_path=/data/traces",
        "+ray_kwargs.ray_init.runtime_env.worker_process_setup_hook=duet.training.profile_patch.patch_verl_for_profiling",
        "actor_rollout_ref.actor.profiler.enable=True",
        "actor_rollout_ref.actor.profiler.all_ranks=False",
        "actor_rollout_ref.actor.profiler.ranks=[0]",
        "actor_rollout_ref.actor.profiler.tool_config.torch.discrete=True",
        "actor_rollout_ref.actor.profiler.tool_config.torch.contents=[cpu,cuda]",
        "actor_rollout_ref.rollout.profiler.enable=True",
        "actor_rollout_ref.rollout.profiler.all_ranks=False",
        "actor_rollout_ref.rollout.profiler.ranks=[0]",
        "actor_rollout_ref.rollout.profiler.tool_config.torch.discrete=True",
    ]
    return cmd


def train_cmd() -> list[str]:
    return [
        "python3",
        "-m",
        "verl.trainer.main_ppo",
        # DATA
        "algorithm.adv_estimator=grpo",
        f"data.train_files={GSM8K_DIR}/train.parquet",
        f"data.val_files={GSM8K_DIR}/test.parquet",
        "data.train_batch_size=512",
        "data.max_prompt_length=1024",
        "data.max_response_length=1024",
        "data.filter_overlong_prompts=True",
        "data.truncation=error",
        "algorithm.use_kl_in_reward=False",
        "custom_reward_function.path=/root/duet/training/reward.py",
        "custom_reward_function.name=custom_reward_function",
        # MODEL
        f"actor_rollout_ref.model.path={MODEL_DIR}",
        "actor_rollout_ref.model.use_remove_padding=True",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        # ACTOR
        "actor_rollout_ref.actor.optim.lr=1e-6",
        "actor_rollout_ref.actor.ppo_mini_batch_size=256",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2",
        "actor_rollout_ref.actor.use_kl_loss=True",
        "actor_rollout_ref.actor.kl_loss_coef=0.001",
        "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
        "actor_rollout_ref.actor.entropy_coeff=0",
        "actor_rollout_ref.actor.fsdp_config.param_offload=False",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False",
        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=3000",
        "actor_rollout_ref.actor.use_dynamic_bsz=True",
        # ROLLOUT
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.3",
        "actor_rollout_ref.rollout.enable_chunked_prefill=False",
        "actor_rollout_ref.rollout.enforce_eager=False",
        "actor_rollout_ref.rollout.free_cache_engine=True",
        "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True",
        "actor_rollout_ref.rollout.max_num_seqs=512",
        "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=4096",
        "actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=256",
        "actor_rollout_ref.rollout.n=5",
        # REF
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2",
        "actor_rollout_ref.ref.fsdp_config.param_offload=True",
        "actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True",
        "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=8192",
        # TRAINER
        "trainer.critic_warmup=0",
        'trainer.logger=["console","mlflow"]',
        "trainer.project_name=verl_grpo_gsm8k_profile",
        "trainer.experiment_name=qwen2.5_math_1.5b_fsdp_200steps",
        "trainer.n_gpus_per_node=1",
        "trainer.nnodes=1",
        "trainer.save_freq=20",
        "trainer.test_freq=5",
        "trainer.total_training_steps=200",
        "trainer.total_epochs=15",
        "trainer.rollout_data_dir=/data/traces/rollout_data",
        "trainer.val_before_train=False",
    ]
