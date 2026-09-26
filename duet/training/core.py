import os
import re
import random
from pathlib import Path

VOLUME_MOUNT = Path("/data")
MODEL_ID = "Qwen/Qwen2.5-Math-1.5B"

GSM8K_DIR = VOLUME_MOUNT / "gsm8k"
MODEL_DIR = VOLUME_MOUNT / "models" / "Qwen2.5-Math-1.5B"


TRAIN_PARQUET = GSM8K_DIR / "train.parquet"
TEST_PARQUET = GSM8K_DIR / "test.parquet"




def download_dataset() -> None:
    import datasets

    os.makedirs(GSM8K_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    print(f"[{download_dataset.__name__}] downloading and preprocessing dataset ...")

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
                "prompt_id": f"{split}_{idx}",
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
                    "predicted_response_length": random.randint(0, 99),
                    "split": split,
                    "index": idx,
                    "answer": example["answer"],
                    "question": question,
                    "prompt_id": f"{split}_{idx}",
                },
            }

        return fn

    print(f"[{download_dataset.__name__}] processing dataset...")
    dataset["train"].map(process("train"), with_indices=True).to_parquet(TRAIN_PARQUET)
    dataset["test"].map(process("test"), with_indices=True).to_parquet(TEST_PARQUET)

    print(f"[{download_dataset.__name__}] downloading & processing completed")
    print(f"GSM8K saved to {GSM8K_DIR}")


def download_model() -> None:
    from huggingface_hub import snapshot_download

    MODEL_CONFIG = MODEL_DIR / "config.json"

    if not MODEL_CONFIG.exists():
        print(f"[{download_model.__name__}] downloading model ...")
        snapshot_download(repo_id=MODEL_ID, local_dir=MODEL_DIR)
        print(f"[{download_model.__name__}] model saved to {MODEL_DIR}...")
    else:
        print(f"[{download_model.__name__}] model already exists at {MODEL_DIR}...")


def setup(force: bool = False):
    if not TRAIN_PARQUET.exists():
        download_dataset()
    elif force:
        download_dataset()
    else:
        print(f"GSM8K already present at {GSM8K_DIR}, skipping.")

    download_model()


def train_cmd(overrides: list[str] = []) -> list[str]:
    cmd = [
        "python3",
        "-m",
        "verl.trainer.main_ppo",
        "--config-dir=/root/",
        "+config=duet_rlhf",
    ]
    return cmd + overrides
