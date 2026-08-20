from pathlib import Path


VOLUME_MOUNT = Path("/data")
MODEL_ID = "Qwen/Qwen2.5-Math-1.5B"

GSM8K_DIR = VOLUME_MOUNT / "gsm8k"
MODEL_DIR = VOLUME_MOUNT / "models" / "Qwen2.5-Math-1.5B"

# Load this from the .yam_config files. 
TRACES_FOLDER_NAME = "test_traces"
ROLLOUT_DATA_FOLDER = "test_rollout_data"

TRAIN_PARQUET = GSM8K_DIR / "train.parquet"
TEST_PARQUET = GSM8K_DIR / "test.parquet"


__all__ = [
    "VOLUME_MOUNT",
    "MODEL_ID",
    "GSM8K_DIR",
    "MODEL_DIR",
    "TRACES_FOLDER_NAME",
    "ROLLOUT_DATA_FOLDER",
    "TRAIN_PARQUET",
    "TEST_PARQUET",
]
