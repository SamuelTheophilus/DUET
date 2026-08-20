# import os
import modal
from image import core_duet_image as IMAGE
from duet.config import load_infra

cfg = load_infra()

VOLUME_MOUNT = "/data"
TRAIN_TYPES = ["baseline", "duet"]

app = modal.App("duet-train", image=IMAGE)
volume = modal.Volume.from_name(cfg.volume_name, create_if_missing=True)


@app.function(
    volumes={VOLUME_MOUNT: volume},
    timeout=3600,
    cpu=4,
    memory=16384,
)
def _setup(force: bool):
    from duet.training import setup
    setup(force)
    volume.commit()


@app.function(
    gpu="a100-40gb",
    volumes={VOLUME_MOUNT: volume},
    timeout=4 * 3600,
    memory=65536,
)
def _train(train_type: str):
    import os
    import subprocess

    if train_type == "baseline":
        from config import GSM8K_DIR, MODEL_DIR
        from duet.training import train_cmd


        assert os.path.exists(f"{GSM8K_DIR}/train.parquet"), (
            "Missing GSM8K data — run with --setup first"
        )
        assert os.path.exists(f"{MODEL_DIR}/config.json"), (
            "Missing model weights — run with --setup first"
        )
        env = os.environ.copy()
        env["MLFLOW_TRACKING_URI"] = "sqlite:////data/duet.db"
        subprocess.run(train_cmd(), check=True, env=env)


@app.function(
    gpu="a100-40gb",
    volumes={VOLUME_MOUNT: volume},
    timeout=4 * 3600,
    memory=65536,
)
def _profile(train_type: str = "baseline"):
    import os
    import subprocess
    from duet.training.reward import custom_reward_function


    print("[Testing Custom Reward Function]....")
    print(
        custom_reward_function(
            data_source="openai/gsm8k",
            solution_str=r"The answer is therefore \boxed{42}.",
            ground_truth="42",
        )
    )

    print(
        custom_reward_function(
            data_source="openai/gsm8k",
            solution_str=r"The answer is therefore \boxed{41}.",
            ground_truth="42",
        )
    )

    if train_type == "baseline":
        print("[training with profiling...]")
        from duet.training import train_cmd

        env = os.environ.copy()
        env["MLFLOW_TRACKING_URI"] = "sqlite:////data/duet.db"
        env["HYDRA_FULL_ERROR"] = "1"
        subprocess.run(train_cmd(), check=True, env=env)
        volume.commit()



@app.local_entrypoint()
def main(
    train_type: str = "baseline",
    setup: bool = False,
    profile: bool = False,
    force: bool = False,
):
    if train_type not in TRAIN_TYPES:
        raise ValueError(
            f"Unknown --train-type '{train_type}'. Choose from: {TRAIN_TYPES}"
        )
    if setup:
        _setup.remote(force)
        return
    if profile:
        _profile.remote(train_type)
    else:
        _train.remote(train_type)
