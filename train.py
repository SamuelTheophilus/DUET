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
def _setup(train_type: str):
    if train_type == "baseline":
        from duet.training import train_grpo

        train_grpo.setup()
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
        from duet.training import train_grpo

        assert os.path.exists(f"{train_grpo.GSM8K_DIR}/train.parquet"), (
            "Missing GSM8K data — run with --setup first"
        )
        assert os.path.exists(f"{train_grpo.MODEL_DIR}/config.json"), (
            "Missing model weights — run with --setup first"
        )
        env = os.environ.copy()
        env["MLFLOW_TRACKING_URI"] = "sqlite:////data/duet.db"
        subprocess.run(train_grpo.train_cmd(), check=True, env=env)


@app.function(
    gpu="a100-40gb",
    volumes={VOLUME_MOUNT: volume},
    timeout=4 * 3600,
    memory=65536,
)
def _profile(train_type: str = "baseline"):
    import os
    import subprocess

    if train_type == "baseline":
        print("[training with profiling...]")
        from duet.training import train_grpo

        env = os.environ.copy()
        env["MLFLOW_TRACKING_URI"] = "sqlite:////data/duet.db"
        env["HYDRA_FULL_ERROR"] = "1"
        subprocess.run(train_grpo.profile_cmd(), check=True, env=env)
        volume.commit()


@app.local_entrypoint()
def main(train_type: str = "baseline", setup: bool = False, profile: bool = False):
    if train_type not in TRAIN_TYPES:
        raise ValueError(
            f"Unknown --train-type '{train_type}'. Choose from: {TRAIN_TYPES}"
        )
    if setup:
        _setup.remote(train_type)
    if profile:
        _profile.remote(train_type)
    else:
        _train.remote(train_type)
