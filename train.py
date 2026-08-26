import modal
from datetime import datetime, timezone


from duet.config import load_infra
from image import core_duet_image as IMAGE

cfg = load_infra()

VOLUME_MOUNT = "/data"
TRAIN_TYPES = ["baseline", "duet"]

app = modal.App("duet-train", image=IMAGE)
volume = modal.Volume.from_name(cfg.volume_name, create_if_missing=True)



def make_run_info(test: bool = False):
    run_description = input("Enter a description for the experiment you're about to run: \n\n").strip()

    if not run_description:
        run_description = "[NO EXPERIMENT DESCRIPTION PROVIDED]"

    if test:
        run_id = "test"
    else:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    return run_id, run_description

def initialize_run_directories(
    run_id: str,
    run_description: str,
    test: bool = False,
):
    import os
    import shutil
    from pathlib import Path

    if test:
        run_dir = Path("/data/runs/test")
        run_dir.mkdir(parents=True, exist_ok=True)
        # Empty the test run directory when running with test
        for item in run_dir.iterdir():
            if item.is_dir() and not item.is_symlink():
                shutil.rmtree(item)
            else:
                item.unlink()
    else:
        run_dir = Path("/data/runs") / run_id
        run_dir.mkdir(parents=True, exist_ok=True)


    (run_dir / "description.txt").write_text(run_description)

    env = os.environ.copy()
    env["DUET_RUN_ID"] = run_id
    env["DUET_RUN_DIR"] = str(run_dir)


    # Defaults
    env["MLFLOW_TRACKING_URI"] = "sqlite:////data/duet.db"
    env["HYDRA_FULL_ERROR"] = "1"

    return env


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
def _profile(
    run_id: str,
    run_description: str,
    test: bool = False
):
    import subprocess
    from duet.training import train_cmd

    print("duet with profiling...")

    env =  initialize_run_directories(
        run_id=run_id,
        run_description=run_description,
        test=test
    )
    subprocess.run(train_cmd(), check=True, env=env)
    volume.commit()



@app.local_entrypoint()
def main(
    train_type: str = "baseline",
    setup: bool = False,
    force: bool = False,

    profile: bool = False,
    test: bool = False
):
    if train_type not in TRAIN_TYPES:
        raise ValueError(
            f"Unknown --train-type '{train_type}'. Choose from: {TRAIN_TYPES}"
        )

    if setup:
        _setup.remote(force)
        return

    run_id, description = make_run_info(test=test)
    if profile:
        _profile.remote(run_id, description, test)
    else:
        _train.remote(train_type)

