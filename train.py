import modal
from datetime import datetime, timezone


from duet.config import load_infra
from image import core_duet_image as IMAGE

cfg = load_infra()

VOLUME_MOUNT = "/data"
TRAIN_TYPES = ["baseline", "duet"]

app = modal.App("duet-train", image=IMAGE)
volume = modal.Volume.from_name(cfg.volume_name, create_if_missing=True)


# ===========================================================================
#  helper functions
# ===========================================================================


def make_run_info(test: bool = False):
    run_description = input(
        "Enter a description for the experiment you're about to run: \n\n"
    ).strip()

    if not run_description:
        run_description = "[NO EXPERIMENT DESCRIPTION PROVIDED]"

    if test:
        run_id = "test"
    else:
        run_id_suffix = input("Enter a suffix to identiy the run:\n\n").strip()
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_id = f"{run_id}_{run_id_suffix}"

    return run_id, run_description


def initialize_run_directories(
    run_id: str,
    run_description: str,
    test: bool = False,
):
    import os
    import shutil
    import subprocess
    import socket
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

    nvidia_cmd = ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"]
    gpu_uuid = subprocess.check_output(nvidia_cmd).decode("utf-8").strip()

    container_hostname = f"--- Container Hostname {socket.gethostname()}"
    gpu_uuid_record = f"--- GPU UUID {gpu_uuid}"



    # ======================================
    # Writing description of the run.
    # ======================================
    run_description = f"{run_description}\n\n{container_hostname}\n\n{gpu_uuid_record}"
    (run_dir / "description.txt").write_text(run_description)

    env = os.environ.copy()
    env["DUET_RUN_ID"] = run_id
    env["DUET_RUN_DIR"] = str(run_dir)

    mlflow_dir = Path("/data/mlflow")
    mlflow_dir.mkdir(parents=True, exist_ok=True)


    # Creating the streak directory
    streak_dir = run_dir / "prompt_streak"
    env["PROMPT_STREAK_DIR"] = str(streak_dir)
    streak_dir.mkdir(parents=True, exist_ok=True)

    env["MLFLOW_TRACKING_URI"] = f"sqlite:////data/mlflow/{run_id}.db"
    env["VLLM_LOGGING_LEVEL"] = "INFO"
    env["HYDRA_FULL_ERROR"] = "1"


    

    return env


def save_train_config(cmd, env):
    import subprocess
    from pathlib import Path

    run_dir = Path(env["DUET_RUN_DIR"])
    config_path = run_dir / "config.yaml"

    result = subprocess.run(
        [*cmd, "--cfg", "job", "--resolve"],
        check=True,
        env=env,
        capture_output=True,
        text=True,
    )

    config_path.write_text(result.stdout)

    print(f"Saved resolved training config to {config_path}")


# ===========================================================================
#  @app.functions (modal specific functions)
# ===========================================================================


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


SWEEP_MATRIX = [
    {"scheduler": "noops", "seed": 42},
    {"scheduler": "noops", "seed": 43},
    {"scheduler": "noops", "seed": 44},
    {"scheduler": "difficulty_only", "seed": 42},
    {"scheduler": "difficulty_only", "seed": 43},
    {"scheduler": "difficulty_only", "seed": 44},
]


@app.function(
    gpu="a100-40gb",
    volumes={VOLUME_MOUNT: volume},
    timeout=24 * 3600,
    memory=65536,
)
def _profile(
    run_id: str,
    run_description: str,
    overrides: list[str] | None = None,
    test: bool = False,
):
    import subprocess
    from pathlib import Path

    from duet.training import train_cmd

    if overrides is None:
        overrides = []

    print(f"[{run_id}] starting — overrides: {overrides}", flush=True)

    env = initialize_run_directories(
        run_id=run_id,
        run_description=run_description,
        test=test,
    )

    cmd = train_cmd(overrides=overrides)


    save_train_config(cmd, env)

    run_dir = Path(env["DUET_RUN_DIR"])
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "train.log"

    (run_dir / "command.txt").write_text(" ".join(cmd) + "\n")
    env["PYTHONUNBUFFERED"] = "1"

    command = " ".join(cmd)

    try:
        subprocess.run(
            [
                "/bin/bash",
                "-c",
                f"{command} 2>&1 | tee -a {log_path}",
            ],
            check=True,
            env=env,
        )
    finally:
        volume.commit()

    print(f"[{run_id}] done", flush=True)


@app.local_entrypoint()
def main(
    train_type: str = "baseline",
    setup: bool = False,
    force: bool = False,
    profile: bool = False,
    # sweep: bool = False,
    test: bool = False,
):
    if setup:
        _setup.remote(force)
        return

    # if sweep:
    #     description = input("Enter a description for this sweep:\n\n").strip()
    #     if not description:
    #         description = "[NO SWEEP DESCRIPTION PROVIDED]"
    #
    #     timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    #
    #     handles = []
    #     for exp in SWEEP_MATRIX:
    #         run_id = f"{timestamp}_{exp['scheduler']}_seed{exp['seed']}"
    #         overrides = [
    #             f"trainer.duet.scheduler={exp['scheduler']}",
    #             f"custom_config.seed={exp['seed']}",
    #         ]
    #         h = _profile.spawn(
    #             run_id=run_id,
    #             run_description=description,
    #             overrides=overrides,
    #             test=test,
    #         )
    #         handles.append((run_id, h))
    #         print(f"Spawned: {run_id}")
    #
    #     print(f"\n{len(handles)} runs launched. Waiting for all to complete...\n")
    #     for run_id, h in handles:
    #         h.get()
    #         print(f"Completed: {run_id}")
    #     return
    #
    if train_type not in TRAIN_TYPES:
        raise ValueError(
            f"Unknown --train-type '{train_type}'. Choose from: {TRAIN_TYPES}"
        )

    run_id, description = make_run_info(test=test)
    if profile:
        _profile.spawn(run_id, description, test=test)
    else:
        _train.remote(train_type)
