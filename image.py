import modal
from pathlib import Path
from duet.config import load_image_config


YAML_CONFIG_PATH: Path = Path("./config/")
MODAL_CONFIG_PATH: str = "/root/config/"

cfg = load_image_config()

VERL_COMMIT_TAG: str = "a07d3e32d2b40ec35e9514ebebd7361db4d6f554"
core_duet_image = (
    modal.Image.from_registry(cfg.base_registry)
    .run_commands(
        f"pip install --no-build-isolation git+https://github.com/volcengine/verl.git@{VERL_COMMIT_TAG}"
    )
    .uv_pip_install(*cfg.packages)
    .add_local_python_source("image", "duet")
    .add_local_file(YAML_CONFIG_PATH / "image.yaml", f"{MODAL_CONFIG_PATH}/image.yaml")
    .add_local_file(YAML_CONFIG_PATH / "infra.yaml", f"{MODAL_CONFIG_PATH}/infra.yaml")
    .add_local_file(
        YAML_CONFIG_PATH / "duet_rlhf.yaml", f"{MODAL_CONFIG_PATH}/duet_rlhf.yaml"
    )
)
