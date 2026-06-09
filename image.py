import modal
from duet.config import load_image_config

cfg = load_image_config()

core_duet_image = (
    modal.Image.from_registry(cfg.base_registry)
    .run_commands(
        f"pip install --no-build-isolation git+https://github.com/volcengine/verl.git@{cfg.verl_commit_tag}"
    )
    .uv_pip_install(*cfg.packages)
    .add_local_python_source("image", "duet")
)
