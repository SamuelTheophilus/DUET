from pathlib import Path

from pydantic import BaseModel
import yaml

_CONFIG_DIR = Path(__file__).parent.parent / "config"


class MLflowConfig(BaseModel):
    app_name: str
    port: int
    mount_path: str
    backend_store_uri: str


class InfraConfig(BaseModel):
    volume_name: str
    mlflow: MLflowConfig


class ImageConfig(BaseModel):
    base_registry: str
    verl_commit_tag: str
    packages: list[str]


def load_infra() -> InfraConfig:
    with open(_CONFIG_DIR / "infra.yaml") as f:
        return InfraConfig(**yaml.safe_load(f))


def load_image_config() -> ImageConfig:
    with open(_CONFIG_DIR / "image.yaml") as f:
        return ImageConfig(**yaml.safe_load(f))
