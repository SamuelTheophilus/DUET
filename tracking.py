import modal
from duet.config import load_infra
from image import core_duet_image as IMAGE

cfg = load_infra()

app = modal.App(cfg.mlflow.app_name, image=IMAGE)
volume = modal.Volume.from_name(cfg.volume_name, create_if_missing=True)

# Extracted as module-level constants so Modal's serializer captures them cleanly
# when running the function body remotely.
_backend_store_uri = cfg.mlflow.backend_store_uri
_port = cfg.mlflow.port


@app.function(volumes={cfg.mlflow.mount_path: volume})
@modal.web_server(port=cfg.mlflow.port, startup_timeout=120)
def experiment_logs():
    import subprocess
    import time

    subprocess.Popen([
        "mlflow", "server",
        "--backend-store-uri", _backend_store_uri,
        "--port", str(_port),
        "--host", "0.0.0.0",
        "--allowed-hosts", "*",
        "--cors-allowed-origins", "*",
    ])
    time.sleep(10)
