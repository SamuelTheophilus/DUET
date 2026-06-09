"""
verify_image.py
-------------
Verifies that a Modal GPU container (A10G or T4) can be launched successfully.
Prints CUDA device info and confirms torch sees the GPU.

Run with:
    modal run verify_gpu.py
"""
import modal
from image import core_duet_image as image

T4_GPU = 't4'
A10_GPU = 'a10g'


# MODAL SET UP.
app = modal.App("verify_image", image=image)
volume = modal.Volume.from_name("duet-volume", create_if_missing=True)



@app.function(gpu=T4_GPU)
def check():
    import torch
    import subprocess

    try:
        import verl
        verl_version = verl.__version__
    except ImportError:
        verl_version = "NOT INSTALLED"

    try:
        import flash_attn
        fa_version = flash_attn.__version__
    except ImportError:
        fa_version = "NOT INSTALLED"
    try:
        import vllm
        vllm_version = vllm.__version__
    except ImportError:
        vllm_version = "NOT INSTALLED"

    smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, 
            text=True
        ).stdout.strip()


    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "flash_attn": fa_version,
        "verl": verl_version,
        "vllm": vllm_version,
        "gpu_device_name": device_name,
        "nvidia_smi": smi
    }


@app.local_entrypoint()
def main():
    print("Launching GPU container on Modal...")
    result = check.remote()

    print("\n--- DUET image verification result ---")
    for k, v in result.items():
        print(f"  {k}: {v}")

