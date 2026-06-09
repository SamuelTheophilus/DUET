# Modal Setup & Auth Flow

## Overview

Modal is a serverless GPU cloud platform. You write Python locally, and Modal
runs it in a managed container — no Dockerfile, no Kubernetes, no SSH into
instances. Infrastructure is declared in code via decorators and image builders.

---

## Prerequisites

- Python 3.10+ (match your local version to what you declare in `add_python`)
- A Modal account at [modal.com](https://modal.com)

---

## Installation

```bash
pip install modal
```

Confirm it installed:

```bash
modal --version
```

---

## Authentication Flow

Modal uses a browser-based OAuth flow. Run:

```bash
modal setup
```

What happens step by step:

1. The CLI prints a URL and opens your browser automatically.
2. You log in to your Modal account (or create one) and select the workspace (`duet`)
3. Modal issues a token that is written to `~/.modal.toml` on your Mac.
4. All subsequent `modal` commands use that token silently — no login prompt.

Your token file lives at:

```
~/.modal.toml
```

To inspect it:

```bash
cat ~/.modal.toml
```

To authenticate as a different workspace or refresh a stale token:

```bash
modal token new
```

---

## Workspaces

Modal organises resources (apps, secrets, volumes) under a **workspace**,
tied to your account or an organisation. After `modal setup` you are
automatically in your personal workspace.

To check which workspace you are currently authenticated against:

```bash
modal profile current
```

To list all profiles (useful if you have multiple workspaces):

```bash
modal profile list
```

---

## verl Image

The final working image uses `verlai/verl:vllm017.dev3` as the base. This image
already contains torch, flash-attn, and vLLM — all pre-built and compatible.
verl and additional packages are installed on top.

```python
import modal

VERL_COMMIT_TAG = "v0.8.0"

image = (
    modal.Image.from_registry("verlai/verl:vllm017.dev3")
    .run_commands(
        f"pip install --no-build-isolation git+https://github.com/volcengine/verl.git@{VERL_COMMIT_TAG}"
    )
    .uv_pip_install(
        "bitsandbytes==0.49.1",
        "transformers==5.10.2",
        "accelerate==1.13.0",
        "datasets==5.0.0",
    )
    .add_local_python_source("duet_image")
)
```

Note: `add_local_python_source("duet_image")` makes the image module importable in remote containers. 

### What's in the base image

| Package | Version |
|---|---|
| CUDA | 12.9.1 |
| torch | 2.10.0+cu129 |
| flash-attn | 2.8.3 |
| vLLM | 0.17.0 |

### Verifying the image

```bash
modal run verify_image.py
```

### ✅ Verified


```
--- Verl image verification result ---
  torch: 2.10.0+cu129
  cuda: 12.9
  flash_attn: 2.8.3
  verl: 0.8.0.dev
  vllm: 0.17.0
  gpu_device_name: Tesla T4
  nvidia_smi: Tesla T4, 15360 MiB, 580.95.05
```

### Why this approach

Building the full stack from a raw CUDA base image fails because:
- flash-attn has no pre-built wheels for torch >= 2.8
- Compiling flash-attn from source takes minutes/hours and requires specific
  clang/compiler versions
- vLLM's GPU wheels are not on PyPI — they require direct GitHub release URLs
- torch, vLLM, and flash-attn have tight mutual version constraints

Using the verl-provided base image sidesteps all of this — the compatibility
problem is already solved upstream.
---
