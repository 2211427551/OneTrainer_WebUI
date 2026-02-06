# OneTrainer

Training GUI for Stable Diffusion.

## WebUI

The new Gradio-based WebUI provides a modern, user-friendly interface for training.

### Quick Start (Linux/macOS)

```bash
./start-webui.sh
```

For more detailed deployment instructions, including remote server setup and token authentication, please refer to the **[Deployment Guide (Chinese)](DEPLOY_GUIDE_CN.md)**.

## Features

- **Dashboard**: Real-time training progress, loss curves, and system resource monitoring.
- **Dynamic Configuration**: 100% parameter coverage with intelligent field visibility (e.g., specific options for LoKr/LoRA).
- **Light Mode**: Default clean light interface for better readability.
- **Dataset Tools**: Built-in tools for dataset scanning and prompt validation.

## Traditional UI

If you prefer the classic UI, you can still run it via:

```bash
python scripts/webui.py --ui classic
```
