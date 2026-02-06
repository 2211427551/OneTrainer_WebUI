from util.import_util import script_imports

script_imports()

import argparse
import os
from pathlib import Path

import uvicorn

from modules.webui.app import create_app
from modules.webui.gradio_app import create_gradio_app
from modules.webui.training_manager import TrainingManager


def main():
    parser = argparse.ArgumentParser(description="OneTrainer WebUI")
    parser.add_argument("--host", type=str, default=os.environ.get("OT_WEBUI_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("OT_WEBUI_PORT", "7860")))
    parser.add_argument("--data-dir", type=str, default=os.environ.get("OT_WEBUI_DATA_DIR", ".ot-webui"))
    parser.add_argument("--token", type=str, default=os.environ.get("OT_WEBUI_TOKEN", ""))
    parser.add_argument("--ui", type=str, default=os.environ.get("OT_WEBUI_UI", "gradio"), choices=["gradio", "classic"])
    args = parser.parse_args()

    if args.ui == "classic":
        app = create_app(
            data_dir=args.data_dir,
            token=args.token,
        )
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
        return

    import gradio as gr

    project_root = str(Path(__file__).absolute().parent.parent)
    manager = TrainingManager(data_dir=args.data_dir)
    demo = create_gradio_app(manager=manager, project_root=project_root)

    auth = None
    if args.token:
        def auth_fn(username: str, password: str) -> bool:
            return bool(username) and password == args.token

        auth = auth_fn

    demo.launch(
        server_name=args.host,
        server_port=args.port,
        auth=auth,
        show_api=False,
        quiet=True,
    )


if __name__ == "__main__":
    main()
