#!/usr/bin/env bash

set -e

source "${BASH_SOURCE[0]%/*}/lib.include.sh"

if [[ -z "${HF_HUB_DISABLE_XET+x}" ]]; then
    export HF_HUB_DISABLE_XET=1
fi

OT_WEBUI_VENV="${OT_WEBUI_VENV:-.ot-webui-venv}"

if [[ -z "${OT_CONDA_CMD+x}" ]]; then
    if [[ -n "${CONDA_EXE:-}" ]]; then
        export OT_CONDA_CMD="${CONDA_EXE}"
    elif command -v conda >/dev/null 2>&1; then
        export OT_CONDA_CMD="$(command -v conda)"
    fi
fi

if [[ ! -f "${OT_WEBUI_VENV}/bin/activate" ]]; then
    print "Creating WebUI venv in \"${OT_WEBUI_VENV}\"..."
    run_python -m venv "${OT_WEBUI_VENV}"
fi

source "${OT_WEBUI_VENV}/bin/activate"

python -m pip install --upgrade pip setuptools
python -m pip install -r requirements-webui.txt

python "scripts/webui.py" "$@"
