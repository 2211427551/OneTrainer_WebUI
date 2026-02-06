import json
import sys
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse

from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.webui.training_manager import TrainingManager


def _get_token_from_request(request: Request) -> str:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return request.query_params.get("token", "")


def _enforce_token(app: FastAPI, request: Request) -> None:
    token: str = app.state.token
    if not token:
        return
    got = _get_token_from_request(request)
    if got != token:
        raise HTTPException(status_code=401, detail="Unauthorized")


def create_app(data_dir: str, token: str = "") -> FastAPI:
    app = FastAPI()
    app.state.token = token or ""
    app.state.manager = TrainingManager(data_dir=data_dir)

    @app.middleware("http")
    async def _auth_middleware(request: Request, call_next):
        _enforce_token(app, request)
        return await call_next(request)

    @app.get("/", response_class=HTMLResponse)
    async def index():
        html = """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>OneTrainer WebUI</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 16px; }
    textarea { width: 100%; min-height: 220px; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    .row3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; }
    .field { display: flex; flex-direction: column; gap: 4px; }
    input, select { padding: 6px; }
    button { margin-right: 8px; margin-bottom: 8px; }
    pre { background: #f6f8fa; padding: 12px; overflow: auto; }
    .muted { color: #666; }
  </style>
</head>
<body>
  <h2>OneTrainer WebUI</h2>
  <div class="muted">建议通过 SSH 端口转发访问；如启用 token，请以 ?token=... 打开本页面。</div>

  <h3>状态</h3>
  <pre id="state">loading...</pre>
  <div>
    <button onclick="startTrain()">开始训练</button>
    <button onclick="stopTrain()">停止</button>
    <button onclick="backupTrain()">备份</button>
    <button onclick="saveTrain()">保存</button>
  </div>

  <h3>常用参数</h3>
  <div class="row3">
    <div class="field">
      <label>training_method</label>
      <select id="q_training_method"></select>
    </div>
    <div class="field">
      <label>model_type</label>
      <select id="q_model_type"></select>
    </div>
    <div class="field">
      <label>base_model_name</label>
      <input id="q_base_model_name" placeholder="stable-diffusion-v1-5/stable-diffusion-v1-5" />
    </div>
    <div class="field">
      <label>workspace_dir</label>
      <input id="q_workspace_dir" placeholder="workspace/run" />
    </div>
    <div class="field">
      <label>cache_dir</label>
      <input id="q_cache_dir" placeholder="workspace-cache/run" />
    </div>
    <div class="field">
      <label>output_model_destination</label>
      <input id="q_output_model_destination" placeholder="models/model.safetensors" />
    </div>
    <div class="field">
      <label>resolution</label>
      <input id="q_resolution" placeholder="512" />
    </div>
    <div class="field">
      <label>epochs</label>
      <input id="q_epochs" type="number" min="1" step="1" />
    </div>
    <div class="field">
      <label>batch_size</label>
      <input id="q_batch_size" type="number" min="1" step="1" />
    </div>
    <div class="field">
      <label>gradient_accumulation_steps</label>
      <input id="q_grad_acc" type="number" min="1" step="1" />
    </div>
    <div class="field">
      <label>learning_rate</label>
      <input id="q_learning_rate" type="number" step="any" />
    </div>
    <div class="field">
      <label>tensorboard</label>
      <select id="q_tensorboard">
        <option value="true">true</option>
        <option value="false">false</option>
      </select>
    </div>
    <div class="field">
      <label>tensorboard_port</label>
      <input id="q_tensorboard_port" type="number" min="1" step="1" />
    </div>
    <div class="field">
      <label>multi_gpu</label>
      <select id="q_multi_gpu">
        <option value="false">false</option>
        <option value="true">true</option>
      </select>
    </div>
    <div class="field">
      <label>device_indexes</label>
      <input id="q_device_indexes" placeholder="0,1" />
    </div>
  </div>
  <div>
    <button onclick="syncQuickFromConfig()">从 config 同步到表单</button>
    <button onclick="applyQuickToConfig()">把表单写回 config</button>
  </div>

  <h3>配置</h3>
  <div class="row">
    <div>
      <div>config.json</div>
      <textarea id="config"></textarea>
      <div>
        <button onclick="saveConfig()">保存 config</button>
        <input type="file" id="configFile" />
        <button onclick="uploadConfig()">上传 config</button>
      </div>
    </div>
    <div>
      <div>secrets.json（可选）</div>
      <textarea id="secrets"></textarea>
      <div>
        <button onclick="saveSecrets()">保存 secrets</button>
        <input type="file" id="secretsFile" />
        <button onclick="uploadSecrets()">上传 secrets</button>
      </div>
    </div>
  </div>

  <h3>数据集检查</h3>
  <div>
    <button onclick="checkDataset()">检查数据集</button>
  </div>
  <pre id="dataset">not checked</pre>

  <h3>训练日志</h3>
  <div class="row">
    <div class="field">
      <label>run_id</label>
      <select id="runSelect" onchange="refreshLog()"></select>
    </div>
    <div class="field">
      <label>tail bytes</label>
      <input id="logBytes" type="number" min="1024" step="1024" value="65536" />
    </div>
  </div>
  <pre id="log">no log</pre>

<script>
  const token = new URLSearchParams(window.location.search).get('token') || '';
  const withToken = (url) => token ? (url + (url.includes('?') ? '&' : '?') + 'token=' + encodeURIComponent(token)) : url;

  async function loadEnums() {
    const r = await fetch(withToken('/api/enums'));
    const j = await r.json();
    const tm = document.getElementById('q_training_method');
    const mt = document.getElementById('q_model_type');
    tm.innerHTML = '';
    mt.innerHTML = '';
    (j.training_method || []).forEach(v => {
      const o = document.createElement('option');
      o.value = v; o.textContent = v;
      tm.appendChild(o);
    });
    (j.model_type || []).forEach(v => {
      const o = document.createElement('option');
      o.value = v; o.textContent = v;
      mt.appendChild(o);
    });
  }

  function safeParse(text, fallback) {
    try { return JSON.parse(text || ''); } catch { return fallback; }
  }

  function syncQuickFromConfig() {
    const cfg = safeParse(document.getElementById('config').value, {});
    document.getElementById('q_training_method').value = cfg.training_method || 'FINE_TUNE';
    document.getElementById('q_model_type').value = cfg.model_type || 'STABLE_DIFFUSION_15';
    document.getElementById('q_base_model_name').value = cfg.base_model_name || '';
    document.getElementById('q_workspace_dir').value = cfg.workspace_dir || '';
    document.getElementById('q_cache_dir').value = cfg.cache_dir || '';
    document.getElementById('q_output_model_destination').value = cfg.output_model_destination || '';
    document.getElementById('q_resolution').value = cfg.resolution || '';
    document.getElementById('q_epochs').value = cfg.epochs ?? '';
    document.getElementById('q_batch_size').value = cfg.batch_size ?? '';
    document.getElementById('q_grad_acc').value = cfg.gradient_accumulation_steps ?? '';
    document.getElementById('q_learning_rate').value = cfg.learning_rate ?? '';
    document.getElementById('q_tensorboard').value = String(cfg.tensorboard ?? true);
    document.getElementById('q_tensorboard_port').value = cfg.tensorboard_port ?? 6006;
    document.getElementById('q_multi_gpu').value = String(cfg.multi_gpu ?? false);
    document.getElementById('q_device_indexes').value = cfg.device_indexes || '';
  }

  function applyQuickToConfig() {
    const cfg = safeParse(document.getElementById('config').value, {});
    cfg.training_method = document.getElementById('q_training_method').value;
    cfg.model_type = document.getElementById('q_model_type').value;
    cfg.base_model_name = document.getElementById('q_base_model_name').value;
    cfg.workspace_dir = document.getElementById('q_workspace_dir').value;
    cfg.cache_dir = document.getElementById('q_cache_dir').value;
    cfg.output_model_destination = document.getElementById('q_output_model_destination').value;
    cfg.resolution = document.getElementById('q_resolution').value;
    cfg.epochs = Number(document.getElementById('q_epochs').value || cfg.epochs || 1);
    cfg.batch_size = Number(document.getElementById('q_batch_size').value || cfg.batch_size || 1);
    cfg.gradient_accumulation_steps = Number(document.getElementById('q_grad_acc').value || cfg.gradient_accumulation_steps || 1);
    cfg.learning_rate = Number(document.getElementById('q_learning_rate').value || cfg.learning_rate || 0);
    cfg.tensorboard = (document.getElementById('q_tensorboard').value === 'true');
    cfg.tensorboard_port = Number(document.getElementById('q_tensorboard_port').value || cfg.tensorboard_port || 6006);
    cfg.multi_gpu = (document.getElementById('q_multi_gpu').value === 'true');
    cfg.device_indexes = document.getElementById('q_device_indexes').value;
    document.getElementById('config').value = JSON.stringify(cfg, null, 2);
  }

  async function loadConfig() {
    const r = await fetch(withToken('/api/config'));
    if (r.ok) {
      const j = await r.json();
      document.getElementById('config').value = JSON.stringify(j || {}, null, 2);
    }
    const r2 = await fetch(withToken('/api/secrets'));
    if (r2.ok) {
      const j2 = await r2.json();
      document.getElementById('secrets').value = JSON.stringify(j2 || {}, null, 2);
    }
    syncQuickFromConfig();
  }

  async function refreshState() {
    const r = await fetch(withToken('/api/state'));
    const j = await r.json();
    document.getElementById('state').textContent = JSON.stringify(j, null, 2);
    await refreshRuns();
    await refreshLog();
  }

  async function refreshRuns() {
    const r = await fetch(withToken('/api/runs'));
    if (!r.ok) return;
    const runs = await r.json();
    const sel = document.getElementById('runSelect');
    const current = sel.value;
    sel.innerHTML = '';
    (runs || []).forEach(run => {
      const o = document.createElement('option');
      o.value = run.run_id;
      o.textContent = run.run_id + (run.has_log ? '' : ' (no log)');
      sel.appendChild(o);
    });
    if (current) sel.value = current;
  }

  async function refreshLog() {
    const sel = document.getElementById('runSelect');
    const runId = sel.value;
    if (!runId) return;
    const bytes = Number(document.getElementById('logBytes').value || 65536);
    const r = await fetch(withToken(`/api/run/${encodeURIComponent(runId)}/log?bytes=${bytes}`));
    if (!r.ok) return;
    const j = await r.json();
    document.getElementById('log').textContent = j.text || '';
  }

  async function checkDataset() {
    const r = await fetch(withToken('/api/dataset/check'));
    const j = await r.json();
    document.getElementById('dataset').textContent = JSON.stringify(j, null, 2);
  }

  async function saveConfig() {
    const text = document.getElementById('config').value;
    const config = JSON.parse(text || '{}');
    await fetch(withToken('/api/config'), { method: 'PUT', headers: { 'content-type': 'application/json' }, body: JSON.stringify(config) });
    await loadConfig();
  }

  async function saveSecrets() {
    const text = document.getElementById('secrets').value;
    const secrets = JSON.parse(text || '{}');
    await fetch(withToken('/api/secrets'), { method: 'PUT', headers: { 'content-type': 'application/json' }, body: JSON.stringify(secrets) });
    await loadConfig();
  }

  async function uploadConfig() {
    const fileInput = document.getElementById('configFile');
    if (!fileInput.files.length) return;
    const fd = new FormData();
    fd.append('file', fileInput.files[0]);
    await fetch(withToken('/api/config/upload'), { method: 'POST', body: fd });
    await loadConfig();
  }

  async function uploadSecrets() {
    const fileInput = document.getElementById('secretsFile');
    if (!fileInput.files.length) return;
    const fd = new FormData();
    fd.append('file', fileInput.files[0]);
    await fetch(withToken('/api/secrets/upload'), { method: 'POST', body: fd });
    await loadConfig();
  }

  async function startTrain() {
    await fetch(withToken('/api/train/start'), { method: 'POST' });
    await refreshState();
  }
  async function stopTrain() {
    await fetch(withToken('/api/train/stop'), { method: 'POST' });
    await refreshState();
  }
  async function backupTrain() {
    await fetch(withToken('/api/train/backup'), { method: 'POST' });
    await refreshState();
  }
  async function saveTrain() {
    await fetch(withToken('/api/train/save'), { method: 'POST' });
    await refreshState();
  }

  loadEnums();
  loadConfig();
  refreshState();
  setInterval(refreshState, 1000);
</script>
</body>
</html>
        """
        return HTMLResponse(content=html)

    @app.get("/api/state")
    async def api_state():
        return app.state.manager.get_state()

    @app.get("/api/config")
    async def api_get_config():
        return app.state.manager.load_config() or {}

    @app.put("/api/config")
    async def api_put_config(payload: dict[str, Any] = Body(...)):
        app.state.manager.save_config(payload)
        return {"ok": True}

    @app.post("/api/config/upload")
    async def api_upload_config(file: UploadFile = File(...)):
        raw = await file.read()
        data = json.loads(raw.decode("utf-8"))
        if not isinstance(data, dict):
            raise HTTPException(status_code=400, detail="config.json must be an object")
        app.state.manager.save_config(data)
        return {"ok": True}

    @app.get("/api/secrets")
    async def api_get_secrets():
        return app.state.manager.load_secrets() or {}

    @app.put("/api/secrets")
    async def api_put_secrets(payload: dict[str, Any] = Body(...)):
        app.state.manager.save_secrets(payload)
        return {"ok": True}

    @app.post("/api/secrets/upload")
    async def api_upload_secrets(file: UploadFile = File(...)):
        raw = await file.read()
        data = json.loads(raw.decode("utf-8"))
        if not isinstance(data, dict):
            raise HTTPException(status_code=400, detail="secrets.json must be an object")
        app.state.manager.save_secrets(data)
        return {"ok": True}

    @app.post("/api/train/start")
    async def api_train_start():
        project_root = str(Path(__file__).absolute().parents[2])
        return app.state.manager.start_training(project_root=project_root)

    @app.post("/api/train/stop")
    async def api_train_stop():
        return app.state.manager.stop_training()

    @app.post("/api/train/backup")
    async def api_train_backup():
        return app.state.manager.request_backup()

    @app.post("/api/train/save")
    async def api_train_save():
        return app.state.manager.request_save()

    @app.get("/api/runs")
    async def api_runs():
        return app.state.manager.list_runs()

    @app.get("/api/run/{run_id}/log")
    async def api_run_log(run_id: str, bytes: int = 65536):
        if bytes < 1024:
            bytes = 1024
        if bytes > 2_000_000:
            bytes = 2_000_000
        text = app.state.manager.read_log_tail(run_id=run_id, max_bytes=bytes)
        return {"run_id": run_id, "text": text}

    @app.get("/api/dataset/check")
    async def api_dataset_check():
        project_root = str(Path(__file__).absolute().parents[2])
        return app.state.manager.dataset_check(project_root=project_root)

    @app.get("/api/enums")
    async def api_enums():
        return {
            "training_method": [str(v) for v in TrainingMethod],
            "model_type": [str(v) for v in ModelType],
        }

    return app
