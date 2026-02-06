import json
import os
import pickle
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from modules.util.commands.TrainCommands import TrainCommands


class TrainingManager:
    def __init__(self, data_dir: str):
        self._data_dir = Path(data_dir).absolute()
        self._configs_dir = self._data_dir / "configs"
        self._runs_dir = self._data_dir / "runs"
        self._configs_dir.mkdir(parents=True, exist_ok=True)
        self._runs_dir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.RLock()
        self._process: subprocess.Popen | None = None
        self._log_file = None
        self._run_id: str | None = None
        self._callback_reader_thread: threading.Thread | None = None
        self._watcher_thread: threading.Thread | None = None
        self._last_progress_time: float | None = None
        self._last_global_step: int | None = None
        self._cpu_prev_total: int | None = None
        self._cpu_prev_idle: int | None = None

        self._state: dict[str, Any] = {
            "running": False,
            "run_id": None,
            "pid": None,
            "returncode": None,
            "status": "idle",
            "progress": {
                "epoch": 0,
                "epoch_step": 0,
                "epoch_sample": 0,
                "global_step": 0,
                "max_step": None,
                "max_epoch": None,
            },
            "speed": {
                "steps_per_sec": None,
                "sec_per_step": None,
            },
            "last_event_time": None,
            "tensorboard": None,
        }

    def _config_path(self) -> Path:
        return self._configs_dir / "config.json"

    def _secrets_path(self) -> Path:
        return self._configs_dir / "secrets.json"

    def load_config(self) -> dict[str, Any] | None:
        path = self._config_path()
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def save_config(self, config: dict[str, Any]) -> None:
        self._configs_dir.mkdir(parents=True, exist_ok=True)
        with self._config_path().open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)

    def load_secrets(self) -> dict[str, Any] | None:
        path = self._secrets_path()
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def save_secrets(self, secrets: dict[str, Any]) -> None:
        self._configs_dir.mkdir(parents=True, exist_ok=True)
        with self._secrets_path().open("w", encoding="utf-8") as f:
            json.dump(secrets, f, indent=4)

    def get_state(self) -> dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps(self._state))

    def list_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        runs: list[dict[str, Any]] = []
        if not self._runs_dir.exists():
            return runs

        for child in sorted(self._runs_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if not child.is_dir():
                continue
            run_id = child.name
            log_path = child / "train.log"
            config_path = child / "config.json"
            runs.append(
                {
                    "run_id": run_id,
                    "mtime": child.stat().st_mtime,
                    "has_log": log_path.exists(),
                    "has_config": config_path.exists(),
                }
            )
            if len(runs) >= limit:
                break
        return runs

    def get_current_run_id(self) -> str | None:
        with self._lock:
            return self._state.get("run_id")

    def read_log_tail(self, run_id: str, max_bytes: int = 262144) -> str:
        log_path = self._runs_dir / run_id / "train.log"
        if not log_path.exists():
            return ""
        size = log_path.stat().st_size
        start = max(0, size - max_bytes)
        with log_path.open("rb") as f:
            f.seek(start)
            data = f.read()
        try:
            return data.decode("utf-8", errors="replace")
        except Exception:
            return data.decode(errors="replace")

    def get_system_stats(self) -> dict[str, Any]:
        cpu_percent = self._read_cpu_percent()
        mem = self._read_mem_info()
        gpus = self._read_gpu_info()
        load1, load5, load15 = (None, None, None)
        try:
            load1, load5, load15 = os.getloadavg()
        except Exception:
            pass
        return {
            "cpu_percent": cpu_percent,
            "loadavg": {"1m": load1, "5m": load5, "15m": load15},
            "mem": mem,
            "gpus": gpus,
        }

    def list_checkpoints(self, project_root: str, limit: int = 10) -> list[dict[str, Any]]:
        config = self.load_config() or {}
        packed, _ = self._pack_config_files(project_root=project_root, config=config)

        result: list[dict[str, Any]] = []
        workspace_dir = str(packed.get("workspace_dir", "") or "")
        if workspace_dir:
            backups = self._resolve_path(project_root, workspace_dir) / "backup"
            if backups.exists() and backups.is_dir():
                for run_dir in sorted(backups.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
                    if not run_dir.is_dir():
                        continue
                    for f in sorted(run_dir.glob("**/*"), key=lambda p: p.stat().st_mtime, reverse=True):
                        if not f.is_file():
                            continue
                        if f.suffix.lower() not in {".safetensors", ".pt", ".bin"}:
                            continue
                        result.append(
                            {
                                "path": str(f),
                                "size_mb": round(f.stat().st_size / (1024 * 1024), 2),
                                "mtime": f.stat().st_mtime,
                            }
                        )
                        if len(result) >= limit:
                            return result

        out_path = str(packed.get("output_model_destination", "") or "")
        if out_path:
            p = self._resolve_path(project_root, out_path)
            if p.exists() and p.is_file():
                result.append(
                    {
                        "path": str(p),
                        "size_mb": round(p.stat().st_size / (1024 * 1024), 2),
                        "mtime": p.stat().st_mtime,
                    }
                )

        return result[:limit]

    def get_train_config_schema(self, project_root: str, force: bool = False) -> dict[str, Any]:
        cache_path = self._configs_dir / "train_config_schema.json"
        if not force and cache_path.exists():
            try:
                with cache_path.open("r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass

        cmd = [
            "./run-cmd.sh",
            "export_train_config_schema",
        ]
        env = self._clean_env_for_conda(dict(os.environ))
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(Path(project_root).absolute()),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
        except Exception as e:
            return {"ok": False, "error": str(e)}

        if proc.returncode != 0:
            return {"ok": False, "error": proc.stderr[-2000:]}

        try:
            raw = proc.stdout.strip()
            schema = None
            if raw:
                try:
                    schema = json.loads(raw)
                except Exception:
                    for line in reversed(raw.splitlines()):
                        line = line.strip()
                        if line.startswith("{") and line.endswith("}"):
                            try:
                                schema = json.loads(line)
                                break
                            except Exception:
                                continue
            if schema is None:
                raise ValueError("invalid schema output")
        except Exception:
            schema = {"ok": False, "error": "invalid schema output"}

        if isinstance(schema, dict):
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with cache_path.open("w", encoding="utf-8") as f:
                    json.dump(schema, f, indent=2, ensure_ascii=False)
            except Exception:
                pass
        return schema

    def dataset_check(self, project_root: str) -> dict[str, Any]:
        config = self.load_config() or {}
        packed, issues = self._pack_config_files(project_root=project_root, config=config)

        concept_issues: list[str] = []
        concepts_info: list[dict[str, Any]] = []
        concepts = packed.get("concepts") or []
        if not isinstance(concepts, list):
            concepts = []
            concept_issues.append("config.concepts 不是数组")

        for idx, concept in enumerate(concepts):
            if not isinstance(concept, dict):
                concept_issues.append(f"concept[{idx}] 不是对象")
                continue

            c_path = str(concept.get("path", "") or "")
            include_sub = bool(concept.get("include_subdirectories", False))
            resolved = self._resolve_path(project_root, c_path) if c_path else None
            exists = bool(resolved and resolved.exists())

            image_count = None
            if exists and resolved and resolved.is_dir():
                image_count = self._count_images_in_dir(resolved, recursive=include_sub)

            prompt_source = ""
            prompt_path = ""
            text_cfg = concept.get("text") or {}
            if isinstance(text_cfg, dict):
                prompt_source = str(text_cfg.get("prompt_source", "") or "")
                prompt_path = str(text_cfg.get("prompt_path", "") or "")

            prompt_exists = None
            if prompt_path:
                prompt_resolved = self._resolve_path(project_root, prompt_path)
                prompt_exists = prompt_resolved.exists()

            concepts_info.append(
                {
                    "index": idx,
                    "name": concept.get("name", ""),
                    "enabled": bool(concept.get("enabled", True)),
                    "type": concept.get("type", ""),
                    "path": c_path,
                    "path_resolved": str(resolved) if resolved else None,
                    "path_exists": exists,
                    "image_count": image_count,
                    "prompt_source": prompt_source,
                    "prompt_path": prompt_path,
                    "prompt_exists": prompt_exists,
                }
            )

            if c_path and not exists:
                concept_issues.append(f"concept[{idx}] 路径不存在：{c_path}")
            if prompt_exists is False:
                concept_issues.append(f"concept[{idx}] prompt_path 不存在：{prompt_path}")

        return {
            "issues": issues + concept_issues,
            "concepts": concepts_info,
        }

    def start_training(self, project_root: str) -> dict[str, Any]:
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                raise RuntimeError("Training is already running")

            config = self.load_config()
            if not config:
                raise RuntimeError("Missing config.json. Upload or paste a config first.")

            secrets = self.load_secrets()
            packed_config, pack_issues = self._pack_config_files(project_root=project_root, config=config)
            if pack_issues:
                self._state["status"] = "config warning: " + "; ".join(pack_issues[:3])

            run_id = uuid.uuid4().hex
            run_dir = (self._runs_dir / run_id)
            run_dir.mkdir(parents=True, exist_ok=False)

            run_config_path = run_dir / "config.json"
            run_secrets_path = run_dir / "secrets.json"
            callback_path = run_dir / "callbacks.pickle"
            command_path = run_dir / "commands.pickle"
            log_path = run_dir / "train.log"

            with run_config_path.open("w", encoding="utf-8") as f:
                json.dump(packed_config, f, indent=4)
            if secrets:
                with run_secrets_path.open("w", encoding="utf-8") as f:
                    json.dump(secrets, f, indent=4)

            command_path.parent.mkdir(parents=True, exist_ok=True)
            command_path.write_bytes(b"")

            args = [
                "./run-cmd.sh",
                "train_remote",
                "--config-path",
                str(run_config_path),
                "--callback-path",
                str(callback_path),
                "--command-path",
                str(command_path),
            ]
            if secrets and run_secrets_path.exists():
                args.extend(["--secrets-path", str(run_secrets_path)])

            log_file = log_path.open("ab", buffering=0)
            self._log_file = log_file

            env = self._clean_env_for_conda(dict(os.environ))
            if "HF_HUB_DISABLE_XET" not in env:
                env["HF_HUB_DISABLE_XET"] = "1"

            self._process = subprocess.Popen(
                args,
                cwd=str(Path(project_root).absolute()),
                env=env,
                stdout=log_file,
                stderr=log_file,
            )
            self._run_id = run_id
            self._state.update(
                {
                    "running": True,
                    "run_id": run_id,
                    "pid": self._process.pid,
                    "returncode": None,
                    "status": "starting",
                    "last_event_time": time.time(),
                    "tensorboard": self._extract_tensorboard_info(packed_config),
                }
            )
            self._last_progress_time = None
            self._last_global_step = None
            self._state["speed"] = {"steps_per_sec": None, "sec_per_step": None}

            self._callback_reader_thread = threading.Thread(
                target=self._callback_reader_loop,
                args=(callback_path,),
                daemon=True,
            )
            self._callback_reader_thread.start()

            self._watcher_thread = threading.Thread(
                target=self._watcher_loop,
                daemon=True,
            )
            self._watcher_thread.start()

            return self.get_state()

    def _watcher_loop(self) -> None:
        while True:
            with self._lock:
                proc = self._process
                run_id = self._run_id
            if proc is None or run_id is None:
                return

            rc = proc.poll()
            if rc is None:
                time.sleep(0.5)
                continue

            with self._lock:
                if self._process is proc:
                    self._state["running"] = False
                    self._state["returncode"] = rc
                    if rc == 0:
                        self._state["status"] = "finished"
                    else:
                        self._state["status"] = "failed"
                    self._state["last_event_time"] = time.time()
                    self._process = None
                    if self._log_file is not None:
                        try:
                            self._log_file.close()
                        except Exception:
                            pass
                        self._log_file = None
                    self._run_id = None
            return

    def stop_training(self) -> dict[str, Any]:
        self._send_command(lambda c: c.stop())
        return self.get_state()

    def request_backup(self) -> dict[str, Any]:
        self._send_command(lambda c: c.backup())
        return self.get_state()

    def request_save(self) -> dict[str, Any]:
        self._send_command(lambda c: c.save())
        return self.get_state()

    def _send_command(self, build: Any) -> None:
        with self._lock:
            run_id = self._run_id
            proc = self._process
        if proc is None or run_id is None or proc.poll() is not None:
            raise RuntimeError("No running training job")

        command_path = self._runs_dir / run_id / "commands.pickle"
        commands = TrainCommands()
        build(commands)

        tmp_path = command_path.with_suffix(".pickle.write")
        with tmp_path.open("wb") as f:
            pickle.dump(commands, f)
        os.replace(tmp_path, command_path)

    def _callback_reader_loop(self, callback_path: Path) -> None:
        last_pos = 0
        while True:
            with self._lock:
                proc = self._process
            if proc is None:
                return

            try:
                if not callback_path.exists():
                    time.sleep(0.2)
                    continue

                size = callback_path.stat().st_size
                if size < last_pos:
                    last_pos = 0

                with callback_path.open("rb") as f:
                    f.seek(last_pos)
                    while True:
                        try:
                            event_name = pickle.load(f)
                            event_args = pickle.load(f)
                        except EOFError:
                            break

                        last_pos = f.tell()
                        self._handle_callback_event(event_name, event_args)
            except Exception:
                time.sleep(0.5)

            time.sleep(0.2)

    def _handle_callback_event(self, event_name: str, event_args: Any) -> None:
        now = time.time()
        with self._lock:
            self._state["last_event_time"] = now

            if event_name == "on_update_status":
                if event_args and isinstance(event_args[0], str):
                    self._state["status"] = event_args[0]
                return

            if event_name == "on_update_train_progress":
                if not event_args or len(event_args) < 3:
                    return
                train_progress = event_args[0]
                max_step = event_args[1]
                max_epoch = event_args[2]

                global_step = getattr(train_progress, "global_step", 0)
                if self._last_progress_time is not None and self._last_global_step is not None:
                    dt = now - self._last_progress_time
                    ds = global_step - self._last_global_step
                    if dt > 0 and ds > 0:
                        sps = ds / dt
                        self._state["speed"] = {
                            "steps_per_sec": round(sps, 4),
                            "sec_per_step": round(1.0 / sps, 4) if sps > 0 else None,
                        }
                self._last_progress_time = now
                self._last_global_step = global_step

                self._state["progress"] = {
                    "epoch": getattr(train_progress, "epoch", 0),
                    "epoch_step": getattr(train_progress, "epoch_step", 0),
                    "epoch_sample": getattr(train_progress, "epoch_sample", 0),
                    "global_step": global_step,
                    "max_step": max_step,
                    "max_epoch": max_epoch,
                }
                return

    def _extract_tensorboard_info(self, config: dict[str, Any]) -> dict[str, Any] | None:
        enabled = bool(config.get("tensorboard", False))
        if not enabled:
            return None

        port = config.get("tensorboard_port", 6006)
        expose = bool(config.get("tensorboard_expose", False))
        return {
            "enabled": True,
            "port": port,
            "expose": expose,
            "url_local": f"http://127.0.0.1:{port}",
        }

    def _resolve_path(self, project_root: str, maybe_path: str) -> Path:
        p = Path(maybe_path)
        if p.is_absolute():
            return p
        return (Path(project_root) / p).absolute()

    def _pack_config_files(self, project_root: str, config: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        packed = json.loads(json.dumps(config))
        issues: list[str] = []

        concept_file_name = str(packed.get("concept_file_name", "") or "")
        if packed.get("concepts") in [None, []]:
            if concept_file_name:
                concept_path = self._resolve_path(project_root, concept_file_name)
                if concept_path.exists():
                    try:
                        with concept_path.open("r", encoding="utf-8") as f:
                            packed["concepts"] = json.load(f)
                    except Exception:
                        issues.append(f"读取 concepts 文件失败：{concept_file_name}")
                else:
                    issues.append(f"concepts 文件不存在：{concept_file_name}")
            else:
                issues.append("缺少 concepts 且 concept_file_name 为空")

        sample_file_name = str(packed.get("sample_definition_file_name", "") or "")
        if packed.get("samples") in [None, []] and sample_file_name:
            sample_path = self._resolve_path(project_root, sample_file_name)
            if sample_path.exists():
                try:
                    with sample_path.open("r", encoding="utf-8") as f:
                        packed["samples"] = json.load(f)
                except Exception:
                    issues.append(f"读取 samples 文件失败：{sample_file_name}")
            else:
                issues.append(f"samples 文件不存在：{sample_file_name}")

        return packed, issues

    def _count_images_in_dir(self, path: Path, recursive: bool) -> int:
        exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
        if recursive:
            it = path.rglob("*")
        else:
            it = path.glob("*")
        count = 0
        for p in it:
            if p.is_file() and p.suffix.lower() in exts:
                count += 1
        return count

    def _read_mem_info(self) -> dict[str, Any]:
        mem_total_kb = None
        mem_avail_kb = None
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        mem_total_kb = int(line.split()[1])
                    elif line.startswith("MemAvailable:"):
                        mem_avail_kb = int(line.split()[1])
        except Exception:
            pass

        if mem_total_kb is None or mem_avail_kb is None or mem_total_kb <= 0:
            return {"total_gb": None, "used_gb": None, "used_percent": None}

        used_kb = mem_total_kb - mem_avail_kb
        used_percent = used_kb / mem_total_kb * 100.0
        return {
            "total_gb": round(mem_total_kb / (1024 * 1024), 2),
            "used_gb": round(used_kb / (1024 * 1024), 2),
            "used_percent": round(used_percent, 2),
        }

    def _read_cpu_percent(self) -> float | None:
        try:
            with open("/proc/stat", "r", encoding="utf-8") as f:
                line = f.readline()
        except Exception:
            return None

        parts = line.split()
        if len(parts) < 5 or parts[0] != "cpu":
            return None

        nums = [int(x) for x in parts[1:]]
        total = sum(nums)
        idle = nums[3] + (nums[4] if len(nums) > 4 else 0)

        if self._cpu_prev_total is None or self._cpu_prev_idle is None:
            self._cpu_prev_total = total
            self._cpu_prev_idle = idle
            return None

        dt_total = total - self._cpu_prev_total
        dt_idle = idle - self._cpu_prev_idle
        self._cpu_prev_total = total
        self._cpu_prev_idle = idle
        if dt_total <= 0:
            return None

        usage = (dt_total - dt_idle) / dt_total * 100.0
        return round(usage, 2)

    def _read_gpu_info(self) -> list[dict[str, Any]]:
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,power.limit",
            "--format=csv,noheader,nounits",
        ]
        try:
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, check=False)
        except Exception:
            return []

        if proc.returncode != 0:
            return []

        gpus: list[dict[str, Any]] = []
        for line in proc.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 9:
                continue
            gpus.append(
                {
                    "index": parts[0],
                    "name": parts[1],
                    "temp_c": parts[2],
                    "util_gpu": parts[3],
                    "util_mem": parts[4],
                    "mem_used_mb": parts[5],
                    "mem_total_mb": parts[6],
                    "power_w": parts[7],
                    "power_limit_w": parts[8],
                }
            )
        return gpus

    def _clean_env_for_conda(self, env: dict[str, str]) -> dict[str, str]:
        venv = env.pop("VIRTUAL_ENV", None)
        env.pop("PYTHONHOME", None)
        if venv:
            path = env.get("PATH", "")
            parts = [p for p in path.split(":") if p and not p.startswith(f"{venv}/")]
            env["PATH"] = ":".join(parts)
        return env
