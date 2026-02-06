# OneTrainer WebUI 部署指南

本指南将帮助您在本地或远程服务器上部署和运行 OneTrainer WebUI。

## 快速开始

### 自动安装与启动
在 Linux/macOS 环境下，您可以使用以下脚本一键完成环境配置并启动：

```bash
./start-webui.sh
```

该脚本会自动执行以下操作：
1. 创建一个独立的虚拟环境（默认在 `.ot-webui-venv` 目录下）。
2. 安装所有必要的依赖（包括 `gradio` 和 WebUI 相关库）。
3. 启动 WebUI 服务（默认地址：`http://127.0.0.1:7860`）。

## 运行模式

### 1. 本地启动（默认）
直接运行启动脚本即可：
```bash
./start-webui.sh
```

### 2. 远程服务器启动（无头模式）
如果您在远程服务器（如云服务器、实验室工作站）上运行，可以使用以下命令：
```bash
./start-webui.sh --host 0.0.0.0 --port 7860
```
*   `--host 0.0.0.0`: 允许外部通过 IP 访问。
*   `--port 7860`: 指定访问端口。

### 3. 带 Token 认证
为了安全起见，您可以设置访问令牌：
```bash
./start-webui.sh --token your_secret_token
```
启动后，访问页面时会提示输入密码（用户名为任意，密码为 `your_secret_token`）。

## 目录结构说明

*   `.ot-webui-venv/`: WebUI 专用的 Python 虚拟环境，包含所有 UI 依赖。
*   `.ot-webui-conda/`: 如果您使用 Conda，这里存储的是环境的核心库和二进制文件。
*   `.ot-webui/`: 存储 WebUI 的运行历史、缓存和临时配置文件。

## 常见问题

### 端口占用
如果提示 `OSError: [Errno 98] Address already in use`，请检查是否有残留的 WebUI 进程：
```bash
ps aux | grep webui.py
kill -9 <PID>
```

### 环境隔离
WebUI 使用独立的虚拟环境，不会干扰 OneTrainer 本身的训练环境。如果您需要重新安装 WebUI 依赖，可以安全地删除 `.ot-webui-venv` 目录并重新运行 `start-webui.sh`。
