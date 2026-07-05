# JoyAgent

> **Anthropic-native Multi-Agent Coding System** — 基于 FastAPI + LangGraph 的自主编程智能体系统，支持多 Agent 协作、沙箱安全执行、消息持久化。

## 目录

- [系统架构](#系统架构)
- [前置条件](#前置条件)
- [第一步：构建 Docker 镜像](#第一步构建-docker-镜像)
- [第二步：配置环境变量](#第二步配置环境变量)
- [第三步：启动容器](#第三步启动容器)
- [第四步：查看日志](#第四步查看日志)
- [第五步：验证服务](#第五步验证服务)
- [更新代码后如何重跑](#更新代码后如何重跑)
- [常用运维命令](#常用运维命令)
- [安全架构](#安全架构)

---

## 系统架构

JoyAgent 由两个 Docker 容器协作运行：

| 容器 | 角色 | 说明 |
|------|------|------|
| `joyagent` | 主容器（常驻） | FastAPI 服务（:8000）、Multi-Agent 协作、LLM 调用 |
| `joyagent-sandbox` | 沙箱容器（按需） | 临时创建，执行 Agent 生成的代码后立即销毁，六层安全防护 |

5 个协作 Agent：**Router → Planner → Coder → Tester → Reviewer**，形成完整的编码工作流。

---

## 前置条件

| 条件 | 要求 |
|------|------|
| 操作系统 | Linux（CentOS 7 / Ubuntu 20.04+ / 任意 Linux 发行版） |
| Docker | 已安装且 daemon 运行中 |
| Python | 3.11+（仅构建阶段需要） |
| API Key | Anthropic API Key（或兼容的 DeepSeek 端点） |

确认 Docker 可用：

```bash
sudo systemctl start docker
sudo systemctl enable docker
docker ps
```

---

## 第一步：构建 Docker 镜像

### 1.1 克隆项目

```bash
git clone https://github.com/your-org/joyagent.git
cd joyagent
```

### 1.2 构建 Sandbox 镜像

Sandbox 是轻量级的安全执行环境（约 137MB），只需构建一次：

```bash
docker build -t joyagent-sandbox:latest -f sandbox_config/Dockerfile .
```

> **说明**：Sandbox 基于 `python:3.11-slim`，创建了非 root 用户 `sandbox`，预装了 `pytest`、`pytest-json-report`、`pytest-timeout`。Agent 生成的所有代码在此容器中以最小权限运行。

### 1.3 构建 JoyAgent 主镜像

```bash
docker build -t joyagent:latest .
```

> **说明**：主镜像约 5.8GB，包含：
> - FastAPI + Uvicorn 服务
> - LangGraph 工作流引擎
> - Anthropic SDK + sentence-transformers 本地嵌入模型
> - GitPython、ChromaDB、Redis 等全套依赖
> - `/models/` 目录下的离线模型文件

### 1.4 确认镜像构建成功

```bash
docker images | grep joyagent
```

期望输出：

```
joyagent           latest         xxx   N days ago   5.83GB
joyagent-sandbox   latest         xxx   N days ago   137MB
```

---

## 第二步：配置环境变量

复制并编辑环境变量文件：

```bash
cp .env.example .env
```

编辑 `.env`，填入你的 API 配置：

```env
# === Anthropic API（必填）===
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxx

# === DeepSeek 兼容端点（使用 DeepSeek 模型时设置）===
# ANTHROPIC_BASE_URL=https://api.deepseek.com/v1

# === 默认模型 ===
DEFAULT_MODEL=claude-sonnet-4-5-20250901

# === 备选模型（主模型过载时自动切换）===
# FALLBACK_MODEL=claude-haiku-4-5-20251001
```

| 变量 | 必填 | 说明 |
|------|------|------|
| `ANTHROPIC_API_KEY` | ✅ 是 | Anthropic API 密钥 |
| `ANTHROPIC_BASE_URL` | 否 | DeepSeek 兼容端点地址 |
| `DEFAULT_MODEL` | 否 | 默认使用的模型 ID |
| `FALLBACK_MODEL` | 否 | 主模型过载时自动切换的备选模型 |

---

## 第三步：启动容器

### 3.1 标准启动（后台运行）

```bash
docker run -d \
  --name joyagent \
  -p 8000:8000 \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v /mnt/hgfs/joyagent:/app \
  joyagent:latest
```

### 参数说明

| 参数 | 含义 |
|------|------|
| `docker run` | 创建并启动一个新容器 |
| `-d` | **D**etached 模式：容器在后台运行，关闭终端服务不挂 |
| `--name joyagent` | 容器命名为 `joyagent`，方便后续操作 |
| `-p 8000:8000` | **P**ort 映射：容器 8000 → 宿主机 8000 |
| `-v /var/run/docker.sock:/var/run/docker.sock` | 挂载 Docker socket，使容器内可操控宿主机 Docker 创建/销毁 Sandbox |
| `-v /mnt/hgfs/joyagent:/app` | 挂载项目代码目录到容器 `/app`。**请替换为你的实际项目路径**（如 `/root/joyagent`、`/home/user/joyagent`） |
| `joyagent:latest` | 使用的镜像名 |

> ⚠️ **重要**：`/mnt/hgfs/joyagent` 是 VMware 共享文件夹路径。如果你的项目在其他位置，请替换为实际路径，例如：
> ```bash
> -v /home/yourname/joyagent:/app
> ```

### 3.2 前台启动（调试用）

如果想去掉 `-d` 在前台运行（日志直接刷在终端，Ctrl+C 停止）：

```bash
docker run \
  --name joyagent \
  -p 8000:8000 \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v /mnt/hgfs/joyagent:/app \
  joyagent:latest
```

---

## 第四步：查看日志

```bash
# 实时日志
docker logs -f joyagent

# 最近 50 行
docker logs joyagent | tail -50
```

启动成功后应该看到：

```
[startup] cleaned N cached .pyc/__pycache__
[OK] Registered 14 tools: ...
[OK] ToolRegistry initialized.
[mailbox] Agent 'router' registered (inbox handlers: ...)
[mailbox] Agent 'planner' registered (inbox handlers: ...)
[mailbox] Agent 'coder' registered (inbox handlers: ...)
[mailbox] Agent 'tester' registered (inbox handlers: ...)
[mailbox] Agent 'reviewer' registered (inbox handlers: ...)
[orchestrator] 5 agents registered: router, planner, coder, tester, reviewer
[router] Watcher started (background, task=Task-3)
[planner] Watcher started (background, task=Task-4)
[coder] Watcher started (background, task=Task-5)
[tester] Watcher started (background, task=Task-6)
[reviewer] Watcher started (background, task=Task-7)
[orchestrator] All 5 agents started (background)
[OK] Multi-Agent system online (backend=file)
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

---

## 第五步：验证服务

### 5.1 查看 API 文档

浏览器打开：`http://宿主机IP:8000/docs`

可看到 Swagger UI 自动生成的完整 API 文档。

### 5.2 查看 Multi-Agent 状态

```bash
curl http://宿主机IP:8000/api/multi-agent/status
```

期望返回 `"is_running": true` 和 5 个 Agent 的统计信息。

### 5.3 测试 Multi-Agent 请求

```bash
curl -X POST http://宿主机IP:8000/api/multi-agent \
  -H "Content-Type: application/json" \
  -d '{"message": "创建一个 FastAPI health check 端点"}'
```

返回的 `correlation_id` 可用于审计追踪：

```bash
curl "http://宿主机IP:8000/api/multi-agent/audit?correlation_id=user_abc12345"
```

### 5.4 测试单 Agent 请求（兼容旧接口）

```bash
curl -X POST http://宿主机IP:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "读取 main.py 文件"}'
```

---

## 更新代码后如何重跑

因为 `-v` 挂载了代码目录，**修改 Python 代码后不需要重新构建镜像**，只需重启容器：

```bash
docker restart joyagent
```

但如果修改了 **Dockerfile**、**pyproject.toml** 或 **models/ 目录**，需要重新构建镜像：

```bash
cd /path/to/joyagent

# 重新构建镜像
docker build -t joyagent:latest .

# 删除旧容器并启动新容器
docker rm -f joyagent
docker run -d \
  --name joyagent \
  -p 8000:8000 \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v /mnt/hgfs/joyagent:/app \
  joyagent:latest
```

---

## 常用运维命令

| 命令 | 作用 |
|------|------|
| `docker ps \| grep joyagent` | 查看容器是否在运行 |
| `docker logs -f joyagent` | 实时查看日志 |
| `docker restart joyagent` | 重启容器 |
| `docker stop joyagent` | 停止容器 |
| `docker rm joyagent` | 删除容器（需先停止） |
| `docker rm -f joyagent` | 强制删除容器（运行中也删） |
| `docker exec -it joyagent bash` | 进入容器内部调试 |
| `docker port joyagent` | 查看端口映射 |

---

## 安全架构

```
┌──────────────────────────────────────────────────────────┐
│ 宿主机（Linux）                                           │
│                                                          │
│  ┌──────────────────────────────┐  ┌──────────────────┐  │
│  │  joyagent 容器                │  │  sandbox 容器     │  │
│  │  - FastAPI :8000              │  │  (按需创建/销毁)   │  │
│  │  - Multi-Agent (Phase 7)      │  │                   │  │
│  │  - 消息持久化 data/mailbox/    │  │  /workspace       │  │
│  │  - 模型 /models/              │  │  (bind mount)     │  │
│  │                              │  │                   │  │
│  │  docker.sock ─────────────────┼──▶ 创建 sandbox      │  │
│  │  /app ← 共享目录              │  │                   │  │
│  └──────────────────────────────┘  └──────────────────┘  │
│           │                            │                 │
│           └──────── bind mount ────────┘                 │
│                    /mnt/hgfs/joyagent                    │
└──────────────────────────────────────────────────────────┘
```

- **JoyAgent 容器**：常驻运行，负责 API、多 Agent 协作、LLM 调用
- **Sandbox 容器**：临时创建，执行 Agent 生成的代码后立即销毁。六层安全防护：
  1. **进程隔离** — 独立容器运行
  2. **资源限制** — CPU/内存配额
  3. **网络隔离** — 无外网访问
  4. **只读根** — 文件系统保护
  5. **权限限制** — 非 root 用户 `sandbox`
  6. **超时控制** — pytest-timeout 防止死循环
- **代码同步**：两个容器通过 bind mount 共享同一份代码目录，Sandbox 的修改即时可见
- **消息持久化**：`data/mailbox/` 目录存储在每个 Agent 的 JSON 文件中，容器重启后自动恢复

---

## 技术栈

| 层级 | 技术 |
|------|------|
| 框架 | FastAPI + Uvicorn |
| LLM | Anthropic API（兼容 DeepSeek） |
| 工作流 | LangGraph |
| 向量存储 | ChromaDB + sentence-transformers |
| 容器化 | Docker（主容器 + 沙箱容器） |
| 消息持久化 | File / Redis（可切换） |
| 代码管理 | GitPython |
| Python | ≥ 3.11 |

---

## 更多文档

- [部署指南](linux_run_docs/01-deployment-guide.md)
- [故障排除](linux_run_docs/02-troubleshooting.md)
- [快速参考](linux_run_docs/03-quick-reference.md)
- [部署经验记录](linux_run_docs/04-memory-deployment-lessons.md)
- [Docker 启动指南](linux_run_docs/05-docker-startup-guide.md)

各阶段的开发文档见 `dev_md/` 目录。
