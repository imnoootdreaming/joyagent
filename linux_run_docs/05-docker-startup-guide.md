# JoyAgent Docker 启动指南

## 前置条件

| 条件 | 要求 |
|------|------|
| 操作系统 | Linux（CentOS 7 / Ubuntu 20.04+ / 任意 Linux 发行版） |
| Docker | 已安装且 daemon 运行中 |
| Docker 镜像 | `joyagent:latest` 和 `joyagent-sandbox:latest` 已构建 |
| 代码目录 | 宿主机上的项目根目录（通过 bind mount 挂入容器） |

---

## 第一步：确认 Docker 可用

```bash
sudo systemctl start docker
sudo systemctl enable docker
docker ps
```

看到容器列表即为正常。

---

## 第二步：确认镜像存在

```bash
docker images | grep joyagent
```

期望输出：

```
joyagent           latest         xxx   N days ago   5.83GB
joyagent-sandbox   latest         xxx   N days ago   137MB
```

如果镜像不存在，需要先构建：

```bash
# 构建 Sandbox 镜像（一次性）
docker build -t joyagent-sandbox:latest -f sandbox_config/Dockerfile .

# 构建 JoyAgent 主镜像
docker build -t joyagent:latest .
```

---

## 第三步：启动命令

```bash
docker run -d \
  --name joyagent \
  -p 8000:8000 \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v /mnt/hgfs/joyagent:/app \
  joyagent:latest
```

### 每一行命令的含义

| 参数 | 含义 |
|------|------|
| `docker run` | 创建并启动一个新容器 |
| `-d` | **D**etached 模式：容器在后台运行，终端关了服务不挂 |
| `--name joyagent` | 给容器起名叫 `joyagent`，方便后续 `docker logs joyagent` 等操作 |
| `-p 8000:8000` | **P**ort 映射：把容器的 8000 端口映射到宿主机的 8000 端口。本机浏览器访问 `http://宿主机IP:8000/docs` 就能看到 Swagger |
| `-v /var/run/docker.sock:/var/run/docker.sock` | **V**olume 挂载：把宿主机的 Docker daemon socket 挂进容器。这样容器内的 JoyAgent 可以通过 `docker.sock` 操控宿主机 Docker，按需创建/销毁 Sandbox 容器来安全执行代码 |
| `-v /mnt/hgfs/joyagent:/app` | **V**olume 挂载：把宿主机项目目录挂到容器的 `/app`。这样 Sandbox 修改的代码文件能即时同步给 JoyAgent 容器，反之亦然。`/mnt/hgfs/joyagent` 是 VMware 共享文件夹路径，如果你的项目在其他位置请替换（如 `/root/joyagent`） |
| `joyagent:latest` | 使用的 Docker 镜像名 |

### 如果不想后台运行（调试用）

去掉 `-d`，日志直接刷在终端上，Ctrl+C 停止：

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

### 5.2 查看 Multi-Agent 状态

```bash
curl http://宿主机IP:8000/api/multi-agent/status
```

期望看到 `"is_running": true` 和 5 个 Agent 的统计信息。

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

### 5.4 测试 Phase 3 单 Agent（兼容旧接口）

```bash
curl -X POST http://宿主机IP:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "读取 main.py 文件"}'
```

---

## 常用运维命令

| 命令 | 作用 |
|------|------|
| `docker ps \| grep joyagent` | 查看容器是否在运行 |
| `docker logs -f joyagent` | 实时查看日志 |
| `docker restart joyagent` | 重启容器 |
| `docker stop joyagent` | 停止容器 |
| `docker rm joyagent` | 删除容器（停止后） |
| `docker rm -f joyagent` | 强制删除容器（运行中也删） |
| `docker exec -it joyagent bash` | 进入容器内部调试 |
| `docker port joyagent` | 查看端口映射 |

---

## 更新代码后如何重跑

因为 `-v` 挂载了代码目录，修改代码后**不需要重新构建镜像**，只需重启容器：

```bash
docker restart joyagent
```

但如果修改了 `Dockerfile`、`pyproject.toml` 或 `models/`，则需要重新构建：

```bash
cd /mnt/hgfs/joyagent
docker build -t joyagent:latest .
docker rm -f joyagent
docker run -d --name joyagent -p 8000:8000 \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v /mnt/hgfs/joyagent:/app \
  joyagent:latest
```

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
- **Sandbox 容器**：临时创建，执行 Agent 生成的代码后立即销毁。六层安全防护（进程隔离、资源限制、网络隔离、只读根、权限限制、超时控制）
- **代码同步**：两个容器通过 bind mount 共享同一份代码目录，Sandbox 的修改即时可见
- **消息持久化**：`data/mailbox/` 目录存储在每个 Agent 的 JSON 文件中，容器重启后自动恢复
