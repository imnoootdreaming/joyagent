# JoyAgent Linux 部署指南

## 前置条件

| 条件 | 要求 |
|------|------|
| 操作系统 | CentOS 7 / Ubuntu 20.04+ / 任意 Linux |
| Python | 不需要手动安装（uv 自动管理） |
| Docker | 已安装且 daemon 运行中 |
| 项目代码 | 已同步到虚拟机（共享文件夹 / Git clone） |

---

## 第一步：确认 Docker 可用

```bash
sudo systemctl start docker
sudo systemctl enable docker
docker ps
```

看到容器列表即为正常。

---

## 第二步：安装 uv（Python 包管理器）

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

安装完成后使其生效：

```bash
source $HOME/.local/bin/env

# 一劳永逸：写入 bashrc
echo 'source $HOME/.local/bin/env' >> ~/.bashrc
```

---

## 第三步：配置环境变量

> **注意：** 如果使用 VMware 共享文件夹（hgfs），必须设置此变量。

```bash
echo 'export UV_LINK_MODE=copy' >> ~/.bashrc
source ~/.bashrc
```

然后进入项目目录，确认 `.env` 存在且有真实 API key：

```bash
cd /mnt/hgfs/joyagent/joyagent   # 或你的项目路径
cat .env
```

如果没有 `.env`，创建它：

```bash
cat > .env << 'EOF'
ANTHROPIC_API_KEY=你的真实API密钥
ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic
DEFAULT_MODEL=DeepSeek-v4-pro[1m]
MAX_ITERATIONS=15
EOF
```

---

## 第四步：创建虚拟环境 + 安装依赖

> **VMware 共享文件夹不支持 symlink，虚拟环境必须放在本地磁盘。**

```bash
# 1. 在本地磁盘创建虚拟环境（/root 下）
uv venv /root/joyagent-venv --python 3.11

# 2. 激活虚拟环境
source /root/joyagent-venv/bin/activate

# 3. 安装项目依赖
UV_LINK_MODE=copy uv sync --active
```

---

## 第五步：构建 Docker 沙箱镜像（一次性）

```bash
docker build -t joyagent-sandbox:latest -f sandbox_config/Dockerfile .
```

验证：

```bash
docker images | grep joyagent-sandbox
# → joyagent-sandbox   latest   xxx   N seconds ago   137MB
```

---

## 第六步：启动服务

```bash
# 确保虚拟环境已激活
source /root/joyagent-venv/bin/activate

# 启动
python main.py
```

期望日志：

```
[OK] Registered 13 tools: read_file, write_file, execute_shell, ...
[!!] Dangerous tools (require confirmation): write_file, execute_shell, git_commit, apply_patch
[OK] Registered 2 hooks: SafetyCheckHook (block_hard=True) + ToolStatsCollector (...)
[OK] JoyAgent startup complete -- ToolRegistry initialized.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

---

## 第七步：验证沙箱正常工作

### 7.1 找到虚拟机 IP

```bash
ip addr | grep inet
```

找到 `192.168.x.x` 或 `192.128.x.x` 那个地址（`ens33` 网卡）。

### 7.2 从 Windows 浏览器访问 Swagger

```
http://<虚拟机IP>:8000/docs
```

### 7.3 测试 Shell Sandbox

```bash
curl -X POST http://<虚拟机IP>:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "use execute_shell to run: echo hello-from-docker && hostname"}'
```

查看响应中的 `tool_calls` 字段：

```json
{
  "tool": "execute_shell",
  "execution_mode": "docker_sandbox"    ← 走的是 Docker 沙箱
}
```

如果 `execution_mode` 为 `host_subprocess`，说明 Docker 不可用，检查 Docker daemon 和镜像。

### 7.4 验证安全检查生效

```bash
curl -X POST http://<虚拟机IP>:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "use execute_shell to run: rm -rf /"}'
```

`tool_calls` 中应看到 `[BLOCKED] Command rejected by safety check`。

---

## 后续每次启动

```bash
# 激活虚拟环境
source /root/joyagent-venv/bin/activate

# 进入项目目录
cd /mnt/hgfs/joyagent/joyagent

# 启动
python main.py
```

---

## 安全架构一览

```
Agent 调用 execute_shell("...")
  │
  ├── ① SafetyCheckHook (应用层)
  │     ├── DENY:  直接拒绝 (rm -rf /, sudo, mkfs, fork炸弹 …)
  │     ├── INJECTION: 检测绕过意图 (base64 | sh, python -c import os …)
  │     └── ASK:   打印警告 + 放行 (rm, curl, pip install …)
  │
  ├── ② Docker Sandbox (系统层)
  │     ├── 容器隔离 (每次新容器，用完销毁)
  │     ├── 根文件只读 (read_only_root=True)
  │     ├── 无网络 (network_mode=none)
  │     ├── 非 root (user="sandbox")
  │     ├── 丢弃所有 capabilities (cap_drop=ALL)
  │     ├── CPU 1核 + 内存 512M + 60s 超时
  │     └── 禁止提权 (no-new-privileges)
  │
  └── ③ HITL 审批 (Phase 9, 流程层)
        └── ASK 模式弹窗确认
```
