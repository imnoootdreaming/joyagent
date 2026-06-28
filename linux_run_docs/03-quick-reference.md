# JoyAgent Linux 速查卡

## 首次部署（7 条命令）

```bash
# ① Docker
sudo systemctl start docker

# ② uv
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.local/bin/env

# ③ 配置
echo 'export UV_LINK_MODE=copy' >> ~/.bashrc && source ~/.bashrc
cat > .env << 'EOF'
ANTHROPIC_API_KEY=你的key
ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic
DEFAULT_MODEL=DeepSeek-v4-pro[1m]
EOF

# ④ 虚拟环境（本地磁盘！）
uv venv /root/joyagent-venv --python 3.11
source /root/joyagent-venv/bin/activate
UV_LINK_MODE=copy uv sync --active

# ⑤ Docker 镜像
docker build -t joyagent-sandbox:latest -f sandbox_config/Dockerfile .

# ⑥ 防火墙
sudo systemctl stop firewalld

# ⑦ 启动
python main.py
```

## 每次启动（3 条命令）

```bash
source /root/joyagent-venv/bin/activate
cd /mnt/hgfs/joyagent/joyagent
python main.py
```

## 验证清单

| 检查项 | 命令 | 期望结果 |
|------|------|------|
| Docker 在跑 | `docker ps` | 看到容器列表 |
| 镜像构建成功 | `docker images | grep joyagent` | `joyagent-sandbox latest` |
| 依赖安装完成 | `source /root/joyagent-venv/bin/activate && pip list | grep fastapi` | `fastapi` 在列表中 |
| 服务启动 | `python main.py` | `Uvicorn running on http://0.0.0.0:8000` |
| Sandbox 生效 | 发 API 请求后看 `execution_mode` | `docker_sandbox` |
| 安全检查生效 | 发 `execute_shell rm -rf /` | `[BLOCKED] Command rejected` |

## 核心概念速记

```
镜像 = 存在磁盘上的模板（docker images 看）
容器 = 镜像的运行实例（docker ps 看）
bind mount = 容器直接读写宿主机上的文件夹（不是复制）

执行流程:
  Agent 调 execute_shell
    → SafetyCheckHook 检查（DENY/ASK/INJECTION）
    → DockerRunner 自动创建临时容器
    → 执行命令
    → 收集日志
    → 销毁容器
```
