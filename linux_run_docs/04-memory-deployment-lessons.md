# Memory 系统部署经验沉淀

本文档记录在 CentOS 7 虚拟机上通过 Docker 部署 JoyAgent Memory 系统（Phase 6）时遇到的全部问题及解决方案。

---

## 1. CentOS 7 glibc 过老，无法安装新版 Python 依赖

**现象：**

```bash
uv pip install -e .
  × Failed to build numpy==2.4.6
  error: ERROR: Unknown compiler(s): [['g++'], ['clang++'], ...]
```

**根因：** CentOS 7 的 glibc 是 2.17，numpy 2.x / scipy 1.17 / scikit-learn 1.9 都需要 glibc ≥ 2.28。同时 CentOS 7 缺少 C++ 编译器，也无法从源码编译。

**解决：**

1. 放弃在 CentOS 7 宿主机上直接装依赖
2. 改用 Docker 容器（python:3.11-slim，基于 Debian，glibc 2.36）
3. 在 `pyproject.toml` 中锁定兼容版本：

```toml
"numpy>=1.26,<2.0",
"scikit-learn>=1.3,<1.6",
"sentence-transformers>=4.0,<5.0",
```

---

## 2. CentOS 7 yum 源已下线

**现象：**

```bash
sudo yum install -y gcc-c++
Could not retrieve mirrorlist http://mirrorlist.centos.org?...
```

**根因：** CentOS 7 于 2024 年 6 月 30 日 EOL，官方镜像源关闭。

**解决：** Docker 容器内自带 Debian apt 源，不受影响。不再尝试在宿主机上修复 yum。

---

## 3. `pyproject.toml` 缺少 build-system 配置

**现象：**

```bash
error: Multiple top-level packages discovered in a flat-layout:
       ['app', 'data', 'static', 'sandbox_config']
```

**根因：** 项目中 `data/`、`static/`、`sandbox_config/` 不是 Python 包（无 `__init__.py`），但 setuptools 自动发现时误判。

**解决：** `pyproject.toml` 中加包发现配置：

```toml
[build-system]
requires = ["setuptools>=64"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["app*"]
```

---

## 4. `tiktoken` 需要 Rust 编译器

**现象：**

```bash
× Failed to build tiktoken==0.13.0
error: can't find Rust compiler
```

**根因：** `tiktoken` 包含 Rust 扩展，需要 `rustc` 和 `cargo`。

**解决：** 在 CentOS 7 宿主机上安装 Rust：

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

或在 Dockerfile 中安装：`RUN curl ... | sh -s -- -y`。

---

## 5. HuggingFace 模型下载被墙

**现象：**

```
Network is unreachable: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/...
```

**根因：** `sentence-transformers` 首次加载 `all-MiniLM-L6-v2` 时需要从 HuggingFace 下载 ~80MB 模型文件，国内网络无法直连。

**走过的弯路：**

| 方案 | 结果 |
|------|------|
| 设置 `HF_ENDPOINT=https://hf-mirror.com` | 503/超时，镜像站不稳定 |
| Docker 构建时 RUN 下载模型 | 卡死 410 秒，build 中断 |
| 切换为 `mirror.sjtu.edu.cn` | 同样超时 |

**最终方案 — 模型离线部署：**

1. Windows 上开 VPN，从 `hf-mirror.com` 手动下载 8 个文件
2. 放到 `joyagent/models/all-MiniLM-L6-v2/` 目录
3. Dockerfile 中将模型打入镜像，设置离线模式：

```dockerfile
COPY models/ /models/
ENV EMBEDDING_MODEL=/models/all-MiniLM-L6-v2
ENV HF_HUB_OFFLINE=1
```

**模型文件清单（共 ~87MB）：**

| 文件 | 大小 | 作用 |
|------|------|------|
| `model.safetensors` | 87 MB | 模型权重 |
| `tokenizer.json` | 456 KB | 分词器词表 |
| `config.json` | 612 B | Transformer 结构配置 |
| `tokenizer_config.json` | 350 B | 分词器行为配置 |
| `special_tokens_map.json` | 112 B | 特殊 token 映射 |
| `modules.json` | 349 B | 模块清单 |
| `sentence_bert_config.json` | 53 B | 池化策略 |
| `1_Pooling/config.json` | 190 B | 池化层配置 |

---

## 6. `tool_use_id` 不匹配导致 400 错误

**现象：**

```json
{
  "error": "unexpected tool_use_id 'write_file_1' in tool_result blocks.
            Each tool_result must have a corresponding tool_use block
            in the previous message."
}
```

**根因：** `MemoryManager.after_tool_call()` 自己编造了 `tool_use_id = "write_file_1"`，但 API 返回的真实 ID 是 `"toolu_01Xyz..."`。当 STM 用 `get_context()` 构建消息时，`tool_result` 的 ID 和 `assistant` 消息里的 `tool_use` ID 对不上。

```
正确流程:
  assistant: tool_use {id: "toolu_01Xyz..."}   ← API 生成
  user: tool_result {tool_use_id: "toolu_01Xyz..."}  ← 引用相同 ID ✅

错误流程（修复前）:
  assistant: tool_use {id: "toolu_01Xyz..."}   ← API 生成
  user: tool_result {tool_use_id: "write_file_1"}    ← MemoryManager 自己编的 ❌
```

**解决：**

- `manager.py`: `after_tool_call` 不再向 STM 追加 tool_result 消息
- `agent.py`: 在 agent loop 构造 tool_result 时（已有真实 ID），同时同步到 STM

```python
# agent.py 修复
tool_result_msg = {"role": "user", "content": tool_results}
messages.append(tool_result_msg)
if mm is not None:
    mm.stm.add_message(tool_result_msg)  # 使用真实 tool_use_id
```

---

## 7. Docker 容器内 sandbox 挂载路径不对

**现象：**

```
Error: DockerError: 400 — bind source path does not exist: /app
```

**根因：** JoyAgent 跑在 Docker 容器内，`os.getcwd()` 返回 `/app`。但当 sandbox 通过 `docker.sock` 在宿主机上创建容器时，`bind mount source` 用的还是容器内路径 `/app`，宿主机根本没有这个路径。

```
容器内: /app 存在 ✅
宿主机: /app 不存在 ❌  →  bind mount 失败
```

**解决：**

1. Dockerfile 中添加环境变量：

```dockerfile
ENV SANDBOX_HOST_MOUNT_PATH=/mnt/hgfs/joyagent
```

2. `execute.py` 中读取此变量：

```python
host_mount = os.environ.get("SANDBOX_HOST_MOUNT_PATH", "")
if host_mount:
    mount_path = host_mount  # 宿主机真实路径
```

---

## 8. `>/dev/null` 被安全策略误拦截

**现象：**

```json
"Error: SAFETY_DENY: (pattern: >\\s*/dev/null\\b)"
```

**根因：** `safety_hook.py` 中 `DENY_PATTERNS` 列表包含 `>/dev/null`，虽然 reason 为空字符串（占位），但匹配后仍返回了 DENY 结果。

**解决：** 检查 DENY 模式时跳过 reason 为空的占位模式：

```python
for pattern, reason in DENY_PATTERNS:
    if not pattern or not reason:  # 空 reason → 占位模式，跳过
        continue
```

---

## 9. Agent 缺少 Memory 工具

**现象：** Agent 收到"保存到长期记忆"的任务后，因为没有对应的 Tool，试图通过 `execute_shell` 在 sandbox 里跑 Python 脚本来操作 ChromaDB——但 sandbox 容器里没有 chromadb 也没有网络。

```
execute_shell: pip install chromadb  →  name resolution failed (sandbox 无网络)
execute_shell: python3 -c "import chromadb"  →  ModuleNotFoundError
```

**解决：** 新增 `RememberTool`（`app/tools/remember.py`），提供两个操作：

- `action=save`: 将文本转为 embedding 后存入 ChromaDB
- `action=search`: 语义检索历史记忆

Agent 现在有 14 个工具（多了 `remember`），会直接调用而非绕过 sandbox。

---

## 10. VPN 导致 Windows 无法访问虚拟机

**现象：** 浏览器访问 `http://192.128.128.128:8000/docs` 返回 `HTTP ERROR 502`，但 CentOS 本地 `curl 127.0.0.1:8000/docs` 返回 200 OK。

**根因：** Windows 系统代理/VPN 接管了全局网络路由，浏览器请求被发送到远程 VPN 服务器，远程服务器无法解析 `192.128.128.128`（VMware NAT 地址）。

**验证方法：**

```cmd
ping 192.128.128.128    # 通 → 网络可达
curl --noproxy "*" http://192.128.128.128:8000/docs  # 绕过代理测试
```

**解决：** 关掉 VPN 或将 `192.128.128.128` 加入代理例外列表。

---

## 11. 过度迭代（max_iterations 不够）

**现象：** 复杂任务（创建文件 + 测试 + 保存记忆）在 15 步内无法完成。

**解决：** `MAX_ITERATIONS` 从 15 提升到 30（`app/core/config.py`）。

---

## 架构总结

```
┌─────────────────────────────────────────────────────────┐
│                    Windows 本机                          │
│  VS Code 编辑代码 ←→ VMware 共享文件夹 →→ CentOS 7 虚拟机 │
│  浏览器访问 docs                                        │
└─────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────┐
│                   CentOS 7 宿主机                        │
│                                                         │
│  Docker daemon (docker.sock)                            │
│  ├── JoyAgent 容器 (joyagent:latest)                    │
│  │   ├── python:3.11-slim (glibc 2.36 ✅)              │
│  │   ├── /app = /mnt/hgfs/joyagent (代码实时同步)        │
│  │   ├── /models/all-MiniLM-L6-v2 (离线嵌入模型)         │
│  │   ├── --network host (127.0.0.1 直通宿主机服务)       │
│  │   ├── SANDBOX_HOST_MOUNT_PATH=/mnt/hgfs/joyagent    │
│  │   └── 通过 docker.sock 创建 sandbox 容器              │
│  │                                                      │
│  └── Sandbox 容器 (joyagent-sandbox:latest)             │
│      ├── 按需创建，用完即销毁                             │
│      ├── bind mount: /mnt/hgfs/joyagent → /workspace   │
│      └── 无网络 / 只读根 / 非 root / CPU 1核 / 内存 512M │
│                                                         │
│  Redis :6379  ◄── JoyAgent 通过 host 网络直连            │
│  MySQL :3306                                            │
│  RabbitMQ :5672                                         │
│                                                         │
│  数据持久化:                                             │
│  /mnt/hgfs/joyagent/data/chroma/                        │
│    ├── chroma.sqlite3 (元数据)                           │
│    └── */data_level0.bin (HNSW 向量索引)                │
└─────────────────────────────────────────────────────────┘
```

---

## 关键经验

1. **CentOS 7 不要用于 AI 技术栈** — glibc 2.17 太老，几乎所有现代 Python AI 库都需要 ≥ 2.28。Docker 是最低成本的解法。

2. **模型离线部署是必选项** — 国内环境 HuggingFace 不稳定，镜像站也不可靠。把模型文件打入镜像或挂载到容器才是最稳的方案。

3. **docker.sock 通信要注意路径映射** — JoyAgent 容器内的路径 ≠ 宿主机路径。需要通过环境变量明确传递宿主机真实路径。

4. **tool_use_id 必须用 API 返回的真实 ID** — 不要自己编造。STM 和 agent loop 的消息管理要协调，避免两套逻辑冲突。

5. **Memory 工具必须显式注册** — Agent 不能通过 `execute_shell` 绕过 sandbox 来操作 ChromaDB。给 Agent 一个正规的 `remember` 工具是最干净的做法。
