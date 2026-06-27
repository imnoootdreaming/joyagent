# JoyAgent Linux 部署常见问题

---

## 问题 1：`uv: command not found`

**现象：**

```bash
[root@localhost joyagent]# uv sync
bash: uv: command not found...
```

**原因：** 安装 `uv` 后没有重新加载 PATH。

**解决：**

```bash
source $HOME/.local/bin/env

# 一劳永逸
echo 'source $HOME/.local/bin/env' >> ~/.bashrc
```

---

## 问题 2：`error 95: Operation not supported`（symlink 错误）

**现象：**

```bash
error: failed to symlink file from ... to ... : Operation not supported (os error 95)
```

**原因：** VMware 共享文件夹（hgfs / vmhgfs）不支持符号链接（symlink），而 Python 虚拟环境必须用 symlink。

**解决：**

```bash
# 必须做两件事：

# ① 虚拟环境放在本地磁盘，不要在共享文件夹里
uv venv /root/joyagent-venv --python 3.11

# ② 设置 UV_LINK_MODE=copy（让 uv 复制文件而不是建 symlink）
export UV_LINK_MODE=copy
echo 'export UV_LINK_MODE=copy' >> ~/.bashrc
```

---

## 问题 3：`.venv` 损坏后 `uv sync` 报错

**现象：**

```bash
error: Project virtual environment directory `/mnt/hgfs/.../.venv`
       cannot be used because it is not a valid Python environment
```

**原因：** 上一次 `uv sync` 失败时留下了一个不完整的 `.venv` 目录。

**解决：**

```bash
rm -rf .venv
source /root/joyagent-venv/bin/activate
UV_LINK_MODE=copy uv sync --active
```

---

## 问题 4：看不到隐藏文件（`.env` 找不到了）

**现象：**

```bash
[root@localhost joyagent]# ll
total 877
... 没看到 .env ...
```

**原因：** `ll`（即 `ls -l`）和 `ls` 不会显示以 `.` 开头的隐藏文件。

**解决：**

```bash
ls -la     # ← 加 -a 参数才能看到 .env, .gitignore 等隐藏文件
```

---

## 问题 5：`no_new_privileges` 参数错误

**现象：**

```json
{
  "tool": "execute_shell",
  "result": "Error: DockerError: run() got an unexpected keyword argument 'no_new_privileges'"
}
```

**原因：** 虚拟机上 docker-py 版本较老（< 7.0），不支持 `no_new_privileges` 作为 SDK 参数。

**解决：** 代码已修复——改用 Docker 原生的 `security_opt=["no-new-privileges:true"]` 写法，全版本兼容。

如果仍遇到此问题，检查 `app/sandbox/security.py` 中 `to_docker_params()` 方法是否使用 `security_opt`。

---

## 问题 6：`Failed to collect container logs`

**现象：**

```json
{
  "tool": "execute_shell",
  "result": "[STDERR]\nFailed to collect container logs."
}
```

**原因：** `auto_remove=True` 导致容器退出后被 Docker 立即删除，代码收集日志时容器已不存在。

**解决：** 代码已修复——`auto_remove` 改为 `False`，由 `DockerRunner` 的 `finally` 块先收集日志再手动删除容器。

---

## 问题 7：`$HOME/.local/bin/env` 文件不存在

**现象：**

```bash
[root@localhost joyagent]$ source $HOME/.local/bin/env
bash: /home/sheet1/.local/bin/env: No such file or directory
```

**原因：** 当前登录的用户（如 `sheet1`）和安装 `uv` 时的用户（如 `root`）不同。`uv` 安装在每个用户自己的 `~/.local/bin/` 下。

**解决：**

```bash
# 在当前用户下重新安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 或用完整路径（如果已经装过）
~/.local/bin/uv --version
```

---

## 问题 8：`uv sync --active` 报 "does not match"

**现象：**

```bash
warning: `VIRTUAL_ENV=/root/joyagent-venv` does not match the project environment path `.venv`
error: failed to symlink file ...
```

**原因：** 没加 `--active` 时，`uv sync` 坚持在项目目录下创建 `.venv`（共享文件夹不支持 symlink）。

**解决：**

```bash
UV_LINK_MODE=copy uv sync --active
```

`--active` 强制 uv 使用已激活的虚拟环境，而不是在项目目录新建。

---

## 问题 9：共享文件夹路径

**现象：** 不知道共享文件夹挂载到哪里。

**解决：**

```bash
# VMware 默认挂载路径
ls /mnt/hgfs/

# 如果没有，手动挂载
sudo mkdir -p /mnt/hgfs
sudo vmhgfs-fuse .host:/共享文件夹名 /mnt/hgfs/共享文件夹名 -o allow_other
```

---

## 问题 10：虚拟机 IP 是哪个

**现象：**

```bash
ip addr | grep inet
# 输出一大堆 IP，不知道用哪个
```

**解决：**

| IP | 是什么 | 用哪个 |
|------|------|------|
| `127.0.0.1` | 虚拟机自己 | ❌ |
| `192.128.128.128` (ens33) | **虚拟机真实 IP** | ✅ **用这个** |
| `192.168.122.1` (virbr0) | libvirt 内部虚拟网桥 | ❌ |
| `172.17.0.1` (docker0) | Docker 容器内部网络 | ❌ |

---

## 问题 11：Windows 访问不到虚拟机 8000 端口

**原因：** CentOS 防火墙默认拦截。

**解决：**

```bash
# 方案 A：开放端口
sudo firewall-cmd --add-port=8000/tcp --permanent
sudo firewall-cmd --reload

# 方案 B：直接关防火墙（仅开发环境）
sudo systemctl stop firewalld
```

---

## 问题 12：Python 版本太老（CentOS 7 自带 2.7）

**不要手动升级系统 Python。** 系统工具（如 `yum`）依赖 Python 2.7。

**解决：** `uv` 会自动下载管理 Python 3.11，不需要你手动装。

```bash
uv venv /root/joyagent-venv --python 3.11   # uv 自动下载 Python 3.11
```
