# ═══════════════════════════════════════════════════════════════════════════
# Phase 5 Step 1: JoyAgent Sandbox Dockerfile
# ═══════════════════════════════════════════════════════════════════════════
# 构建命令：
#   docker build -t joyagent-sandbox:latest -f sandbox_config/Dockerfile .
# 验证镜像：
#   docker run --rm joyagent-sandbox:latest python -c "print('Sandbox OK')"
# ═══════════════════════════════════════════════════════════════════════════

# ── 基础镜像 ─────────────────────────────────────────────────────────────
# python:3.11-slim: 约 150MB，比 full 版本（~900MB）小得多
# 3.11 提供 ast.end_lineno 支持（Phase 4 AST 分析依赖）
FROM python:3.11-slim

# ── 安全：创建非 root 用户 ──────────────────────────────────────────────
# Agent 生成的代码在此用户下运行，遵循最小权限原则
# 即使用户请求 "rm -rf /"，也只能删除 sandbox 用户有权限的文件
RUN useradd --create-home --shell /bin/bash sandbox

# ── 切换到非特权用户 ────────────────────────────────────────────────────
# 所有后续 RUN 和容器运行时的默认用户
USER sandbox

# ── Python 环境配置 ──────────────────────────────────────────────────────
# 将用户本地 bin 目录加入 PATH（--user 安装的包放在这里）
ENV PATH="/home/sandbox/.local/bin:${PATH}"

# ── 预装测试依赖 ────────────────────────────────────────────────────────
# --user: 安装到用户目录（不需要 root 权限）
# --no-cache-dir: 减小镜像体积
# pytest: 测试框架
# pytest-json-report: 生成 JSON 格式测试报告（Phase 5 Step 3 依赖）
# pytest-timeout: 测试超时控制
RUN pip install --user --no-cache-dir \
    pytest \
    pytest-json-report \
    pytest-timeout

# ── 工作目录 ─────────────────────────────────────────────────────────────
# 容器启动后的默认目录，宿主机代码通过 bind mount 挂载到这里
WORKDIR /workspace

# ── 默认命令 ─────────────────────────────────────────────────────────────
# 如果用户没有提供命令，显示环境信息
CMD ["python", "--version"]
