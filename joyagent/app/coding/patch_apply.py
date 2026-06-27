"""
Phase 4 Step 5: Patch Apply — 将 unified diff 应用到实际文件。

PatchApplier 是 Coding Agent 的"变更落地层"——它将 DiffGenerator 生成的
unified diff 精准应用到实际文件系统，支持系统 patch 命令和 Python 原生手动应用
两种模式（Windows 兼容）。

双模式设计：
  apply_system() — 使用系统 patch 命令（Linux/Mac/Git-Bash）
    ✅ 快速、原生、经过数十年验证
    ❌ Windows 默认无此命令

  apply_manual() — Python 原生 hunk-by-hunk 手动应用
    ✅ 跨平台、不依赖外部命令
    ✅ 可自定义偏移容错（fuzz tolerance）
    ✅ 精确的逐 hunk 错误报告
    ❌ 比系统 patch 慢（对 AI Agent 使用场景可忽略）

安全措施：
  1. 应用前自动备份 (.bak 文件)
  2. 验证原始内容与 diff 预期一致（防误打 patch）
  3. hunk 偏移容错（原始行号偏移 N 行时仍能匹配）
  4. 应用失败时自动恢复备份
  5. 结果精确报告（每个 hunk 成功/失败）
"""

# ── Python 标准库 ──
import os                              # 文件系统操作（路径、备份、删除）
import re                              # 正则解析 hunk 头部 (@@ -a,b +c,d @@)
import shutil                          # 文件复制（创建备份）
import subprocess                      # 调用系统 patch 命令
import tempfile                        # 临时 diff 文件（系统 patch 模式用）
from dataclasses import dataclass, field  # 数据类装饰器

# ── 项目内导入 ──
from app.coding.diff_generator import DiffResult  # Diff 生成结果（结构化 diff）


# ═══════════════════════════════════════════════════════════════════════════════
# 数据模型
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PatchHunkResult:
    """
    单个 hunk 的应用结果。

    每个 hunk 独立报告成功/失败，便于 Agent 诊断"哪部分对了、哪部分错了"。
    """
    header: str                        # hunk 的 @@ 头部行
    applied: bool                      # 是否成功应用
    old_start: int                      # 在原文件中的目标行号
    offset_used: int = 0               # 实际使用的行偏移（0=精确匹配，N=偏移了N行）
    error: str = ""                    # 失败原因（如 "context mismatch at line 45"）


@dataclass
class PatchResult:
    """
    单个文件的 Patch 应用结果。

    这是 PatchApplier.apply() 的返回类型。对比 Phase 1-2 的 write_file 盲写模式，
    PatchResult 提供了精确的逐 hunk 应用状态——Agent 可以据此判断：
      - 全部成功 → 任务完成
      - 部分成功 → 需检查失败的 hunk
      - 全部失败 → diff 可能有误，需重新生成

    Example:
      PatchResult(
          success=True,
          file_path="app/api/agent.py",
          hunks_applied=2,
          hunks_failed=0,
          hunk_details=[PatchHunkResult(...), PatchHunkResult(...)],
          backup_path="app/api/agent.py.bak",
      )
    """
    success: bool                       # 是否所有 hunk 都成功应用
    file_path: str                      # 被修改的文件路径
    hunks_applied: int                  # 成功应用的 hunk 数
    hunks_failed: int                   # 失败的 hunk 数
    hunk_details: list[PatchHunkResult] # 每个 hunk 的详细结果
    error_message: str = ""             # 整体错误消息（仅在 success=False 时可能非空）
    backup_path: str = ""               # 备份文件路径（回滚用）
    applied_by: str = ""               # 应用方式："system_patch" 或 "manual"


# ═══════════════════════════════════════════════════════════════════════════════
# PatchApplier
# ═══════════════════════════════════════════════════════════════════════════════

class PatchApplier:
    """
    unified diff 应用器。

    职责：
      1. 解析 diff 文本 → 提取文件路径 + hunk 列表
      2. 备份目标文件（.bak）
      3. 逐 hunk 匹配原始文件内容（支持偏移容错）
      4. 应用变更（删旧行、插入新行）
      5. 写入文件 + 报告结果

    双模式：
      apply()         → 自动选择：先试系统 patch，失败则降级到手动
      apply_manual()  → 直接 Python 手动应用（跨平台）

    使用方式：
      applier = PatchApplier(backup=True, offset_tolerance=3)
      result = applier.apply(diff_text, working_dir="/path/to/repo")
      if result.success:
          print(f"Applied {result.hunks_applied} hunks")
      else:
          print(f"Failed hunks: {result.hunks_failed}")
    """

    def __init__(self, backup: bool = True, offset_tolerance: int = 3):
        """
        Args:
            backup:           应用前是否创建 .bak 备份（默认 True）
            offset_tolerance: 手动模式下 hunk 匹配容错行数（默认 3）
                              值越大匹配越宽松，但错位风险也越大
        """
        self.backup = backup                 # 是否创建备份
        self.offset_tolerance = offset_tolerance  # hunk 偏移容错

    # ── 公共 API ──────────────────────────────────────────────

    def apply(
        self,
        diff_text: str,
        working_dir: str = ".",
    ) -> PatchResult:
        """
        应用 unified diff（自动选择模式）。

        优先尝试系统 patch 命令（快速），
        失败时自动降级到 Python 手动应用（跨平台兜底）。

        Args:
            diff_text:   unified diff 格式文本
            working_dir: diff 的基准目录（文件路径相对此目录）

        Returns:
            PatchResult — 成功时 hunks_applied > 0, 失败时含详细错误
        """
        # ── 0. 空 diff 保护 ──
        if not diff_text or not diff_text.strip():
            return PatchResult(
                success=False,
                file_path="",
                hunks_applied=0,
                hunks_failed=0,
                hunk_details=[],
                error_message="Empty diff text — nothing to apply.",
            )

        # ── 1. 解析 diff 获取文件路径 ──
        file_path = self._extract_file_path(diff_text)

        if not file_path:
            return PatchResult(
                success=False,
                file_path="",
                hunks_applied=0,
                hunks_failed=0,
                hunk_details=[],
                error_message="Could not extract file path from diff.",
            )

        full_path = os.path.join(working_dir, file_path)

        # ── 2. 尝试系统 patch 命令 ──
        result = self._try_system_patch(diff_text, working_dir, file_path)
        if result is not None:
            # 系统 patch 可能成功也可能失败（如命令不存在）
            if result.success or result.error_message == "patch command not found":
                return result
            # 系统 patch 失败了但有具体错误 → 降级到手动

        # ── 3. 降级：Python 手动应用 ──
        result = self.apply_manual(diff_text, working_dir)
        result.applied_by = "manual"         # 标记为手动模式
        return result

    def apply_manual(
        self,
        diff_text: str,
        working_dir: str = ".",
    ) -> PatchResult:
        """
        Python 原生逐 hunk 手动应用 diff。

        不依赖系统 patch 命令，可在任何 Python 环境中运行。
        支持偏移容错：当 hunk 声明的行号与实际文件内容不符时，
        在 ±offset_tolerance 范围内搜索匹配位置。

        Args:
            diff_text:   unified diff 文本
            working_dir: diff 的基准目录

        Returns:
            PatchResult
        """
        # ── 0. 空 diff ──
        if not diff_text or not diff_text.strip():
            return PatchResult(
                success=False, file_path="",
                hunks_applied=0, hunks_failed=0, hunk_details=[],
                error_message="Empty diff text.",
                applied_by="manual",
            )

        # ── 1. 解析 diff 文本 ──
        file_path = self._extract_file_path(diff_text)
        if not file_path:
            return PatchResult(
                success=False, file_path="",
                hunks_applied=0, hunks_failed=0, hunk_details=[],
                error_message="Could not extract file path from diff header.",
                applied_by="manual",
            )

        full_path = os.path.join(working_dir, file_path)
        hunks_data = self._parse_diff_hunks(diff_text)

        if not hunks_data:
            return PatchResult(
                success=False, file_path=file_path,
                hunks_applied=0, hunks_failed=0, hunk_details=[],
                error_message="No hunks found in diff text.",
                applied_by="manual",
            )

        # ── 2. 读取目标文件原始内容 ──
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                original_text = f.read()
        except FileNotFoundError:
            return PatchResult(
                success=False, file_path=file_path,
                hunks_applied=0, hunks_failed=len(hunks_data),
                hunk_details=[],
                error_message=f"Target file not found: {full_path}",
                applied_by="manual",
            )
        except PermissionError:
            return PatchResult(
                success=False, file_path=file_path,
                hunks_applied=0, hunks_failed=len(hunks_data),
                hunk_details=[],
                error_message=f"Permission denied: {full_path}",
                applied_by="manual",
            )

        original_lines = original_text.split("\n")

        # ── 3. 创建备份 ──
        backup_path = ""
        if self.backup:
            backup_path = full_path + ".bak"
            try:
                shutil.copy2(full_path, backup_path)  # copy2 保留文件元数据
            except OSError:
                pass                       # 备份失败不阻塞主流程

        # ── 4. 逐 hunk 应用 ──
        # 从后向前应用 hunk（后方的 hunk 先应用），这样前面的行号不会因后方插入/删除而错位
        hunks_data.sort(key=lambda h: h.get("old_start", 1), reverse=True)

        hunk_results: list[PatchHunkResult] = []
        all_succeeded = True

        for hunk in hunks_data:
            hr = self._apply_single_hunk(original_lines, hunk)
            hunk_results.append(hr)
            if not hr.applied:
                all_succeeded = False

        # ── 5. 检查结果 ──
        if not all_succeeded:
            # 部分或全部失败 → 恢复备份
            if backup_path and os.path.exists(backup_path):
                try:
                    shutil.copy2(backup_path, full_path)
                except OSError:
                    pass
            failed_count = sum(1 for h in hunk_results if not h.applied)
            return PatchResult(
                success=False,
                file_path=file_path,
                hunks_applied=len(hunk_results) - failed_count,
                hunks_failed=failed_count,
                hunk_details=hunk_results,
                error_message=f"{failed_count}/{len(hunk_results)} hunks failed — restored from backup.",
                backup_path=backup_path,
                applied_by="manual",
            )

        # ── 6. 全部成功 → 写入文件 ──
        result_text = "\n".join(original_lines)
        try:
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(result_text)
        except (PermissionError, OSError) as e:
            return PatchResult(
                success=False, file_path=file_path,
                hunks_applied=0, hunks_failed=len(hunks_data),
                hunk_details=[],
                error_message=f"Failed to write modified file: {e}",
                backup_path=backup_path,
                applied_by="manual",
            )

        return PatchResult(
            success=True,
            file_path=file_path,
            hunks_applied=len(hunk_results),
            hunks_failed=0,
            hunk_details=hunk_results,
            backup_path=backup_path,
            applied_by="manual",
        )

    def apply_batch(
        self,
        diffs: list[DiffResult | str],
        working_dir: str = ".",
    ) -> list[PatchResult]:
        """
        批量应用多个文件的 diff。

        Args:
            diffs:       DiffResult 对象列表或 diff_text 字符串列表
            working_dir: 工作目录

        Returns:
            list[PatchResult] — 每个文件一个结果
        """
        results: list[PatchResult] = []
        for diff in diffs:
            if isinstance(diff, DiffResult):
                text = diff.diff_text
            else:
                text = diff              # 字符串
            results.append(self.apply(text, working_dir))
        return results

    # ── 私有：hunk 应用核心算法 ──────────────────────────────

    def _apply_single_hunk(
        self,
        lines: list[str],
        hunk: dict,
    ) -> PatchHunkResult:
        """
        在行列表中定位并应用单个 hunk。

        算法步骤：
          1. 从 hunk 提取基线行号（声明位置）
          2. 从基线位置开始，在 ±offset_tolerance 范围内搜索匹配
          3. 如果找到匹配 → 执行替换（删旧行 + 插入新行 + 保留上下文）
          4. 如果没找到 → 返回失败（context mismatch）

        Args:
            lines:  目标文件的当前行列表（原地修改）
            hunk:   {"old_start": int, "old_count": int, "new_start": int,
                     "new_count": int, "header": str, "lines": [str...]}

        Returns:
            PatchHunkResult（含 applied / offset_used / error）
        """
        old_start = hunk.get("old_start", 1)    # 1-based 行号
        header = hunk.get("header", "@@ unknown @@")
        hunk_lines = hunk.get("lines", [])       # 包含 @@ 头部和所有内容行

        if not hunk_lines:
            return PatchHunkResult(
                header=header, applied=False,
                old_start=old_start,
                error="Empty hunk (no content lines).",
            )

        # ── 1. 构建匹配模式：只取 ' ' (context) 和 '-' (removed) 行 ──
        # 这些行必须在原始文件中存在（一字不差）
        # '+' 行是新增内容，在原始文件中不存在，不参与匹配
        expected_lines: list[str] = []
        for l in hunk_lines:
            if l.startswith(" ") or l.startswith("-"):
                # 去掉前缀（' ' 或 '-'）与原始文件行匹配
                expected_lines.append(l[1:])

        if not expected_lines:
            # 全新增文件（无原始内容可匹配）
            # 这是一个特殊情况：创建一个新文件
            new_content = []
            for l in hunk_lines:
                if l.startswith("+") or l.startswith(" "):
                    new_content.append(l[1:])
            # 在指定位置插入新行
            base_idx = old_start - 1           # 转为 0-based
            if base_idx > len(lines):
                base_idx = len(lines)
            for i, new_line in enumerate(new_content):
                lines.insert(base_idx + i, new_line)
            return PatchHunkResult(
                header=header, applied=True,
                old_start=old_start, offset_used=0,
            )

        # ── 2. 搜索匹配位置 ──
        base_idx = old_start - 1               # 0-based 索引
        tolerance = self.offset_tolerance
        offset = 0                             # 当前尝试的偏移

        for offset in self._offset_sequence(tolerance):
            idx = base_idx + offset

            # 边界检查
            if idx < 0 or idx + len(expected_lines) > len(lines):
                continue

            # 逐行比较：期望行 vs 实际行
            all_match = True
            for j, expected in enumerate(expected_lines):
                actual = lines[idx + j]
                if actual != expected:
                    all_match = False
                    break
            if all_match:
                # 找到了！执行替换
                self._do_replace(lines, idx, expected_lines, hunk_lines)
                return PatchHunkResult(
                    header=header, applied=True,
                    old_start=old_start, offset_used=offset,
                )

        # ── 3. 匹配失败 ──
        return PatchHunkResult(
            header=header, applied=False,
            old_start=old_start,
            error=(
                f"Context mismatch at line ~{old_start} "
                f"(tried offsets within ±{tolerance}). "
                f"File may have been modified since diff was generated."
            ),
        )

    @staticmethod
    def _do_replace(
        lines: list[str],
        idx: int,
        expected_lines: list[str],
        hunk_lines: list[str],
    ):
        """
        在匹配位置执行 hunk 替换（原地修改 lines 列表）。

        操作：
          1. 删除 idx 位置开始的 len(expected_lines) 行（context + removed）
          2. 在同一位置插入所有 '+' 和 ' ' 开头的行（新内容）
        """
        # ── 1. 删除旧行 ──
        # expected_lines 包含 ' ' 和 '-' 行 → 全部从文件中移除
        del lines[idx:idx + len(expected_lines)]

        # ── 2. 插入新行 ──
        # 新内容 = 所有 '+' 和 ' ' 开头的行（去掉前缀）
        new_lines: list[str] = []
        for l in hunk_lines:
            if l.startswith("+") or l.startswith(" "):
                new_lines.append(l[1:])   # 去掉 +/-/  前缀

        # 在 idx 位置逐个插入（用 reversed 保持顺序）
        for i, new_line in enumerate(new_lines):
            lines.insert(idx + i, new_line)

    @staticmethod
    def _offset_sequence(tolerance: int):
        """
        生成偏移搜索序列：0, -1, +1, -2, +2, ..., ±tolerance。

        优先尝试偏移 0（精确匹配），然后交替尝试近邻偏移。
        这样当精确匹配存在时，不会因为先尝试偏移 1 而误匹配。
        """
        yield 0                              # 优先精确匹配
        for d in range(1, tolerance + 1):
            yield -d                         # 尝试向前偏移
            yield d                          # 尝试向后偏移

    # ── 私有：diff 解析 ──────────────────────────────────────

    @staticmethod
    def _extract_file_path(diff_text: str) -> str:
        """
        从 unified diff 头部提取目标文件路径。

        格式：
          --- a/file/path.py          → 提取 "file/path.py"
          --- file/path.py            → 提取 "file/path.py"
          +++ b/file/path.py          → 提取 "file/path.py"（fallback）

        优先从 --- 提取（原始文件），fallback 到 +++（新文件）。
        """
        for line in diff_text.split("\n"):
            # --- a/path 或 --- path
            if line.startswith("--- "):
                path = line[4:]              # 去掉 "--- "
                # 移除 git diff 的 a/ b/ 前缀
                if path.startswith("a/"):
                    path = path[2:]
                elif path.startswith("b/"):
                    path = path[2:]
                return path.strip()

        # Fallback: 从 +++ 提取
        for line in diff_text.split("\n"):
            if line.startswith("+++ "):
                path = line[4:]
                if path.startswith("b/"):
                    path = path[2:]
                elif path.startswith("a/"):
                    path = path[2:]
                return path.strip()

        return ""                            # 无法提取

    @staticmethod
    def _parse_diff_hunks(diff_text: str) -> list[dict]:
        """
        从 diff 文本中解析出所有 hunk。

        返回格式：
          [{
              "old_start": int,
              "old_count": int,
              "new_start": int,
              "new_count": int,
              "header": str,       # 完整的 @@ 头部行
              "lines": [str...],   # hunk 的所有内容行（含头部）
          }]

        Hunk 分组逻辑：
          遇到 @@ 开头 → 新 hunk 开始
          遇到 ---/+++/diff/index/== 等元数据行 → 跳过
          其他行 → 当前 hunk 的内容行
        """
        hunks: list[dict] = []
        current_hunk: dict | None = None

        for line in diff_text.split("\n"):
            # ── 新 hunk 头部 ──
            if line.startswith("@@"):
                # 保存上一个 hunk（如有）
                if current_hunk:
                    hunks.append(current_hunk)

                # 解析新 hunk 的头部信息
                match = re.match(
                    r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@',
                    line,
                )
                if match:
                    current_hunk = {
                        "old_start": int(match.group(1)),
                        "old_count": int(match.group(2)) if match.group(2) else 1,
                        "new_start": int(match.group(3)),
                        "new_count": int(match.group(4)) if match.group(4) else 1,
                        "header": line.strip(),
                        "lines": [line],     # 头部也是 hunk 的一部分
                    }
                else:
                    current_hunk = None      # 无法解析的 @@ 行 → 忽略

            elif current_hunk is not None:
                # ── hunk 内容行 ──
                # 遇到新的元数据行（如第二个文件的 ---）→ 结束当前 hunk
                if line.startswith("--- ") or line.startswith("+++ "):
                    continue                 # 跳过后续文件的头部

                if line.startswith("diff ") or line.startswith("index "):
                    continue                 # 跳过 git diff 元数据

                # 空行和内容行都保留
                current_hunk["lines"].append(line)

        # 保存最后一个 hunk
        if current_hunk:
            hunks.append(current_hunk)

        return hunks

    # ── 私有：系统 patch 尝试 ────────────────────────────────

    def _try_system_patch(
        self,
        diff_text: str,
        working_dir: str,
        file_path: str,
    ) -> PatchResult | None:
        """
        尝试使用系统 patch 命令应用 diff。

        返回值：
          PatchResult — 系统 patch 执行完成（成功或失败）
          None — 系统 patch 命令不可用，应由调用方降级到手动模式
        """
        # ── 0. 检查 patch 命令是否可用 ──
        try:
            check = subprocess.run(
                ["patch", "--version"],
                capture_output=True, text=True, timeout=5,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            # Windows 默认无 patch 命令 → 返回 None 表示需要降级
            return PatchResult(
                success=False, file_path=file_path,
                hunks_applied=0, hunks_failed=0, hunk_details=[],
                error_message="patch command not found",
                applied_by="system_patch",
            )
        except Exception:
            return None                      # 其他错误也降级

        # ── 1. 写入临时 diff 文件 ──
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".diff",
                delete=False, encoding="utf-8",
            ) as f:
                f.write(diff_text)
                diff_temp_path = f.name
        except OSError:
            return None                      # 无法创建临时文件 → 降级

        try:
            # ── 2. 执行系统 patch ──
            # -p1: 去掉 diff 中 a/ b/ 的第一层目录前缀
            # -i:  从文件读取 diff（而非 stdin）
            # -N:  允许创建新文件
            # --dry-run: 先试运行检查冲突（可选）
            result = subprocess.run(
                ["patch", "-p1", "-i", diff_temp_path, "-N"],
                cwd=working_dir,
                capture_output=True,
                text=True,
                timeout=30,                  # 30 秒超时
            )

            # ── 3. 解析结果 ──
            if result.returncode == 0:
                return PatchResult(
                    success=True,
                    file_path=file_path,
                    hunks_applied=1,          # 系统 patch 不提供逐 hunk 细节
                    hunks_failed=0,
                    hunk_details=[],
                    applied_by="system_patch",
                )
            else:
                return PatchResult(
                    success=False,
                    file_path=file_path,
                    hunks_applied=0,
                    hunks_failed=1,
                    hunk_details=[],
                    error_message=result.stderr.strip() or "patch command failed",
                    applied_by="system_patch",
                )

        except subprocess.TimeoutExpired:
            return PatchResult(
                success=False, file_path=file_path,
                hunks_applied=0, hunks_failed=1, hunk_details=[],
                error_message="system patch timed out after 30 seconds",
                applied_by="system_patch",
            )
        finally:
            # ── 4. 清理临时文件 ──
            try:
                os.unlink(diff_temp_path)
            except OSError:
                pass
