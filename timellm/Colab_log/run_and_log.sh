#!/bin/bash
# =============================================================================
# run_and_log.sh — 一键跑实验 + 结构化命名日志 + 自动推回 git
#
# 用法（在 Colab 一个 cell 里）：
#   !cd /content/EEG-LLM917/timellm && git pull --ff-only \
#     && bash Colab_log/run_and_log.sh "DEAP-val2__VQ-GPT2__focal"
#
#   $1  slug  —— 描述这次实验的字段串，建议格式：
#               <数据-任务>__<模型-LLM>__<标签>
#               例： DEAP-val2__VQ-GPT2__focal
#   $2  要跑的脚本（可选，默认 run_deap_vq.sh）
#
# 产物（命名约定见 Colab_log/README.md）：
#   Colab_log/<YYYYMMDD-HHMM>__<slug>__g<gitshort>.log
#   日志开头自带一段 RUN META（date/git/host/GPU/命令），保证可溯源。
#
# 环境变量：
#   PUSH=0   只在本地写日志，不 commit / push（默认 PUSH=1）
# =============================================================================
set -uo pipefail

SLUG="${1:?usage: bash run_and_log.sh \"<slug>\" [run_script]}"
RUN_SCRIPT="${2:-run_deap_vq.sh}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../timellm/Colab_log
TIMELLM_DIR="$(dirname "$SCRIPT_DIR")"                        # .../timellm
# Colab 容器里 clone 的 repo 常被 git 判为 dubious ownership（任何 git 操作报 exit 128），先放行
git config --global --add safe.directory '*' 2>/dev/null || true
REPO_DIR="$(git -C "$TIMELLM_DIR" rev-parse --show-toplevel)"

DATETIME="$(date +%Y%m%d-%H%M)"
GITSHORT="$(git -C "$REPO_DIR" rev-parse --short HEAD)"
BRANCH="$(git -C "$REPO_DIR" rev-parse --abbrev-ref HEAD)"
GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"

LOGFILE="${DATETIME}__${SLUG}__g${GITSHORT}.log"
LOGPATH="$SCRIPT_DIR/$LOGFILE"

# ---- 写日志头（完整溯源信息）---------------------------------------------
{
  echo "# ===== RUN META ====="
  echo "# date:    $(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "# git:     $GITSHORT ($BRANCH)"
  echo "# host:    ${HOSTNAME:-unknown} / GPU: ${GPU:-n/a}"
  echo "# slug:    $SLUG"
  echo "# script:  $RUN_SCRIPT"
  echo "# ====================="
  echo ""
} > "$LOGPATH"

echo "[run_and_log] writing -> Colab_log/$LOGFILE"

# ---- 跑实验，stdout+stderr 同时进日志和屏幕 -------------------------------
( cd "$TIMELLM_DIR" && bash "$RUN_SCRIPT" ) 2>&1 | tee -a "$LOGPATH"
STATUS=${PIPESTATUS[0]}
{ echo ""; echo "# exit_status: $STATUS"; } >> "$LOGPATH"
echo "[run_and_log] finished, exit_status=$STATUS"

# ---- 自动提交并推回 git（PUSH=0 可跳过）-----------------------------------
if [ "${PUSH:-1}" = "1" ]; then
  # Colab 上若未配置 git 身份，给个占位，避免 commit 失败
  git -C "$REPO_DIR" config user.email >/dev/null 2>&1 || git -C "$REPO_DIR" config user.email "colab@runner.local"
  git -C "$REPO_DIR" config user.name  >/dev/null 2>&1 || git -C "$REPO_DIR" config user.name  "Colab Runner"

  git -C "$REPO_DIR" add "$LOGPATH"
  git -C "$REPO_DIR" commit -m "log: $SLUG (g$GITSHORT, exit $STATUS)" \
    && git -C "$REPO_DIR" pull --rebase --autostash origin "$BRANCH" \
    && git -C "$REPO_DIR" push origin "$BRANCH" \
    && echo "[run_and_log] pushed log to origin/$BRANCH" \
    || echo "[run_and_log] commit/push 失败：检查 Colab 的 git 推送凭证（token/SSH）"
else
  echo "[run_and_log] PUSH=0，仅本地保存日志，未推送"
fi
