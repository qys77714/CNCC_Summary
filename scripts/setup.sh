#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mode="${1:-gpu}"
if [[ "$mode" != "gpu" && "$mode" != "cpu" && "$mode" != "speakers" ]]; then
  echo "用法：bash scripts/setup.sh [gpu|cpu|speakers]"
  exit 2
fi
task_python="${PYTHON:-python3.11}"
if ! command -v "$task_python" >/dev/null 2>&1; then
  task_python=python3
fi
"$task_python" -c 'import sys; assert (3,11) <= sys.version_info[:2] < (3,13), "请使用 Python 3.11 或 3.12"'
# 不复用从 macOS 复制来的 .venv。服务器环境使用独立目录。
"$task_python" -m venv .venv-server
task_pip=.venv-server/bin/pip
"$task_pip" install --upgrade pip
if [[ "$mode" == "cpu" ]]; then
  "$task_pip" install -e '.[dev]'
else
  "$task_pip" install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
  if [[ "$mode" == "speakers" ]]; then
    "$task_pip" install -c constraints-gpu.txt -e '.[gpu,speakers,dev]'
  else
    "$task_pip" install -c constraints-gpu.txt -e '.[gpu,dev]'
  fi
fi
if [[ ! -f .env ]]; then
  cp .env.example .env
fi
"$task_pip" freeze > environment.freeze.txt
echo "环境已创建。下一步：下载模型、填写 .env，然后运行 .venv-server/bin/cncc-minutes doctor --llm"
