#!/usr/bin/env bash
# AIOps Agent — 环境安装 + LangGraph 可视化调试服务一键启动
#
# 用法:
#   ./start_langgraph.sh              # 安装依赖并启动
#   ./start_langgraph.sh --skip-install   # 跳过安装，直接启动
#   ./start_langgraph.sh --tunnel         # 启动并开启 LangGraph tunnel
#
# 可通过环境变量覆盖默认配置:
#   CONDA_ENV=RCAgentX-3.11  HOST=0.0.0.0  PORT=2024  ./start_langgraph.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"

# ── 可配置项 ──────────────────────────────────────────────
CONDA_ENV="${CONDA_ENV:-RCAgentX-3.11}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-2024}"
SKIP_INSTALL=0
ENABLE_TUNNEL=0

# ── 参数解析 ──────────────────────────────────────────────
usage() {
    cat <<'EOF'
AIOps Agent LangGraph 启动脚本

用法:
  ./start_langgraph.sh [选项]

选项:
  --skip-install    跳过依赖安装，直接启动 langgraph dev
  --tunnel          启动时附加 --tunnel（LangGraph 公网隧道）
  -h, --help        显示帮助

环境变量:
  CONDA_ENV         Conda 环境名（默认: RCAgentX-3.11）
  PYTHON_BIN        直接指定 Python 可执行文件（优先级高于 CONDA_ENV）
  HOST              监听地址（默认: 0.0.0.0）
  PORT              监听端口（默认: 2024）

示例:
  ./start_langgraph.sh
  CONDA_ENV=zh_aiops PORT=8123 ./start_langgraph.sh
  PYTHON_BIN=/opt/conda/envs/RCAgentX-3.11/bin/python ./start_langgraph.sh --skip-install
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-install) SKIP_INSTALL=1 ;;
        --tunnel)       ENABLE_TUNNEL=1 ;;
        -h|--help)      usage; exit 0 ;;
        *)
            echo "未知参数: $1" >&2
            usage
            exit 1
            ;;
    esac
    shift
done

log()  { echo "[start_langgraph] $*"; }
warn() { echo "[start_langgraph] WARNING: $*" >&2; }
die()  { echo "[start_langgraph] ERROR: $*" >&2; exit 1; }

# ── 选择 Python 解释器 ────────────────────────────────────
resolve_python() {
    if [[ -n "${PYTHON_BIN:-}" ]]; then
        [[ -x "$PYTHON_BIN" ]] || die "PYTHON_BIN 不可执行: $PYTHON_BIN"
        echo "$PYTHON_BIN"
        return
    fi

    # 已激活的 conda 环境
    if [[ -n "${CONDA_PREFIX:-}" ]] && [[ -x "${CONDA_PREFIX}/bin/python" ]]; then
        echo "${CONDA_PREFIX}/bin/python"
        return
    fi

    # 按名称查找 conda 环境
    local candidates=(
        "/opt/conda/envs/${CONDA_ENV}/bin/python"
        "${HOME}/.conda/envs/${CONDA_ENV}/bin/python"
        "/home/serveradmin/.conda/envs/${CONDA_ENV}/bin/python"
    )
    for py in "${candidates[@]}"; do
        if [[ -x "$py" ]]; then
            echo "$py"
            return
        fi
    done

    # 尝试 conda activate
    if command -v conda &>/dev/null; then
        eval "$(conda shell.bash hook 2>/dev/null)" || true
        if conda activate "$CONDA_ENV" 2>/dev/null; then
            echo "$(which python)"
            return
        fi
    fi

    die "未找到 Python 3.10+ 环境。请设置 PYTHON_BIN 或创建 conda 环境: conda create -n ${CONDA_ENV} python=3.11"
}

check_python_version() {
    local py="$1"
    local version
    version="$("$py" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    local major minor
    IFS='.' read -r major minor <<< "$version"
    if (( major < 3 || (major == 3 && minor < 10) )); then
        die "需要 Python >= 3.10，当前: ${version} ($py)"
    fi
    log "Python ${version} → $py"
}

# ── 环境文件 ──────────────────────────────────────────────
ensure_env_file() {
    cd "$PROJECT_ROOT"
    if [[ ! -f .env ]]; then
        if [[ -f .env.example ]]; then
            cp .env.example .env
            warn ".env 不存在，已从 .env.example 复制。请编辑 .env 填入 DEEPSEEK_API_KEY。"
        else
            die "缺少 .env 和 .env.example，无法继续。"
        fi
    fi

    if grep -q 'your_deepseek_api_key_here' .env 2>/dev/null; then
        warn "DEEPSEEK_API_KEY 仍为占位符，请在 .env 中配置真实 API Key。"
    fi
}

# ── 依赖安装 ──────────────────────────────────────────────
install_dependencies() {
    local py="$1"

    cd "$PROJECT_ROOT"
    log "安装/更新项目依赖..."

    "$py" -m pip install -e .
    "$py" -m pip install -r requirements.txt
    "$py" -m pip install pyod

    log "验证核心模块导入..."
    "$py" -c "from app.agents.main_graph import main_graph; print('main_graph OK')"

    if ! "$py" -c "import langgraph_cli" 2>/dev/null; then
        die "langgraph-cli 未安装，请检查 requirements.txt 安装是否成功。"
    fi

    log "依赖安装完成。"
}

# ── 输出目录 ──────────────────────────────────────────────
ensure_output_dirs() {
    mkdir -p "${PROJECT_ROOT}/outputs/graphs"
    mkdir -p "${PROJECT_ROOT}/outputs/reports"
    mkdir -p "${PROJECT_ROOT}/log"
}

# ── 启动 LangGraph ────────────────────────────────────────
start_langgraph() {
    local py="$1"
    local langgraph_bin
    langgraph_bin="$(dirname "$py")/langgraph"

    if [[ ! -x "$langgraph_bin" ]]; then
        langgraph_bin="$(command -v langgraph || true)"
    fi
    [[ -n "$langgraph_bin" && -x "$langgraph_bin" ]] || die "未找到 langgraph 命令，请先完成依赖安装。"

    cd "$PROJECT_ROOT"

    local tunnel_args=()
    if [[ "$ENABLE_TUNNEL" -eq 1 ]]; then
        tunnel_args+=(--tunnel)
    fi

    local public_base
    public_base="$(grep -E '^LANGGRAPH_PUBLIC_BASE_URL=' .env 2>/dev/null | cut -d= -f2- | tr -d '"' || true)"
    public_base="${public_base:-http://127.0.0.1:${PORT}}"

    echo ""
    echo "============================================================"
    echo " AIOps Agent — LangGraph 调试服务"
    echo "============================================================"
    echo " 项目目录:     ${PROJECT_ROOT}"
    echo " Python:       ${py}"
    echo " API 地址:     http://${HOST}:${PORT}"
    echo " Studio UI:    http://127.0.0.1:8123  (langgraph dev 默认)"
    echo " 拓扑图:       ${public_base}/topology/latest"
    echo " 分析报告:     ${public_base}/report/latest"
    echo ""
    echo " 按 Ctrl+C 停止服务"
    echo "============================================================"
    echo ""

    exec "$langgraph_bin" dev \
        --host "$HOST" \
        --port "$PORT" \
        "${tunnel_args[@]+"${tunnel_args[@]}"}"
}

# ── 主流程 ────────────────────────────────────────────────
main() {
    log "项目目录: ${PROJECT_ROOT}"

    local python_bin
    python_bin="$(resolve_python)"
    check_python_version "$python_bin"
    ensure_env_file
    ensure_output_dirs

    if [[ "$SKIP_INSTALL" -eq 0 ]]; then
        install_dependencies "$python_bin"
    else
        log "跳过依赖安装 (--skip-install)。"
    fi

    start_langgraph "$python_bin"
}

main
