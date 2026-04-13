#!/usr/bin/env bash
# =============================================================================
# 仅配置虚拟环境，不启动任何实验、不启动 Ray。
#
# 用法：
#   bash /workspace/Search-R1/setup_econ_train_vllm_venv.sh
#
# 可选环境变量：
#   VLLM_SETUP_VENV   目标 venv 目录（默认 /workspace/venvs/econ_train_vllm）
#   PYTHON_FOR_VENV   用于创建 venv 的解释器（默认 python3）
#   FORCE_VLLM_STACK  设为 legacy 或 blackwell 可跳过自动检测（一般不用）
#
# 分支说明：
#   - legacy（如 Hopper 及更早，compute capability 主版本 < 12）：
#       按 Search-R1 README：torch 2.4.0 (cu121) + vllm==0.6.3，并做 import + CUDA 张量自检。
#   - blackwell（sm_120 等，主版本 >= 12）：
#       torch 2.4+cu121 无法在 GPU 上执行；vllm 0.6.3 的 wheel 又与 torch 2.12/cu128
#       ABI 不兼容。脚本会安装：PyTorch nightly cu128（与当前 Blackwell 驱动常见组合）
#       + verl 依赖（不含通过 pip 安装的 vllm），并打印后续「源码编译 vLLM / 换 VERL 白名单」说明。
# =============================================================================
set -euo pipefail

VENV_DIR="${VLLM_SETUP_VENV:-/workspace/venvs/econ_train_vllm}"
PYTHON_BIN="${PYTHON_FOR_VENV:-python3}"
SEARCH_R1_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

detect_stack() {
  if [[ -n "${FORCE_VLLM_STACK:-}" ]]; then
    echo "${FORCE_VLLM_STACK}"
    return
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "legacy"
    return
  fi
  local cap
  cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' \r')"
  if [[ -z "${cap}" ]]; then
    echo "legacy"
    return
  fi
  local major="${cap%%.*}"
  if [[ "${major}" =~ ^[0-9]+$ ]] && [[ "${major}" -ge 12 ]]; then
    echo "blackwell"
  else
    echo "legacy"
  fi
}

STACK="$(detect_stack)"

echo "==> 使用解释器: $(${PYTHON_BIN} -V 2>&1)"
echo "==> 目标 venv: ${VENV_DIR}"
echo "==> 检测到的安装分支: ${STACK}（FORCE_VLLM_STACK=${FORCE_VLLM_STACK:-空}）"

if [[ ! -d "${VENV_DIR}" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

python -m pip install -U pip setuptools wheel

install_verl_no_vllm() {
  echo "==> 安装 verl 依赖（不含 vllm，可编辑安装包本体）"
  pip install "accelerate" "codetiming" "datasets" "dill" "hydra-core" "numpy" "pybind11" "ray" \
    "tensordict<0.6" "transformers<4.48"
  pip install -e "${SEARCH_R1_ROOT}" --no-deps
}

if [[ "${STACK}" == "legacy" ]]; then
  echo "==> 安装 PyTorch 2.4.0 (cu121) [Search-R1 README]"
  pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu121

  echo "==> 安装 vLLM 0.6.3 [VERL third_party 白名单]"
  pip install "vllm==0.6.3"

  echo "==> tensordict<0.6（与仓库 requirements 对齐）"
  pip install "tensordict<0.6"

  echo "==> 可编辑安装 Search-R1 / verl"
  pip install -e "${SEARCH_R1_ROOT}"

else
  echo "==> Blackwell：卸载可能存在的旧版 vllm/xformers（避免与 torch ABI 混用）"
  pip uninstall -y vllm xformers 2>/dev/null || true

  echo "==> Blackwell：安装 PyTorch nightly (cu128)（支持 sm_120；与 README 的 cu121 栈不同）"
  pip install --upgrade --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu128

  install_verl_no_vllm

  echo
  echo "------------------------------------------------------------------"
  echo "Blackwell 说明：VERL 目前仅对接 pip 版 vllm 0.3.1/0.4.2/0.5.4/0.6.3。"
  echo "这些 wheel 针对 torch 2.4 构建，与当前 torch 2.12 nightly 的 ABI 不一致，"
  echo "直接 pip install vllm==0.6.3 会在 import 时失败。"
  echo "若要坚持 vLLM：需用**同一套** torch 从**源码**编译 vLLM，并确认"
  echo "verl/third_party/vllm 中对应版本的补丁仍适用（或升级上游 VERL 以支持新 vLLM）。"
  echo "在此之前，训练 rollout 请继续使用 HF（你现有 tmux 实验无需改动）。"
  echo "------------------------------------------------------------------"
  echo
fi

echo "==> 常用训练辅助"
pip install wandb matplotlib IPython

echo "==> 验证（不启动 Ray / 不跑训练；忽略外层 PYTHONPATH 以免误扫到仓库外目录）"
if [[ "${STACK}" == "legacy" ]]; then
  env -u PYTHONPATH python <<'PY'
import importlib.metadata as im
import torch
import vllm
from vllm import SamplingParams

print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("vllm (pip):", im.version("vllm"))
assert im.version("vllm").startswith("0.6.3"), "请使用 vllm==0.6.3 以匹配 VERL 白名单"
assert hasattr(SamplingParams(), "temperature")
if not torch.cuda.is_available():
    raise SystemExit("CUDA 不可用")
cap = torch.cuda.get_device_capability()
print("GPU capability (major, minor):", cap)
x = torch.zeros(1, device="cuda")
print("cuda tensor ok:", x.device)
import verl.third_party.vllm as v
print("verl third_party vllm bridge ok, vllm_version:", v.vllm_version)
PY
else
  env -u PYTHONPATH python <<'PY'
import importlib.metadata as im
import torch

print("torch:", torch.__version__, "cuda:", torch.version.cuda)
if not torch.cuda.is_available():
    raise SystemExit("CUDA 不可用")
print("GPU capability:", torch.cuda.get_device_capability())
x = torch.zeros(1, device="cuda")
print("cuda tensor ok:", x.device)
try:
    v = im.version("vllm")
except im.PackageNotFoundError:
    print("pip 未安装 vllm（Blackwell 分支预期；勿依赖外层 PYTHONPATH 里的同名目录）")
else:
    print("注意：pip 已安装 vllm", v, "请自行验证与当前 torch 的 ABI：python -c \"import verl.third_party.vllm\"")
PY
fi

echo
echo "完成。"
if [[ "${STACK}" == "legacy" ]]; then
  echo "之后： source ${VENV_DIR}/bin/activate"
  echo "       export VLLM_TRAIN_VENV=${VENV_DIR}"
  echo "       bash ${SEARCH_R1_ROOT}/train_econ_grpo_small_formal_v1_vllm.sh"
else
  echo "本机为 Blackwell：vLLM 训练脚本需待「与 torch 2.12 匹配的 vLLM 构建」就绪后再用。"
  echo "当前 venv 已具备与 HF 并行的 PyTorch+verl 依赖基础（仍未安装可用 pip vllm）。"
fi
echo "（单卡时勿与正在跑的 HF 训练同时占 GPU。）"
