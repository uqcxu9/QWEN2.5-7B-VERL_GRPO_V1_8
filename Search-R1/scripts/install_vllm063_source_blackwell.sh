#!/usr/bin/env bash
# =============================================================================
# 在 Blackwell (sm_120) + torch 2.12 nightly cu128 环境下，从源码安装 vLLM v0.6.3，
# 以满足 VERL 对版本的白名单（importlib 报告 0.6.3）且不与 torch ABI 冲突。
#
# 前置：
#   - 已用 nightly cu128 装好 torch，且 python -c "import torch; print(torch.cuda.get_device_capability())"
#     输出 (12, 0) 且能在 GPU 上建张量。
#   - 系统有 nvcc / CUDA toolkit（与驱动匹配）。
#
# 用法：
#   export VLLM_BUILD_VENV=/workspace/venvs/blackwell_train
#   bash /workspace/Search-R1/scripts/install_vllm063_source_blackwell.sh
#
# 说明：
#   - 容器里 /usr/local/cuda 常为精简：缺 libnvrtc 时 CMake 报 CUDA_nvrtc_LIBRARY NOTFOUND。脚本会从 venv 内
#     nvidia-cuda-nvrtc-cu12 解析 libnvrtc 并传入 CMake。
#   - 勿对 vLLM 执行无 --no-build-isolation 的 pip install -e .，否则会拉 torch==2.4。
#   - 编译耗时较长；若仍失败请查日志与 CMakeError.log。
# =============================================================================
set -euo pipefail

VLLM_BUILD_VENV="${VLLM_BUILD_VENV:-/workspace/venvs/blackwell_train}"
VLLM_SRC_DIR="${VLLM_SRC_DIR:-/workspace/src/vllm-v0.6.3}"
VLLM_TAG="${VLLM_TAG:-v0.6.3}"
FLASH_ATTN_TAG="${FLASH_ATTN_TAG:-013f0c4fc47e6574060879d9734c1df8c5c273bd}"
FLASH_ATTN_SRC_DIR="${FLASH_ATTN_SRC_DIR:-/workspace/src/flash-attention-vllm-${FLASH_ATTN_TAG:0:8}}"

# Blackwell；若仅编译 CPU 扩展可去掉（不要用于本训练）
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
export MAX_JOBS="${MAX_JOBS:-8}"

# shellcheck source=/dev/null
source "${VLLM_BUILD_VENV}/bin/activate"

# vLLM CMakeLists 要求 CMake >= 3.26；系统 /usr/bin/cmake 常为 3.22，必须用 venv 里的 pip cmake
export PATH="${VLLM_BUILD_VENV}/bin:${PATH}"
# 避免把外层 PYTHONPATH 带进 vLLM 的 cmake（过长/无关路径）
unset PYTHONPATH

echo "==> cmake: $(command -v cmake) ($(cmake --version | head -1))"
echo "==> Python: $(which python)"
python -c "import torch; print('torch', torch.__version__, 'cap', torch.cuda.get_device_capability())"

echo "==> 卸载 pip vllm/xformers（避免与源码安装混用）"
pip uninstall -y vllm xformers 2>/dev/null || true

mkdir -p "$(dirname "${VLLM_SRC_DIR}")"
if [[ ! -d "${VLLM_SRC_DIR}/.git" ]]; then
  git clone --depth 1 --branch "${VLLM_TAG}" https://github.com/vllm-project/vllm.git "${VLLM_SRC_DIR}"
else
  cd "${VLLM_SRC_DIR}"
  git fetch --depth 1 origin tag "${VLLM_TAG}" || true
  git checkout "${VLLM_TAG}"
fi

cd "${VLLM_SRC_DIR}"

# vLLM v0.6.3 依赖的 vllm-project/flash-attention 在较新架构上（如 Blackwell sm_120 / cc 12.0）
# 会错误地只接受 sm8x/sm90，从而抛：
#   "FlashAttention only supports Ampere GPUs or newer."
# 这里提前拉取同 commit 的源码并打补丁，然后在编译 vLLM 时用 VLLM_FLASH_ATTN_SRC_DIR 指向本地源码。
mkdir -p "$(dirname "${FLASH_ATTN_SRC_DIR}")"
if [[ ! -d "${FLASH_ATTN_SRC_DIR}/.git" ]]; then
  git clone https://github.com/vllm-project/flash-attention.git "${FLASH_ATTN_SRC_DIR}"
fi
cd "${FLASH_ATTN_SRC_DIR}"
git fetch origin "${FLASH_ATTN_TAG}" >/dev/null 2>&1 || true
git checkout "${FLASH_ATTN_TAG}" >/dev/null 2>&1 || true

FLASH_API_CPP="${FLASH_ATTN_SRC_DIR}/csrc/flash_attn/flash_api.cpp"
if [[ -f "${FLASH_API_CPP}" ]]; then
  if ! grep -q "is_sm120_or_newer" "${FLASH_API_CPP}" 2>/dev/null; then
    echo "==> 补丁：vllm-flash-attn 放开 sm_120（Blackwell）架构 gate"
    # 在 3 处同样的 gate 前插入 is_sm120_or_newer，并把 TORCH_CHECK 条件改为允许 major>=12
    sed -i \
      -e 's/bool is_sm90 = dprops->major == 9 && dprops->minor == 0;/bool is_sm90 = dprops->major == 9 \&\& dprops->minor == 0;\n    bool is_sm120_or_newer = dprops->major >= 12;/' \
      -e 's/TORCH_CHECK(is_sm90 || is_sm8x, \"FlashAttention only supports Ampere GPUs or newer\\.\"\);/TORCH_CHECK(is_sm120_or_newer || is_sm90 || is_sm8x, \"FlashAttention only supports Ampere GPUs or newer.\"\);/g' \
      -e 's/TORCH_CHECK(is_sm90 || is_sm8x, \"bfloat16 is only supported on Ampere GPUs or newer\"\);/TORCH_CHECK(is_sm120_or_newer || is_sm90 || is_sm8x, \"bfloat16 is only supported on Ampere GPUs or newer\"\);/g' \
      "${FLASH_API_CPP}"
  fi
else
  echo "WARNING: 未找到 ${FLASH_API_CPP}，将使用 vLLM 默认 FetchContent 下载的 vllm-flash-attn" >&2
fi

# 导出给 vLLM 的 CMakeLists：优先使用本地 flash-attention 源码（含补丁），避免重新下载未打补丁的版本
export VLLM_FLASH_ATTN_SRC_DIR="${FLASH_ATTN_SRC_DIR}"
echo "==> VLLM_FLASH_ATTN_SRC_DIR -> ${VLLM_FLASH_ATTN_SRC_DIR}"

cd "${VLLM_SRC_DIR}"

# PyTorch 2.12 + Blackwell 会带 sm_120（12.0）；v0.6.3 默认 CUDA_SUPPORTED_ARCHS 只到 9.0，
# 交集为空会导致 CMake/编译阶段异常。为本地构建加入 12.0（上游未正式支持，属实验性补丁）。
_CMAKE_PATCH_MARK='9.0;12.0'
if ! grep -qF "${_CMAKE_PATCH_MARK}" CMakeLists.txt 2>/dev/null; then
  echo "==> 补丁：CMakeLists.txt 的 CUDA_SUPPORTED_ARCHS 增加 12.0（Blackwell）"
  sed -i 's/set(CUDA_SUPPORTED_ARCHS "7.0;7.5;8.0;8.6;8.9;9.0")/set(CUDA_SUPPORTED_ARCHS "7.0;7.5;8.0;8.6;8.9;9.0;12.0")/' CMakeLists.txt
fi

echo "==> 构建依赖（cmake>=3.26 供 Ninja 配置；须排在 PATH 前于 /usr/bin/cmake）"
pip install -U pip setuptools wheel packaging ninja 'cmake>=3.26'
python - <<'PY' || { echo "ERROR: cmake 需 >= 3.26，请确认 PATH 中 ${VLLM_BUILD_VENV}/bin 在 /usr/bin 之前" >&2; exit 1; }
import re, subprocess
out = subprocess.check_output(["cmake", "--version"], text=True)
m = re.search(r"version (\d+)\.(\d+)", out)
assert m, out
maj, minor = int(m.group(1)), int(m.group(2))
assert (maj, minor) >= (3, 26), f"cmake {maj}.{minor} < 3.26: {out!r}"
print("cmake ok:", out.splitlines()[0])
PY

command -v ninja >/dev/null || { echo "ERROR: 未找到 ninja，请确认 pip install ninja 已成功" >&2; exit 1; }
echo "==> ninja: $(command -v ninja)"

echo "==> 确保 venv 内有 NVRTC（供 CMake 链接 _core_C）"
pip install -q 'nvidia-cuda-nvrtc-cu12>=12.1' || true
echo "==> 确保 venv 内有 CUDA header 包（cublas/cusparse/cusolver 等）"
pip install -q 'nvidia-cublas-cu12>=12.1' 'nvidia-cusparse-cu12>=12.1' 'nvidia-cusolver-cu12>=12.1' || true

NVRTC_LIB="$("${VLLM_BUILD_VENV}/bin/python" - <<'PY'
import sys
from pathlib import Path

def pick(root: Path):
    d = root / "nvidia" / "cuda_nvrtc" / "lib"
    if not d.is_dir():
        return None
    cands = sorted(d.glob("libnvrtc.so*"))
    for c in cands:
        if "builtins" in c.name:
            continue
        try:
            r = c.resolve()
        except OSError:
            r = c
        if r.is_file():
            return str(r)
    return None

sp = Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
lib = pick(sp)
if lib:
    print(lib)
    raise SystemExit(0)
for p in ("/usr/local/cuda/lib64/libnvrtc.so.12", "/usr/local/cuda/lib64/libnvrtc.so"):
    if Path(p).is_file():
        print(p)
        raise SystemExit(0)
sys.stderr.write("ERROR: 找不到 libnvrtc。请: pip install nvidia-cuda-nvrtc-cu12\n")
sys.exit(1)
PY
)"
echo "==> CUDA_nvrtc_LIBRARY -> ${NVRTC_LIB}"

NVIDIA_CUDA_INCLUDE_BASE="$("${VLLM_BUILD_VENV}/bin/python" - <<'PY'
import sys
from pathlib import Path

sp = Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
nvidia = sp / "nvidia"
if not nvidia.is_dir():
    raise SystemExit(1)
print(str(nvidia))
PY
)"
echo "==> nvidia include base -> ${NVIDIA_CUDA_INCLUDE_BASE}"

echo "==> 补丁：setup.py 支持 VLLM_EXTRA_CMAKE_ARGS（传入 -DCUDA_nvrtc_LIBRARY=...）"
PATCH_VLLM_SRC="${VLLM_SRC_DIR}" "${VLLM_BUILD_VENV}/bin/python" - <<'PY'
import os
from pathlib import Path

p = Path(os.environ["PATCH_VLLM_SRC"]) / "setup.py"
text = p.read_text()
marker = "# searchr1-patch: VLLM_EXTRA_CMAKE_ARGS"
if marker in text:
    print("setup.py 已含补丁，跳过")
    raise SystemExit(0)
old = """        cmake_args += ['-DVLLM_PYTHON_PATH={}'.format(":".join(sys.path))]\n"""
if old not in text:
    raise SystemExit("setup.py 与预期不符，无法自动打补丁（请检查 vLLM 版本是否为 v0.6.3）")
new = old + f"""
        {marker}
        _vllm_xcmake = os.environ.get("VLLM_EXTRA_CMAKE_ARGS", "").strip()
        if _vllm_xcmake:
            import shlex
            cmake_args += shlex.split(_vllm_xcmake)
"""
p.write_text(text.replace(old, new, 1))
print("setup.py 已打补丁")
PY

VLLM_CUDA_EXTRA_INCLUDES="-I${NVIDIA_CUDA_INCLUDE_BASE}/cuda_runtime/include -I${NVIDIA_CUDA_INCLUDE_BASE}/cublas/include -I${NVIDIA_CUDA_INCLUDE_BASE}/cusparse/include -I${NVIDIA_CUDA_INCLUDE_BASE}/cusolver/include -I${NVIDIA_CUDA_INCLUDE_BASE}/cuda_nvrtc/include"
# 注意：这里必须把带空格的 flags 作为一个 cmake 参数传递，否则会被拆成独立的 -I... 并被 cmake 视为未知参数。
# 利用 :STRING 并用双引号包起来，让 shlex.split 仍能保持它为单个 token。
# 同时补 C++ 编译（vllm-flash-attn 的部分文件是纯 C++ 编译，也会 include ATen/cuda 进而需要 cusparse/cusolver 头）。
export VLLM_EXTRA_CMAKE_ARGS="-DCUDA_nvrtc_LIBRARY=${NVRTC_LIB} -DCMAKE_CUDA_FLAGS:STRING=\"${VLLM_CUDA_EXTRA_INCLUDES}\" -DCMAKE_CXX_FLAGS:STRING=\"${VLLM_CUDA_EXTRA_INCLUDES}\""
echo "==> VLLM_EXTRA_CMAKE_ARGS -> ${VLLM_EXTRA_CMAKE_ARGS}"

echo "==> 源码安装 vLLM（链接当前 venv 内的 torch；显式 PATH 防 pip 子进程用到 /usr/bin/cmake）"
# 若之前编译/安装过，可能会复用旧的 CMake/Ninja 缓存，导致补丁不生效。
# 这里尽量清理 vLLM 与 vllm-flash-attn 的 build 目录，强制重新配置/编译。
rm -rf "${VLLM_SRC_DIR}/build" \
       "${VLLM_SRC_DIR}/.setuptools-cmake-build" \
       "${VLLM_SRC_DIR}/.ninja_log" "${VLLM_SRC_DIR}/.ninja_deps" \
       "${VLLM_SRC_DIR}/.deps/vllm-flash-attn-build" \
       "${VLLM_SRC_DIR}/.deps/vllm-flash-attn-subbuild" \
       "${VLLM_SRC_DIR}/.deps/vllm-flash-attn-subbuild" \
       "${VLLM_SRC_DIR}/.deps/vllm-flash-attn-subbuild" 2>/dev/null || true
# --no-build-isolation：使用已安装的 torch 参与编译
# 仍失败时：VLLM_VERBOSE_BUILD=1 并把输出重定向到文件，在 build/temp*/CMakeFiles/CMakeError.log 查看详情
# 关键：用 SETUPTOOLS_SCM_PRETEND_VERSION 强制版本为 0.6.3，匹配 VERL 白名单（否则会出现 0.6.4.dev0+...）
export SETUPTOOLS_SCM_PRETEND_VERSION="0.6.3"
echo "==> 编译日志将写入 /tmp/vllm_build.log"
# pip 26 的 editable 安装会强制走 PEP517 editable_wheel，容易导致构建隔离环境（/tmp/pip-build-env-*/）
# 从而丢失我们需要的 CUDA_nvrtc_LIBRARY / include flags 注入。
# 这里改为“非 editable”本地安装：直接 build 并安装 wheel/扩展到 site-packages，最稳。
PATH="${VLLM_BUILD_VENV}/bin:${PATH}" pip install . --no-build-isolation --no-deps -v 2>&1 | tee /tmp/vllm_build.log

echo "==> 验证 pip 版本与 VERL 桥"
python -c "import importlib.metadata as m; print('pip vllm', m.version('vllm'))"

echo "==> 确保当前 venv 内可 import verl（Search-R1/verl 是本仓库包名）"
pip install -e /workspace/Search-R1 --no-deps -q || true
python -c "import verl; import verl.third_party.vllm as v; print('verl bridge', v.vllm_version)"

echo "完成。若 import 失败，请把完整编译日志末尾几十行保存下来再排查。"
