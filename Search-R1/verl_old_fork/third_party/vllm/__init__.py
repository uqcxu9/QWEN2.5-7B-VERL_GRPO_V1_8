# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from importlib.metadata import version, PackageNotFoundError

try:
    # packaging is a transitive dependency in most envs; used to normalize local version suffixes (e.g. 0.6.3+cu129)
    from packaging.version import Version
except Exception:  # pragma: no cover
    Version = None  # type: ignore


def get_version(pkg):
    try:
        v = version(pkg)
        # Normalize versions like "0.6.3+cu129" to base "0.6.3" so we can match the supported shims.
        if Version is not None:
            try:
                vv = Version(v)
                if vv.local is not None:
                    return vv.public.split("+", 1)[0]
            except Exception:
                pass
        if "+" in v:
            return v.split("+", 1)[0]
        return v
    except PackageNotFoundError:
        # In some editable/source installs, the dist-info metadata may be missing
        # even though `import vllm` works. Try importing the module and reading
        # `__version__` as a fallback.
        try:
            if pkg != "vllm":
                return None
            import vllm as _vllm  # type: ignore

            v = getattr(_vllm, "__version__", None)
            if not v:
                return None
            if Version is not None:
                try:
                    vv = Version(v)
                    if vv.local is not None:
                        return vv.public.split("+", 1)[0]
                except Exception:
                    pass
            if "+" in v:
                return v.split("+", 1)[0]
            return v
        except Exception:
            return None


package_name = 'vllm'
package_version = get_version(package_name)

if package_version == '0.3.1':
    vllm_version = '0.3.1'
    from .vllm_v_0_3_1.llm import LLM
    from .vllm_v_0_3_1.llm import LLMEngine
    from .vllm_v_0_3_1 import parallel_state
elif package_version == '0.4.2':
    vllm_version = '0.4.2'
    from .vllm_v_0_4_2.llm import LLM
    from .vllm_v_0_4_2.llm import LLMEngine
    from .vllm_v_0_4_2 import parallel_state
elif package_version == '0.5.4':
    vllm_version = '0.5.4'
    from .vllm_v_0_5_4.llm import LLM
    from .vllm_v_0_5_4.llm import LLMEngine
    from .vllm_v_0_5_4 import parallel_state
elif package_version == '0.6.3':
    vllm_version = '0.6.3'
    from .vllm_v_0_6_3.llm import LLM
    from .vllm_v_0_6_3.llm import LLMEngine
    from .vllm_v_0_6_3 import parallel_state
else:
    if package_version is None:
        raise ValueError(
            'vllm is not installed in this Python environment (no pip distribution named "vllm"). '
            'Install a supported version (0.3.1, 0.4.2, 0.5.4, 0.6.3). '
            'On Blackwell (sm_120): do not use `pip install vllm==0.6.3` alone—it often pulls torch 2.4 '
            'without sm_120 kernels; use torch 2.12+cu128 (nightly) and build vLLM v0.6.3 from source, '
            'see Search-R1/scripts/install_vllm063_source_blackwell.sh. '
            'Or set rollout.name=hf until vLLM is ready.'
        )
    raise ValueError(
        f'vllm version {package_version} not supported. Currently supported versions are 0.3.1, 0.4.2, 0.5.4 and 0.6.3.'
    )
