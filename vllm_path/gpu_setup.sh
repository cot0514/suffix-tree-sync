#!/bin/bash
# GPU 실측용 환경: vLLM + arctic-inference를 새 venv(~/venv_vllm)에 설치. GPU는 쓰지 않음.
set -e
python3 -m venv ~/venv_vllm
source ~/venv_vllm/bin/activate
pip install -U pip
pip install vllm
pip install "arctic-inference[vllm]"
python -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__)"
pip list 2>/dev/null | grep -i -E "vllm|arctic|torch|numpy"
