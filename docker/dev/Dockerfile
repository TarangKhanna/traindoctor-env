# DEV-mode image: repo-owned, reviewed Gloo test code on Linux CPU. Not a sandbox for untrusted/agent-edited code
# (use gVisor / Modal Sandbox for that). Build context is a temp dir holding only this file + pyproject.toml + uv.lock.
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 HF_HOME=/opt/hf HF_HUB_OFFLINE=1 LANG=C.UTF-8
RUN pip install uv==0.6.6
COPY pyproject.toml uv.lock /build/
# Third-party deps exactly as pinned in uv.lock, minus the Mac/CUDA-only ones; torch is the CPU wheel (same version).
RUN cd /build \
 && uv export --frozen --no-hashes --no-emit-project --no-header \
    | grep -v -E '^(torch|triton|nvidia-|cuda-|mlx|inductive-mlxrl)' > req.txt \
 && uv pip install --system --no-deps -r req.txt \
 && pip install "torch==$(grep -A1 '^name = \"torch\"' uv.lock | sed -n 's/version = \"\(.*\)\"/\1/p')" --index-url https://download.pytorch.org/whl/cpu
# Tokenizer files only (no weights) so executed token-cap tests really run offline; revision is recorded in the image.
RUN HF_HUB_OFFLINE=0 python -c "from huggingface_hub import snapshot_download as s; p=s('mlx-community/Qwen3-0.6B-4bit', allow_patterns=['*.json','*.txt'], ignore_patterns=['*.safetensors*']); open('/opt/hf/REVISION','w').write(p)"
RUN useradd -u 65534 -M nobody 2>/dev/null || true
WORKDIR /work
