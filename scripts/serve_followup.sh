#!/bin/bash
# Follow-up serve script. Knobs via env:
#   FU_PORT        HTTP port (default 8140)
#   FU_ENGINE_PORT VLLM_PORT, engine-internal (default FU_PORT+100)
#   FU_FBW         VLLM_DEBUG_FBW level: 1 (full trace) | 2 (META only) | "" (off)
#   FU_MAMBA_MODE  align (default) | none
#   FU_BLOCKS      value for --num-gpu-blocks-override ("" = engine default)
#   FU_MAX_SEQS    --max-num-seqs (default 64)
# Everything else is identical to serve_fbw.sh / serve_27b_main.sh.
set -u
PORT=${FU_PORT:-8140}
EPORT=${FU_ENGINE_PORT:-$((PORT + 100))}
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_PORT=$EPORT
export VLLM_COMPUTE_NANS_IN_LOGITS=1
export VLLM_DEBUG_GDN_PREFIX_STATE=1
export VLLM_DEBUG_FBW=${FU_FBW:-1}
export VLLM_USE_FLASHINFER_SAMPLER=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_HOME=/usr/local/cuda-13.1
export PATH=<VENV>/bin:/usr/local/cuda-13.1/bin:$PATH
L=<BUILDDIR>/cache_27bmain
export VLLM_CACHE_ROOT=$L/vllm
export TRITON_CACHE_DIR=$L/triton
export TORCHINDUCTOR_CACHE_DIR=$L/inductor
export CUDA_CACHE_PATH=$L/nv
mkdir -p "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_CACHE_PATH"
cd <WORKDIR>/worktrees/vllm-main
echo "SERVE27B_START $(hostname) $(date +%T) port=$PORT fbw=${FU_FBW:-1} mamba=${FU_MAMBA_MODE:-align} blocks=${FU_BLOCKS:-default}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
EXTRA=()
if [ -n "${FU_BLOCKS:-}" ]; then EXTRA+=(--num-gpu-blocks-override "${FU_BLOCKS}"); fi
exec <VENV>/bin/python -m vllm.entrypoints.openai.api_server \
  --model <MODEL_DIR>/Qwen__Qwen3.8-27B --served-model-name qwen38 \
  --host 127.0.0.1 --port "$PORT" --tensor-parallel-size 4 --trust-remote-code \
  --dtype bfloat16 --enable-prefix-caching --gpu-memory-utilization 0.88 --max-model-len 32768 \
  --max-num-batched-tokens 8192 --max-num-seqs "${FU_MAX_SEQS:-64}" \
  --speculative-config '{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":5,"prompt_lookup_min":2}' \
  --mamba-cache-mode "${FU_MAMBA_MODE:-align}" --no-enable-log-requests "${EXTRA[@]}"
