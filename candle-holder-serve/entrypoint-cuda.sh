#!/bin/bash

# Based on https://github.com/huggingface/text-embeddings-inference/blob/main/cuda-all-entrypoint.sh
if ! command -v nvidia-smi &>/dev/null; then
  echo "Error: 'nvidia-smi' command not found."
  exit 1
fi

compute_cap=$(nvidia-smi --query-gpu=compute_cap --format=csv | sed -n '2p' | sed 's/\.//g')

ARGS=()

if [ -z "$CANDLE_HOLDER_MODEL" ] || [ -z "$CANDLE_HOLDER_PIPELINE" ]; then
  echo "Error: Both CANDLE_HOLDER_MODEL and CANDLE_HOLDER_PIPELINE must be provided."
  exit 1
fi

if [ ! -z "$CANDLE_HOLDER_MODEL" ]; then
  ARGS+=("--model" "$CANDLE_HOLDER_MODEL")
fi

if [ ! -z "$CANDLE_HOLDER_PIPELINE" ]; then
  ARGS+=("--pipeline" "$CANDLE_HOLDER_PIPELINE")
fi

if [ ! -z "$CANDLE_HOLDER_DEVICE" ]; then
  ARGS+=("--device" "$CANDLE_HOLDER_DEVICE")
fi

if [ ! -z "$CANDLE_HOLDER_DTYPE" ]; then
  ARGS+=("--dtype" "$CANDLE_HOLDER_DTYPE")
fi

if [ ! -z "$CANDLE_HOLDER_NUM_WORKERS" ]; then
  ARGS+=("--num-workers" "$CANDLE_HOLDER_NUM_WORKERS")
fi

if [ ! -z "$CANDLE_HOLDER_BUFFER_SIZE" ]; then
  ARGS+=("--buffer-size" "$CANDLE_HOLDER_BUFFER_SIZE")
fi

CANDLE_HOLDER_HOST=${CANDLE_HOLDER_HOST:-0.0.0.0:8080}
ARGS+=("--host" "$CANDLE_HOLDER_HOST")

if [ ${compute_cap} -eq 75 ]; then
  exec candle-holder-serve-cuda75 "${ARGS[@]}"
elif [ ${compute_cap} -ge 80 -a ${compute_cap} -lt 90 ]; then
  exec candle-holder-serve-cuda80 "${ARGS[@]}"
elif [ ${compute_cap} -eq 90 ]; then
  exec candle-holder-serve-cuda90 "${ARGS[@]}"
else
  echo "cuda compute cap ${compute_cap} is not supported"
  exit 1
fi
