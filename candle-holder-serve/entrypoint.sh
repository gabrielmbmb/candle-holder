#!/bin/bash

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

exec candle-holder-serve "${ARGS[@]}"
