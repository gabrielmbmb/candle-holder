FROM lukemathwalker/cargo-chef:latest-rust-1.80.1 AS chef
WORKDIR /app

FROM chef AS planner
COPY . .
RUN cargo chef prepare --recipe-path recipe.json

FROM chef AS builder

# Define build argument
ARG BUILD_TARGET=cpu
ARG TARGETARCH
ARG TARGETOS

# Install build essentials
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    libssl-dev \
    && if [ "$TARGETARCH" != "arm64" ]; then \
    apt-get install -y intel-mkl-full; \
    fi \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=planner /app/recipe.json recipe.json

RUN cargo chef cook --release --recipe-path recipe.json

COPY . .

ENV RUSTFLAGS="-C target-cpu=native"
ENV OPENBLAS_NUM_THREADS=1
ENV BLAS_LIB=OpenBLAS

RUN if [ "$TARGETOS" = "darwin" ] && [ "$TARGETARCH" = "arm64" ]; then \
    cargo build --release --package candle-holder-serve --features accelerate; \
    elif [ "$TARGETARCH" = "arm64" ]; then \
    cargo build --release --package candle-holder-serve; \
    elif [ "$BUILD_TARGET" = "gpu" ]; then \
    cargo build --release --package candle-holder-serve --features gpu; \
    else \
    cargo build --release --package candle-holder-serve --features mkl; \
    fi

FROM ubuntu:22.04

ARG TARGETARCH

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libssl3 \
    ca-certificates \
    && if [ "$TARGETARCH" != "arm64" ]; then \
    apt-get install -y intel-mkl-full; \
    fi \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/candle-holder-serve /usr/local/bin/candle-holder-serve

COPY candle-holder-serve/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]

ENV CANDLE_HOLDER_HOST=0.0.0.0:8080
