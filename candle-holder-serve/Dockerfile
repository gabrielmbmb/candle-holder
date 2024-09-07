FROM lukemathwalker/cargo-chef:latest-rust-1.80.1 AS chef
WORKDIR /app

FROM chef AS planner
COPY . .
RUN cargo chef prepare --recipe-path recipe.json

FROM chef AS builder

ARG TARGETARCH
ARG TARGETOS

RUN apt-get update && apt-get install -y \
  build-essential \
  pkg-config \
  libssl-dev \
  wget \
  gnupg \
  && if [ "$TARGETARCH" != "arm64" ]; then \
  wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
  | gpg --dearmor | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null \
  && echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
  | tee /etc/apt/sources.list.d/oneAPI.list \
  && apt-get update \
  && apt-get install -y intel-oneapi-mkl-devel; \
  fi \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=planner /app/recipe.json recipe.json

RUN cargo chef cook --release --recipe-path recipe.json

COPY . .

ENV RUSTFLAGS="-C target-cpu=native"

RUN if [ "$TARGETOS" = "darwin" ] && [ "$TARGETARCH" = "arm64" ]; then \
  cargo build --release --package candle-holder-serve --features accelerate; \
  elif [ "$TARGETARCH" = "arm64" ]; then \
  cargo build --release --package candle-holder-serve; \
  else \
  cargo build --release --package candle-holder-serve --features mkl; \
  fi

FROM ubuntu:22.04

ARG TARGETARCH

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
  libomp-dev \
  && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/candle-holder-serve /usr/local/bin/candle-holder-serve

COPY candle-holder-serve/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]

ENV CANDLE_HOLDER_HOST=0.0.0.0:8080
