# syntax=docker/dockerfile:1

FROM rust:1.87-slim AS builder

RUN apt-get update && apt-get install -y pkg-config && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY Cargo.toml Cargo.lock ./
COPY src/ src/

RUN cargo build --release --locked

# ---

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/target/release/coproxy /usr/local/bin/coproxy

ENV RUST_LOG=info

EXPOSE 8080

ENTRYPOINT ["coproxy"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8080", "--api-surface", "all"]
