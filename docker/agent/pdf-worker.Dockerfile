FROM debian:13-slim

RUN apt-get update \
    && apt-get install --no-install-recommends --yes mupdf-tools \
    && rm -rf /var/lib/apt/lists/*

USER nobody:nogroup
WORKDIR /workspace/source
