FROM debian:13-slim

RUN apt-get update \
    && apt-get install --no-install-recommends --yes \
        mupdf-tools \
        pandoc \
        tesseract-ocr \
        tesseract-ocr-eng \
        tesseract-ocr-swe \
    && rm -rf /var/lib/apt/lists/*

USER nobody:nogroup
WORKDIR /workspace/source
