ARG BASE_IMAGE=ubuntu:24.04

FROM node:22-bookworm AS web-build

ARG PACKAGE_ARCHIVE=agent-package.tar.gz
COPY ${PACKAGE_ARCHIVE} /tmp/agent-package.tar.gz

RUN mkdir -p /opt/llama-agent \
    && tar -xzf /tmp/agent-package.tar.gz -C /opt/llama-agent \
    && cd /opt/llama-agent/share/llama-agent/examples/llama-agent.web \
    && npm ci --ignore-scripts \
    && npm run build

FROM ${BASE_IMAGE}

ARG PACKAGE_ARCHIVE=agent-package.tar.gz
ARG BACKEND=cpu
ARG INSTALL_DEV_TOOLS=false
ARG CONFIG_FILE=agent-config.docker.example.json
ARG IMAGE_CHANNEL=release

ENV DEBIAN_FRONTEND=noninteractive
ENV LLAMA_AGENT_IMAGE_CHANNEL=${IMAGE_CHANNEL}

RUN apt-get update \
    && apt-get install --no-install-recommends --yes \
        ca-certificates \
        libgomp1 \
        libssl3 \
        libvulkan1 \
        nginx \
        openssl \
        gettext-base \
        python3 \
        zlib1g \
    && if [ "$INSTALL_DEV_TOOLS" = "true" ]; then \
        apt-get install --no-install-recommends --yes \
            curl git build-essential cmake ninja-build pkg-config \
            clang clangd gdb pandoc tesseract-ocr mupdf-tools ghostscript; \
    fi \
    && rm -rf /var/lib/apt/lists/*

COPY ${PACKAGE_ARCHIVE} /tmp/agent-package.tar.gz
COPY --from=web-build /opt/llama-agent/share/llama-agent/examples/llama-agent.web/dist /opt/llama-agent-web/dist
COPY nginx/llama-agent-web.http.conf /opt/llama-agent/nginx/llama-agent-web.http.conf
COPY nginx/llama-agent-web.https.conf.template /opt/llama-agent/nginx/llama-agent-web.https.conf.template

RUN mkdir -p /opt/llama-agent \
    && tar -xzf /tmp/agent-package.tar.gz -C /opt/llama-agent \
    && rm /tmp/agent-package.tar.gz \
    && test -x /opt/llama-agent/bin/llama-agent-provider-bootstrap \
    && test -f /opt/llama-agent/bin/agent-provider-bootstrap.py \
    && test -f /opt/llama-agent/share/llama-agent/examples/agent-host-config-openalex.json \
    && test -f /opt/llama-agent/share/llama-agent/examples/openalex-works-openapi.json \
    && test -f /opt/llama-agent/share/llama-agent/examples/openapi-openalex-generated.json \
    && useradd --create-home --home-dir /home/llama-agent --shell /usr/sbin/nologin llama-agent \
    && mkdir -p /etc/llama-agent /models /var/lib/llama-agent/data /var/log/llama-agent \
        /var/lib/llama-agent/nginx/client_body \
        /var/lib/llama-agent/nginx/proxy \
        /var/lib/llama-agent/nginx/fastcgi \
        /var/lib/llama-agent/nginx/uwsgi \
        /var/lib/llama-agent/nginx/scgi \
    && chown -R llama-agent:llama-agent /etc/llama-agent /models /var/lib/llama-agent /var/log/llama-agent

COPY ${CONFIG_FILE} /opt/llama-agent/share/llama-agent/examples/agent-config.docker.example.json
COPY entrypoint.sh /usr/local/bin/llama-agent-entrypoint
COPY capability-smoke.sh /usr/local/bin/llama-agent-capability-smoke

RUN chmod 0755 /usr/local/bin/llama-agent-entrypoint /usr/local/bin/llama-agent-capability-smoke \
    && chmod 0755 /opt/llama-agent /opt/llama-agent/nginx \
    && chown -R llama-agent:llama-agent /opt/llama-agent-web /opt/llama-agent/nginx \
    && if [ "$INSTALL_DEV_TOOLS" = "true" ]; then \
        /usr/local/bin/llama-agent-capability-smoke; \
    fi

ENV PATH=/opt/llama-agent/bin:${PATH}
ENV LD_LIBRARY_PATH=/opt/llama-agent/lib

# Mount models read-only, while the configuration, stores and daemon log are
# retained outside the container lifecycle. Bind mounts or named volumes can
# override these.
VOLUME ["/models", "/etc/llama-agent", "/var/lib/llama-agent/data", "/var/log/llama-agent"]

USER llama-agent
WORKDIR /var/lib/llama-agent

ENTRYPOINT ["/usr/local/bin/llama-agent-entrypoint"]
CMD ["llama-agent-daemon"]

EXPOSE 8080 8081 8443
