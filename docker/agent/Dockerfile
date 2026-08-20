ARG BASE_IMAGE=ubuntu:24.04
FROM ${BASE_IMAGE}

ARG PACKAGE_ARCHIVE=agent-package.tar.gz
ARG BACKEND=cpu

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install --no-install-recommends --yes \
        ca-certificates \
        libgomp1 \
        libssl3 \
        libvulkan1 \
        zlib1g \
    && rm -rf /var/lib/apt/lists/*

COPY ${PACKAGE_ARCHIVE} /tmp/agent-package.tar.gz

RUN mkdir -p /opt/llama-agent \
    && tar -xzf /tmp/agent-package.tar.gz -C /opt/llama-agent \
    && rm /tmp/agent-package.tar.gz \
    && useradd --create-home --home-dir /home/llama-agent --shell /usr/sbin/nologin llama-agent \
    && mkdir -p /etc/llama-agent /models /var/lib/llama-agent/data /var/log/llama-agent \
    && chown -R llama-agent:llama-agent /etc/llama-agent /models /var/lib/llama-agent /var/log/llama-agent

COPY agent-config.docker.example.json /opt/llama-agent/share/llama-agent/examples/agent-config.docker.example.json
COPY entrypoint.sh /usr/local/bin/llama-agent-entrypoint

RUN chmod 0755 /usr/local/bin/llama-agent-entrypoint

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
