#include "agent-resource-backend.h"

#include <cstdio>
#include <vector>

int main() {
    const std::vector<agent_resource_backend_candidate> ordered{
        {agent_resource_backend_kind::local_mupdf, "pdf-render.mupdf"},
        {agent_resource_backend_kind::local_ghostscript, "pdf-render.ghostscript"},
        {agent_resource_backend_kind::docker, "pdf-render.docker"},
        {agent_resource_backend_kind::kubernetes, "pdf-render.kubernetes"},
        {agent_resource_backend_kind::lxc, "pdf-render.lxc"},
    };

    agent_resource_backend_capabilities local;
    local.has_mupdf = true;
    local.has_ghostscript = true;
    local.has_docker = true;
    local.has_kubernetes = true;
    const auto local_result = resolve_agent_resource_backend(ordered, local);
    if (!local_result.available || local_result.id != "pdf-render.mupdf") {
        std::fprintf(stderr, "local backend priority was not preserved\n");
        return 1;
    }

    agent_resource_backend_capabilities isolated;
    isolated.has_docker = true;
    isolated.has_kubernetes = true;
    const auto isolated_result = resolve_agent_resource_backend(ordered, isolated);
    if (!isolated_result.available || isolated_result.id != "pdf-render.docker") {
        std::fprintf(stderr, "isolated backend fallback was not deterministic\n");
        return 1;
    }

    const auto unavailable_result = resolve_agent_resource_backend(ordered, {});
    if (unavailable_result.available) {
        std::fprintf(stderr, "unavailable backend set was accepted\n");
        return 1;
    }

    std::printf("local=%s isolated=%s\n",
        local_result.id.c_str(), isolated_result.id.c_str());
    return 0;
}
