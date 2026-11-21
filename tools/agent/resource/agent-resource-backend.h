#pragma once

#include <string>
#include <vector>

enum class agent_resource_backend_kind {
    local_process,
    local_mupdf,
    local_ghostscript,
    local_tesseract,
    local_pandoc,
    docker,
    kubernetes,
    lxc,
};

struct agent_resource_backend_capabilities {
    // These are host-resolved runtime capabilities, not model decisions.
    // The host may derive them from explicit paths, PATH lookup, or an
    // approved sandbox backend after performing its own health/policy checks.
    bool has_local_process = true;
    bool has_mupdf = false;
    bool has_ghostscript = false;
    bool has_tesseract = false;
    bool has_pandoc = false;
    bool has_docker = false;
    bool has_kubernetes = false;
    bool has_lxc = false;
};

struct agent_resource_backend_candidate {
    agent_resource_backend_kind kind;
    std::string id;
};

struct agent_resource_backend_resolution {
    bool available = false;
    agent_resource_backend_kind kind = agent_resource_backend_kind::local_mupdf;
    std::string id;
};

inline bool is_agent_resource_backend_available(
        agent_resource_backend_kind kind,
        const agent_resource_backend_capabilities & capabilities) {
    switch (kind) {
        case agent_resource_backend_kind::local_process:
            return capabilities.has_local_process;
        case agent_resource_backend_kind::local_mupdf:
            return capabilities.has_mupdf;
        case agent_resource_backend_kind::local_ghostscript:
            return capabilities.has_ghostscript;
        case agent_resource_backend_kind::local_tesseract:
            return capabilities.has_tesseract;
        case agent_resource_backend_kind::local_pandoc:
            return capabilities.has_pandoc;
        case agent_resource_backend_kind::docker:
            return capabilities.has_docker;
        case agent_resource_backend_kind::kubernetes:
            return capabilities.has_kubernetes;
        case agent_resource_backend_kind::lxc:
            return capabilities.has_lxc;
    }
    return false;
}

// Resolve an already host-ordered list. The caller owns representation and
// policy matching; this helper only applies availability deterministically.
inline agent_resource_backend_resolution resolve_agent_resource_backend(
        const std::vector<agent_resource_backend_candidate> & candidates,
        const agent_resource_backend_capabilities & capabilities) {
    for (const auto & candidate : candidates) {
        if (candidate.id.empty() ||
                !is_agent_resource_backend_available(candidate.kind, capabilities)) {
            continue;
        }
        return {true, candidate.kind, candidate.id};
    }
    return {};
}
