#pragma once

#include "agent/adaptation/flydelta/flydelta-activation.h"
#include "agent/adaptation/flydelta/flydelta-sideband-review-store.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

// Request-scoped host input for resolving the already configured profile.
// The factory never reads files or lets the model choose a revision; artifact
// materialization is supplied through the host-owned loader callback below.
struct common_flydelta_deployment_request {
    common_agent_model_profile profile;
    common_flydelta_compatibility compatibility;
    common_flydelta_applicability applicability;
    common_flydelta_runtime_authority authority = common_flydelta_runtime_authority::active_only;
    std::string allocation_key;
    uint64_t now_epoch_ms = 0;
    size_t observed_default = 0;
    size_t model_n_embd = 0;
    size_t model_n_layers = 0;
    std::string model_profile_fingerprint;
    std::string capture_layout_revision;
    common_flydelta_gate_config gate_config;
    common_flydelta_gate_request gate_request;
    common_flydelta_sparse_code code;
    size_t max_overlay_bytes = 0;
    size_t max_artifact_weights = 1U << 20;
    size_t max_artifact_bytes = 4U * 1024U * 1024U;
};

struct common_flydelta_resolved_deployment_entry {
    std::string binding_key;
    std::string revision_id;
    double scale = 1.0;
    bool canary = false;
    std::string canary_event_id;
};

struct common_flydelta_deployment_result {
    bool active_only = true;
    bool canary_considered = false;
    bool canary_selected = false;
    bool fallback_to_active = false;
    std::string fallback_reason;
    std::string baseline_deployment_fingerprint;
    std::string candidate_deployment_fingerprint;
    std::vector<common_flydelta_resolved_deployment_entry> baseline;
    std::vector<common_flydelta_resolved_deployment_entry> effective;
    std::shared_ptr<const common_flydelta_activation_result> activation;
};

using common_flydelta_open_canary_provider = std::function<bool(
        std::vector<common_flydelta_sideband_review> & reviews,
        std::string & error)>;

using common_flydelta_observation_counter = std::function<size_t(
        const std::string & binding_key,
        const std::string & canary_event_id)>;

// The host owns artifact-store access and converts one verified manifest into
// the existing activation result. This keeps paths, artifact loading and gate
// context outside the deployment resolver.
using common_flydelta_activation_loader = std::function<bool(
        const common_flydelta_sideband_manifest & manifest,
        double effective_scale,
        const common_flydelta_deployment_request & request,
        common_flydelta_activation_result & activation,
        std::string & error)>;

struct common_flydelta_deployment_factory_config {
    const common_flydelta_sideband_registry * registry = nullptr;
    common_flydelta_open_canary_provider open_canaries;
    common_flydelta_observation_counter observation_counter;
    common_flydelta_activation_loader load_activation;
};

// Resolves the profile's ordered sidebands, optionally replaces or extends
// them with an explicitly selected canary, and returns one immutable
// request-scoped activation for the existing server-context seam.
bool common_flydelta_resolve_deployment(
        const common_flydelta_deployment_factory_config & config,
        const common_flydelta_deployment_request & request,
        common_flydelta_deployment_result & result,
        std::string & error);

std::string common_flydelta_deployment_fingerprint(
        const common_flydelta_deployment_request & request,
        const std::vector<common_flydelta_resolved_deployment_entry> & entries);
