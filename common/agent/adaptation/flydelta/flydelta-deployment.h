#pragma once

#include "agent/adaptation/flydelta/flydelta-activation.h"
#include "agent/adaptation/flydelta/flydelta-sideband-review-store.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <string>
#include <vector>

enum class common_flydelta_canary_mode {
    disabled,
    manual,
    policy,
};

const char * common_flydelta_canary_mode_name(common_flydelta_canary_mode mode);
bool parse_common_flydelta_canary_mode(
        const std::string & value, common_flydelta_canary_mode & mode, std::string & error);

enum class common_flydelta_canary_disposition {
    retain,
    expand_scope,
    promote_active,
    close,
};

const char * common_flydelta_canary_disposition_name(
        common_flydelta_canary_disposition disposition);

// Host-owned policy controls disposition only. It never changes FlyDelta
// search, Oracle truth, learning credit, or the ordinary active-only path.
struct common_flydelta_canary_policy {
    common_flydelta_canary_mode mode = common_flydelta_canary_mode::manual;
    size_t min_observations = 8;
    size_t min_unique_allocations = 2;
    float min_target_gain = 0.0f;
    float max_control_regression = 0.0f;
    float max_competitor_regression = 0.0f;
    size_t max_harmed_results = 0;
    std::vector<uint32_t> traffic_steps_basis_points = {100, 500, 1000};
    // Host-owned, preconfigured applicability steps. Policy may select only
    // one of these values; it never derives a scope from model output.
    std::vector<std::string> scope_step_fingerprints;
    uint64_t pristine_control_interval = 16;
    bool allow_scope_expansion = false;
    bool allow_promotion = false;
    bool auto_close_on_harmed = true;
};

bool common_flydelta_canary_policy_validate(
        const common_flydelta_canary_policy & policy, std::string & error);

struct common_flydelta_canary_policy_input {
    size_t completed_observations = 0;
    size_t unique_allocations = 0;
    size_t harmed_results = 0;
    float target_gain = 0.0f;
    float control_regression = 0.0f;
    float competitor_regression = 0.0f;
    uint32_t current_traffic_basis_points = 0;
    bool next_scope_available = false;
    bool semantic_evidence_complete = false;
};

struct common_flydelta_canary_policy_decision {
    common_flydelta_canary_disposition disposition =
        common_flydelta_canary_disposition::retain;
    uint32_t next_traffic_basis_points = 0;
    std::string reason;
};

// Deterministic projection of journaled canary observation/evaluation events.
// It is policy input, not a second semantic evidence store.
struct common_flydelta_canary_policy_snapshot {
    std::string canary_event_id;
    size_t reserved_observations = 0;
    size_t terminal_observations = 0;
    size_t evaluated_observations = 0;
    size_t unique_allocations = 0;
    size_t harmed_results = 0;
    float target_gain = 0.0f;
    float control_regression = 0.0f;
    float competitor_regression = 0.0f;
    bool semantic_evidence_complete = false;
};

bool common_flydelta_build_canary_policy_snapshot(
        const std::vector<common_flydelta_sideband_review> & reviews,
        const std::string & canary_event_id,
        common_flydelta_canary_policy_snapshot & snapshot,
        std::string & error);

bool common_flydelta_decide_canary_disposition(
        const common_flydelta_canary_policy & policy,
        const common_flydelta_canary_policy_input & input,
        common_flydelta_canary_policy_decision & decision,
        std::string & error);

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
    std::string observation_id;
};

struct common_flydelta_deployment_result {
    bool active_only = true;
    bool canary_considered = false;
    bool canary_selected = false;
    bool fallback_to_active = false;
    std::string fallback_reason;
    std::string baseline_deployment_fingerprint;
    std::string candidate_deployment_fingerprint;
    bool has_canary_evaluation_context = false;
    common_flydelta_canary_evaluation_context canary_evaluation_context;
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

// Reservation is deliberately separate from semantic evidence. It closes
// the race between cohort selection and generation without making a request
// count as HELPED or promotion evidence.
using common_flydelta_observation_reserver = std::function<bool(
        const common_flydelta_sideband_review & canary_review,
        const std::string & allocation_id,
        size_t max_observations,
        std::string & observation_id,
        std::string & error)>;

// Request-exposure accounting only. Semantic/counterfactual evidence remains
// in the existing lifecycle journal and is never inferred from this ledger.
class common_flydelta_observation_budget final {
public:
    bool try_reserve(
            const std::string & binding_key,
            const std::string & canary_event_id,
            size_t max_observations,
            std::string & error);
    void seed(
            const std::string & binding_key,
            const std::string & canary_event_id,
            size_t count);
    size_t reserved(
            const std::string & binding_key,
            const std::string & canary_event_id) const;

private:
    static std::string make_key(
            const std::string & binding_key,
            const std::string & canary_event_id);

    mutable std::mutex mutex_;
    std::unordered_map<std::string, size_t> reservations_;
};

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
    common_flydelta_observation_reserver reserve_observation;
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
