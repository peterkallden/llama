#include "../../daemon/agent-daemon-flydelta-internal.h"

namespace agent_daemon_flydelta_internal {

bool daemon_flydelta_lifecycle_record_matches_scope(
        const common_learning_lifecycle_record & record,
        const common_agent_scope & scope) {
    return record.namespace_id == scope.namespace_id &&
        record.project_id == scope.project_id &&
        record.session_id == scope.session_id;
}

// Resource authority is execution-scoped, not a property of the resident
// model. Worker callbacks receive the immutable job scope and use this
// lightweight provider view so concurrent jobs cannot overwrite one another's
// authority while sharing the same host and resource store.
std::shared_ptr<daemon_flydelta_resource_provider>
daemon_flydelta_provider_for_scope(
        const std::shared_ptr<daemon_flydelta_resource_provider> & source,
        const common_agent_scope & scope) {
    if (!source) return {};
    auto scoped = std::make_shared<daemon_flydelta_resource_provider>();
    scoped->host = source->host;
    scoped->resources = source->resources;
    scoped->authority.namespace_id = scope.namespace_id.empty()
        ? source->authority.namespace_id : scope.namespace_id;
    scoped->authority.project_id = scope.project_id;
    scoped->authority.session_id = scope.session_id.empty()
        ? source->authority.session_id : scope.session_id;
    scoped->authority.turn_id = scope.turn_id;
    scoped->authority.now = source->authority.now;
    scoped->model_profile_fingerprint = source->model_profile_fingerprint;
    scoped->capture_layout_revision = source->capture_layout_revision;
    scoped->teaching_material_runtime = source->teaching_material_runtime;
    scoped->n_predict = source->n_predict;
    scoped->n_threads = source->n_threads;
    scoped->model_n_embd = source->model_n_embd;
    scoped->model_n_layers = source->model_n_layers;
    scoped->lifecycle_store = source->lifecycle_store;
    scoped->resolve_bootstrap_zoom_state = source->resolve_bootstrap_zoom_state;
    scoped->resolve_bootstrap_zoom_state_for_job =
        source->resolve_bootstrap_zoom_state_for_job;
    scoped->resolve_representation_augmentation_state =
        source->resolve_representation_augmentation_state;
    scoped->resolve_representation_augmentation_state_for_job =
        source->resolve_representation_augmentation_state_for_job;
    scoped->resolve_search_orchestration_state = source->resolve_search_orchestration_state;
    scoped->resolve_search_orchestration_state_for_job =
        source->resolve_search_orchestration_state_for_job;
    return scoped;
}

} // namespace agent_daemon_flydelta_internal
