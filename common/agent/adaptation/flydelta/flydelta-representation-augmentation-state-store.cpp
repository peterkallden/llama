#include "agent/adaptation/flydelta/flydelta-representation-augmentation-state-store.h"

#include "hash/hash.h"

#include <utility>

namespace {

bool bounded(const std::string & value, size_t maximum = 512) {
    return !value.empty() && value.size() <= maximum;
}

std::string hash_text(const std::string & value) {
    return "sha256:" + hash_sha256_hex(value.data(), value.size());
}

} // namespace

bool common_flydelta_configure_representation_augmentation_lifecycle_callbacks(
        common_learning_lifecycle_store & store,
        const common_flydelta_lifecycle_event_context & context,
        common_flydelta_evaluator_callbacks & callbacks,
        std::string & error) {
    error.clear();
    if (!bounded(context.source_id) || !bounded(context.created_at) ||
            context.scope.namespace_id.size() > 512 || context.scope.project_id.size() > 512 ||
            context.scope.session_id.size() > 512) {
        error = "FlyDelta representation augmentation lifecycle context is invalid";
        return false;
    }
    callbacks.resolve_representation_augmentation_state = [&store](
            const std::string & state_ref,
            common_flydelta_representation_augmentation_state & state,
            std::string & resolve_error) {
        if (state_ref.rfind("flydelta://state/representation-augmentation/", 0) != 0 ||
                state_ref.size() > 512) {
            resolve_error = "FlyDelta representation augmentation state reference is invalid";
            return false;
        }
        const auto records = store.list(resolve_error);
        if (!resolve_error.empty()) return false;
        for (auto it = records.rbegin(); it != records.rend(); ++it) {
            if (it->kind != common_learning_lifecycle_kind::flydelta_experiment ||
                    it->subject_id != state_ref) continue;
            if (!common_flydelta_representation_augmentation_state_from_json(
                    it->payload_json, state,
                    common_flydelta_representation_augmentation_config{}, resolve_error)) {
                return false;
            }
            if (state.state_ref != state_ref) {
                resolve_error = "FlyDelta representation augmentation state ref does not match";
                return false;
            }
            return true;
        }
        resolve_error = "FlyDelta representation augmentation state was not found";
        return false;
    };
    callbacks.persist_representation_augmentation_state = [&store, context](
            const common_flydelta_representation_augmentation_state & input,
            std::string & state_ref,
            std::string & persist_error) {
        auto state = input;
        if (!common_flydelta_representation_augmentation_state_validate(
                state, common_flydelta_representation_augmentation_config{}, persist_error)) {
            return false;
        }
        if (state.state_ref.empty()) {
            state.state_ref = "flydelta://state/representation-augmentation/" +
                hash_text(common_flydelta_representation_augmentation_state_to_json(state)).substr(7, 32);
        }
        if (!common_flydelta_representation_augmentation_state_validate(
                state, common_flydelta_representation_augmentation_config{}, persist_error)) {
            return false;
        }
        const auto payload = common_flydelta_representation_augmentation_state_to_json(state);
        common_learning_lifecycle_record record;
        record.event_id = "flydelta://event/representation-augmentation/" +
            hash_text(state.state_ref).substr(7, 32);
        record.subject_id = state.state_ref;
        record.kind = common_learning_lifecycle_kind::flydelta_experiment;
        record.status = common_learning_lifecycle_status::running;
        record.idempotency_key = "flydelta/representation-augmentation/" +
            hash_text(state.state_ref).substr(7, 32);
        record.source_id = context.source_id;
        record.namespace_id = context.scope.namespace_id;
        record.project_id = context.scope.project_id;
        record.session_id = context.scope.session_id;
        record.content_hash = hash_text(payload);
        record.created_at = context.created_at;
        record.payload_json = payload;
        if (!store.append(record, persist_error)) return false;
        state_ref = state.state_ref;
        return true;
    };
    return true;
}
