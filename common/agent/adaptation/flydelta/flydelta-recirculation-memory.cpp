#include "agent/adaptation/flydelta/flydelta-recirculation-memory.h"

#include "hash/hash.h"
#include <nlohmann/json.hpp>

#include <chrono>
#include <cmath>

namespace {
using json = nlohmann::json;

bool bounded(const std::string & value) { return !value.empty() && value.size() <= 512; }

bool valid_scope(const common_agent_scope & scope) {
    return bounded(scope.namespace_id) && scope.project_id.size() <= 512 &&
        ((scope.memory_scope == common_memory_scope::session && bounded(scope.session_id)) ||
         (scope.memory_scope == common_memory_scope::project && bounded(scope.project_id)));
}

std::string journal_session(const common_agent_scope & scope) {
    return scope.memory_scope == common_memory_scope::project ? "" : scope.session_id;
}

json config_json(const common_agent_recirculation_request & request) {
    return {{"source_layer", request.source_layer}, {"destination_layer", request.destination_layer},
        {"alpha", request.alpha}, {"rounds", request.rounds}, {"window_tokens", request.window_tokens},
        {"target", static_cast<unsigned>(request.target)}, {"threshold", request.threshold}};
}

bool parse_config(const json & value, common_agent_recirculation_request & request) {
    if (!value.is_object()) return false;
    try {
        request.source_layer = value.at("source_layer").get<uint32_t>();
        request.destination_layer = value.at("destination_layer").get<uint32_t>();
        request.alpha = value.at("alpha").get<float>();
        request.rounds = value.at("rounds").get<uint32_t>();
        request.window_tokens = value.at("window_tokens").get<uint32_t>();
        const auto target = value.at("target").get<unsigned>();
        request.target = static_cast<common_agent_recirculation_target>(target);
        request.threshold = value.at("threshold").get<float>();
        return target <= static_cast<unsigned>(common_agent_recirculation_target::curvature) &&
            request.source_layer > request.destination_layer &&
            request.rounds >= 1 && request.rounds <= 3 &&
            request.window_tokens >= 4 && request.window_tokens <= 8 &&
            std::isfinite(request.alpha) && request.alpha >= 0 &&
            std::isfinite(request.threshold) && request.threshold >= 0;
    } catch (const json::exception &) { return false; }
}

std::string hash_text(const std::string & value) {
    return "sha256:" + hash_sha256_hex(value.data(), value.size());
}
}

bool common_flydelta_recirculation_remember(
        common_learning_lifecycle_store & journal, const common_agent_scope & scope,
        const std::string & model_profile_fingerprint, const std::string & context_key,
        const common_agent_recirculation_request & request,
        const common_agent_recirculation_observation & observation, std::string & error) {
    error.clear();
    common_agent_recirculation_request validated;
    if (!valid_scope(scope) ||
            !bounded(model_profile_fingerprint) || !bounded(context_key) ||
            !parse_config(config_json(request), validated) ||
            !observation.attempted || !observation.committed || observation.target_position < 0 ||
            observation.completed_rounds != request.rounds ||
            !std::isfinite(observation.baseline_entropy) || !std::isfinite(observation.refined_entropy) ||
            observation.baseline_entropy - observation.refined_entropy < 0.001) {
        error = "recirculation memory requires a committed, scope-bound confidence observation";
        return false;
    }
    const json payload = {{"record_type", "experimental_recirculation_profile"},
        {"schema_version", 1}, {"model_profile_fingerprint", model_profile_fingerprint},
        {"context_key", context_key}, {"config", config_json(request)},
        {"baseline_entropy", observation.baseline_entropy},
        {"refined_entropy", observation.refined_entropy},
        {"target_position", observation.target_position}};
    const std::string text = payload.dump();
    const std::string identity = hash_text(scope.namespace_id + "\n" + scope.project_id + "\n" +
        journal_session(scope) + "\n" + model_profile_fingerprint + "\n" + context_key + "\n" +
        config_json(request).dump());
    bool exists = false;
    if (!journal.contains_idempotency(identity, exists, error)) return false;
    if (exists) return true;
    common_learning_lifecycle_record record;
    record.event_id = "flydelta://event/recirculation/" + identity.substr(7, 32);
    record.subject_id = "flydelta://recirculation/" + identity.substr(7, 32);
    record.kind = common_learning_lifecycle_kind::flydelta_experiment;
    record.status = common_learning_lifecycle_status::observed;
    record.idempotency_key = identity;
    record.source_id = "agent-recirculation";
    record.namespace_id = scope.namespace_id;
    record.project_id = scope.project_id;
    record.session_id = journal_session(scope);
    record.content_hash = hash_text(text);
    record.created_at = "unix-ns:" + std::to_string(
        std::chrono::system_clock::now().time_since_epoch().count());
    record.payload_json = text;
    return journal.append(record, error);
}

bool common_flydelta_recirculation_recall(
        const common_learning_lifecycle_store & journal, const common_agent_scope & scope,
        const std::string & model_profile_fingerprint, const std::string & context_key,
        common_agent_recirculation_request & request, bool & found, std::string & error) {
    found = false;
    error.clear();
    if (!valid_scope(scope) ||
            !bounded(model_profile_fingerprint) || !bounded(context_key)) {
        error = "recirculation recall requires a complete experiment identity";
        return false;
    }
    std::vector<common_learning_lifecycle_record> records;
    uint64_t watermark = 0;
    if (!journal.list_scope_with_watermark(scope.namespace_id, scope.project_id,
            journal_session(scope), records, watermark, error)) return false;
    for (auto it = records.rbegin(); it != records.rend(); ++it) {
        const auto & record = *it;
        if (record.kind != common_learning_lifecycle_kind::flydelta_experiment ||
                record.status != common_learning_lifecycle_status::observed) continue;
        const auto payload = json::parse(record.payload_json, nullptr, false);
        if (!payload.is_object() || !payload.contains("record_type") ||
                !payload.at("record_type").is_string() ||
                !payload.contains("model_profile_fingerprint") ||
                !payload.at("model_profile_fingerprint").is_string() ||
                payload.at("model_profile_fingerprint") != model_profile_fingerprint ||
                !payload.contains("context_key") || !payload.at("context_key").is_string() ||
                payload.at("context_key") != context_key) continue;
        if (payload.at("record_type") == "experimental_recirculation_suppression") return true;
        if (payload.at("record_type") != "experimental_recirculation_profile") continue;
        common_agent_recirculation_request candidate;
        if (!payload.contains("config") || !parse_config(payload.at("config"), candidate)) {
            error = "stored recirculation profile is invalid";
            return false;
        }
        request = candidate;
        found = true;
        return true;
    }
    return true;
}

bool common_flydelta_recirculation_suppress(
        common_learning_lifecycle_store & journal, const common_agent_scope & scope,
        const std::string & model_profile_fingerprint, const std::string & context_key,
        const common_agent_recirculation_request & request, std::string & error) {
    error.clear();
    common_agent_recirculation_request validated;
    if (!valid_scope(scope) || !bounded(model_profile_fingerprint) || !bounded(context_key) ||
            !parse_config(config_json(request), validated)) {
        error = "recirculation suppression needs a valid scoped profile";
        return false;
    }
    const json payload = {{"record_type", "experimental_recirculation_suppression"},
        {"schema_version", 1}, {"model_profile_fingerprint", model_profile_fingerprint},
        {"context_key", context_key}, {"config", config_json(request)},
        {"reason", "recalled_profile_did_not_pass_entropy_gain"}};
    const std::string text = payload.dump();
    const std::string identity = hash_text(scope.namespace_id + "\n" + scope.project_id + "\n" +
        journal_session(scope) + "\n" + model_profile_fingerprint + "\n" + context_key + "\n" +
        config_json(request).dump() + "\nsuppression");
    bool exists = false;
    if (!journal.contains_idempotency(identity, exists, error)) return false;
    if (exists) return true;
    common_learning_lifecycle_record record;
    record.event_id = "flydelta://event/recirculation/" + identity.substr(7, 32);
    record.subject_id = "flydelta://recirculation/" + identity.substr(7, 32);
    record.kind = common_learning_lifecycle_kind::flydelta_experiment;
    record.status = common_learning_lifecycle_status::observed;
    record.idempotency_key = identity;
    record.source_id = "agent-recirculation";
    record.namespace_id = scope.namespace_id;
    record.project_id = scope.project_id;
    record.session_id = journal_session(scope);
    record.content_hash = hash_text(text);
    record.created_at = "unix-ns:" + std::to_string(
        std::chrono::system_clock::now().time_since_epoch().count());
    record.payload_json = text;
    return journal.append(record, error);
}
