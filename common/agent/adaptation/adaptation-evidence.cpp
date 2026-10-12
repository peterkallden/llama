#include "agent/adaptation/adaptation-evidence.h"

#include <nlohmann/json.hpp>

using json = nlohmann::ordered_json;

namespace {

bool bounded_nonempty(const std::string & value, size_t max_size = 512) {
    return !value.empty() && value.size() <= max_size;
}

std::optional<common_adaptation_evidence_source> parse_source(const std::string & value) {
    const common_adaptation_evidence_source sources[] = {
        common_adaptation_evidence_source::tool_repair,
        common_adaptation_evidence_source::reflection_alternative,
        common_adaptation_evidence_source::planning_revision,
        common_adaptation_evidence_source::research_alternative,
        common_adaptation_evidence_source::dataset_resource,
        common_adaptation_evidence_source::workflow_code,
        common_adaptation_evidence_source::procedure_blueprint,
        common_adaptation_evidence_source::user_correction,
        common_adaptation_evidence_source::user_taught_concept,
    };
    for (const auto source : sources) {
        if (value == common_adaptation_evidence_source_name(source)) return source;
    }
    return std::nullopt;
}

common_learning_cause parse_cause(const std::string & value) {
    if (value == "model_behavior") return common_learning_cause::model_behavior;
    if (value == "host_contract") return common_learning_cause::host_contract;
    if (value == "policy") return common_learning_cause::policy;
    if (value == "missing_evidence") return common_learning_cause::missing_evidence;
    if (value == "project_knowledge") return common_learning_cause::project_knowledge;
    return common_learning_cause::unknown;
}

} // namespace

const char * common_adaptation_evidence_source_name(
        common_adaptation_evidence_source source) {
    switch (source) {
        case common_adaptation_evidence_source::tool_repair: return "tool_repair";
        case common_adaptation_evidence_source::reflection_alternative: return "reflection_alternative";
        case common_adaptation_evidence_source::planning_revision: return "planning_revision";
        case common_adaptation_evidence_source::research_alternative: return "research_alternative";
        case common_adaptation_evidence_source::dataset_resource: return "dataset_resource";
        case common_adaptation_evidence_source::workflow_code: return "workflow_code";
        case common_adaptation_evidence_source::procedure_blueprint: return "procedure_blueprint";
        case common_adaptation_evidence_source::user_correction: return "user_correction";
        case common_adaptation_evidence_source::user_taught_concept: return "user_taught_concept";
    }
    return "tool_repair";
}

std::optional<common_adaptation_evidence_source>
common_adaptation_evidence_source_from_name(const std::string & value) {
    return parse_source(value);
}

std::optional<common_adaptation_evidence_source>
common_adaptation_evidence_source_for_signal(
        common_learning_signal_type type) {
    switch (type) {
        case common_learning_signal_type::tool_failure:
        case common_learning_signal_type::successful_recovery:
            return common_adaptation_evidence_source::tool_repair;
        case common_learning_signal_type::repair_echo_failure:
            return common_adaptation_evidence_source::tool_repair;
        case common_learning_signal_type::reflection_hint:
            return common_adaptation_evidence_source::reflection_alternative;
        case common_learning_signal_type::user_correction:
            return common_adaptation_evidence_source::user_correction;
        case common_learning_signal_type::user_taught_concept:
            return common_adaptation_evidence_source::user_taught_concept;
        case common_learning_signal_type::planning_revision:
            return common_adaptation_evidence_source::planning_revision;
        case common_learning_signal_type::research_verification:
            return common_adaptation_evidence_source::research_alternative;
        case common_learning_signal_type::procedure_verification:
        case common_learning_signal_type::blueprint_verification:
            return common_adaptation_evidence_source::procedure_blueprint;
    }
    return std::nullopt;
}

bool common_adaptation_evidence_validate(
        const common_adaptation_evidence & evidence,
        size_t max_transactions,
        std::string & error) {
    error.clear();
    if (evidence.schema_version != 2 || !bounded_nonempty(evidence.id) ||
            evidence.scope.namespace_id.empty() || evidence.scope.session_id.empty() ||
            !bounded_nonempty(evidence.behavior_key) ||
            !bounded_nonempty(evidence.task_fingerprint) ||
            !bounded_nonempty(evidence.baseline_ref) ||
            !bounded_nonempty(evidence.candidate_ref) ||
            evidence.baseline_ref == evidence.candidate_ref ||
            evidence.transaction_ids.size() > max_transactions) {
        error = "adaptation evidence identity or bounds are invalid";
        return false;
    }
    for (const auto & transaction_id : evidence.transaction_ids) {
        if (!bounded_nonempty(transaction_id)) {
            error = "adaptation evidence contains an invalid transaction reference";
            return false;
        }
    }
    if (evidence.host_verified && !bounded_nonempty(evidence.verifier_ref)) {
        error = "host-verified adaptation evidence requires a verifier reference";
        return false;
    }
    return true;
}

std::string common_adaptation_evidence_to_json(
        const common_adaptation_evidence & evidence) {
    return json{
        {"schema_version", evidence.schema_version},
        {"id", evidence.id},
        {"source", common_adaptation_evidence_source_name(evidence.source)},
        {"scope", {
            {"namespace_id", evidence.scope.namespace_id},
            {"session_id", evidence.scope.session_id},
            {"project_id", evidence.scope.project_id},
            {"turn_id", evidence.scope.turn_id},
        }},
        {"teaching_key", evidence.teaching_key},
        {"behavior_key", evidence.behavior_key},
        {"task_fingerprint", evidence.task_fingerprint},
        {"baseline_ref", evidence.baseline_ref},
        {"candidate_ref", evidence.candidate_ref},
        {"verifier_ref", evidence.verifier_ref},
        {"transaction_ids", evidence.transaction_ids},
        {"cause", common_learning_cause_name(evidence.cause)},
        {"host_verified", evidence.host_verified},
    }.dump();
}

bool common_adaptation_evidence_from_json(
        const std::string & text,
        common_adaptation_evidence & evidence,
        std::string & error) {
    error.clear();
    try {
        const auto value = json::parse(text);
        if (!value.is_object()) {
            error = "adaptation evidence is not an object";
            return false;
        }
        evidence = {};
        evidence.schema_version = value.value("schema_version", 0);
        evidence.id = value.value("id", "");
        const auto source = parse_source(value.value("source", ""));
        if (!source) {
            error = "adaptation evidence source is unknown";
            return false;
        }
        evidence.source = *source;
        const auto scope = value.value("scope", json::object());
        if (!scope.is_object()) {
            error = "adaptation evidence scope is invalid";
            return false;
        }
        evidence.scope.namespace_id = scope.value("namespace_id", "");
        evidence.scope.session_id = scope.value("session_id", "");
        evidence.scope.project_id = scope.value("project_id", "");
        evidence.scope.turn_id = scope.value("turn_id", "");
        evidence.teaching_key = value.value("teaching_key", "");
        evidence.behavior_key = value.value("behavior_key", "");
        evidence.task_fingerprint = value.value("task_fingerprint", "");
        evidence.baseline_ref = value.value("baseline_ref", "");
        evidence.candidate_ref = value.value("candidate_ref", "");
        evidence.verifier_ref = value.value("verifier_ref", "");
        evidence.transaction_ids = value.value("transaction_ids", std::vector<std::string>{});
        evidence.cause = parse_cause(value.value("cause", "unknown"));
        evidence.host_verified = value.value("host_verified", false);
    } catch (const std::exception & exception) {
        error = std::string("invalid adaptation evidence JSON: ") + exception.what();
        return false;
    }
    return common_adaptation_evidence_validate(evidence, 64, error);
}
