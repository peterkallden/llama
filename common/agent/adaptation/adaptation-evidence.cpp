#include "agent/adaptation/adaptation-evidence.h"

#include <nlohmann/json.hpp>

using json = nlohmann::ordered_json;

namespace {

bool bounded_nonempty(const std::string & value, size_t max_size = 512) {
    return !value.empty() && value.size() <= max_size;
}

common_adaptation_evidence_source parse_source(const std::string & value, bool & ok) {
    ok = true;
    const common_adaptation_evidence_source sources[] = {
        common_adaptation_evidence_source::tool_repair,
        common_adaptation_evidence_source::reflection_alternative,
        common_adaptation_evidence_source::planning_revision,
        common_adaptation_evidence_source::research_alternative,
        common_adaptation_evidence_source::dataset_resource,
        common_adaptation_evidence_source::workflow_code,
        common_adaptation_evidence_source::user_correction,
    };
    for (const auto source : sources) {
        if (value == common_adaptation_evidence_source_name(source)) return source;
    }
    ok = false;
    return common_adaptation_evidence_source::tool_repair;
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
        case common_adaptation_evidence_source::user_correction: return "user_correction";
    }
    return "tool_repair";
}

bool common_adaptation_evidence_validate(
        const common_adaptation_evidence & evidence,
        size_t max_transactions,
        std::string & error) {
    error.clear();
    if (evidence.schema_version != 1 || !bounded_nonempty(evidence.id) ||
            evidence.scope.namespace_id.empty() || evidence.scope.session_id.empty() ||
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
        bool source_ok = false;
        evidence.source = parse_source(value.value("source", ""), source_ok);
        if (!source_ok) {
            error = "adaptation evidence source is unknown";
            return false;
        }
        const auto scope = value.value("scope", json::object());
        if (!scope.is_object()) {
            error = "adaptation evidence scope is invalid";
            return false;
        }
        evidence.scope.namespace_id = scope.value("namespace_id", "");
        evidence.scope.session_id = scope.value("session_id", "");
        evidence.scope.project_id = scope.value("project_id", "");
        evidence.scope.turn_id = scope.value("turn_id", "");
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

