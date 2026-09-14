#include "agent/adaptation/flydelta/flydelta-contracts.h"

#include <algorithm>
#include <cmath>
#include <optional>

#include <nlohmann/json.hpp>

using json = nlohmann::ordered_json;

namespace {

bool nonempty_bounded(const std::string & value, size_t max_size = 512) {
    return !value.empty() && value.size() <= max_size;
}

bool verified_model_observation(const common_learning_observation & observation) {
    return observation.collection_allowed &&
        observation.cause == common_learning_cause::model_behavior &&
        (observation.verification == common_learning_verification::host_verified ||
         observation.verification == common_learning_verification::user_confirmed) &&
        common_learning_observation_qualifies(observation);
}

} // namespace

const char * common_flydelta_candidate_status_name(common_flydelta_candidate_status status) {
    switch (status) {
        case common_flydelta_candidate_status::observed: return "observed";
        case common_flydelta_candidate_status::eligible: return "eligible";
        case common_flydelta_candidate_status::approved: return "approved";
        case common_flydelta_candidate_status::rejected: return "rejected";
        case common_flydelta_candidate_status::revoked: return "revoked";
    }
    return "rejected";
}

bool common_flydelta_static_overlay_validate(
        const common_flydelta_static_overlay & overlay,
        size_t model_n_embd,
        size_t model_n_layers,
        size_t max_bytes,
        std::string & error) {
    error.clear();
    if (!overlay.enabled) return true;
    if (overlay.artifact_id.empty() || overlay.n_embd <= 0 || overlay.il_start < 1 ||
            overlay.il_end < overlay.il_start || overlay.scale <= 0.0f ||
            !std::isfinite(overlay.scale) || overlay.scale > 1.0f) {
        error = "FlyDelta static overlay identity or bounds are invalid";
        return false;
    }
    if (model_n_embd == 0 || model_n_layers < 2 ||
            static_cast<size_t>(overlay.n_embd) != model_n_embd ||
            static_cast<size_t>(overlay.il_end) >= model_n_layers) {
        error = "FlyDelta static overlay does not match model dimensions";
        return false;
    }
    const size_t required_values = model_n_embd * (model_n_layers - 1);
    if (overlay.data.size() < required_values ||
            (max_bytes != 0 && overlay.data.size() * sizeof(float) > max_bytes)) {
        error = "FlyDelta static overlay data exceeds its bound";
        return false;
    }
    for (const float value : overlay.data) {
        if (!std::isfinite(value)) {
            error = "FlyDelta static overlay contains a non-finite value";
            return false;
        }
    }
    return true;
}

bool common_flydelta_capture_manifest_validate(
        const common_flydelta_capture_manifest & manifest,
        size_t max_captured_bytes,
        std::string & error) {
    error.clear();
    if (manifest.schema_version != 2 || !nonempty_bounded(manifest.id) ||
            !nonempty_bounded(manifest.observation_id) ||
            !nonempty_bounded(manifest.behavior_key) ||
            !nonempty_bounded(manifest.model_profile_fingerprint) ||
            !nonempty_bounded(manifest.template_fingerprint) ||
            !nonempty_bounded(manifest.positive_execution_ref) ||
            !nonempty_bounded(manifest.negative_execution_ref) ||
            !nonempty_bounded(manifest.capture_layout_revision) ||
            !nonempty_bounded(manifest.evidence_hash)) {
        error = "FlyDelta capture manifest identity is incomplete";
        return false;
    }
    if (!manifest.redaction_attested) {
        error = "FlyDelta capture manifest requires redaction attestation";
        return false;
    }
    if (manifest.captured_bytes > max_captured_bytes) {
        error = "FlyDelta capture exceeds byte bound";
        return false;
    }
    return true;
}

std::string common_flydelta_capture_manifest_to_json(
        const common_flydelta_capture_manifest & manifest) {
    return json{
        {"schema_version", manifest.schema_version},
        {"id", manifest.id},
        {"observation_id", manifest.observation_id},
        {"source", common_adaptation_evidence_source_name(manifest.source)},
        {"behavior_key", manifest.behavior_key},
        {"model_profile_fingerprint", manifest.model_profile_fingerprint},
        {"template_fingerprint", manifest.template_fingerprint},
        {"positive_execution_ref", manifest.positive_execution_ref},
        {"negative_execution_ref", manifest.negative_execution_ref},
        {"capture_layout_revision", manifest.capture_layout_revision},
        {"evidence_hash", manifest.evidence_hash},
        {"redaction_attested", manifest.redaction_attested},
        {"captured_bytes", manifest.captured_bytes},
    }.dump();
}

bool common_flydelta_capture_manifest_from_json(
        const std::string & text,
        size_t max_captured_bytes,
        common_flydelta_capture_manifest & manifest,
        std::string & error) {
    error.clear();
    try {
        const auto value = json::parse(text);
        manifest = {};
        manifest.schema_version = value.value("schema_version", 0);
        manifest.id = value.value("id", "");
        manifest.observation_id = value.value("observation_id", "");
        const auto source = common_adaptation_evidence_source_from_name(
            value.value("source", ""));
        if (!source) {
            error = "FlyDelta capture manifest source is unknown";
            return false;
        }
        manifest.source = *source;
        manifest.behavior_key = value.value("behavior_key", "");
        manifest.model_profile_fingerprint = value.value("model_profile_fingerprint", "");
        manifest.template_fingerprint = value.value("template_fingerprint", "");
        manifest.positive_execution_ref = value.value("positive_execution_ref", "");
        manifest.negative_execution_ref = value.value("negative_execution_ref", "");
        manifest.capture_layout_revision = value.value("capture_layout_revision", "");
        manifest.evidence_hash = value.value("evidence_hash", "");
        manifest.redaction_attested = value.value("redaction_attested", false);
        manifest.captured_bytes = value.value("captured_bytes", 0U);
    } catch (const std::exception & exception) {
        error = std::string("invalid FlyDelta capture manifest JSON: ") + exception.what();
        return false;
    }
    return common_flydelta_capture_manifest_validate(manifest, max_captured_bytes, error);
}

bool common_flydelta_candidate_validate(
        const common_flydelta_candidate & candidate,
        const common_flydelta_candidate_policy & policy,
        std::string & error) {
    error.clear();
    if (candidate.schema_version != 1 || candidate.id.empty() || candidate.scope.namespace_id.empty()) {
        error = "FlyDelta candidate identity is incomplete";
        return false;
    }
    if (candidate.transaction_ids.empty() || candidate.transaction_ids.size() > policy.max_observations ||
            candidate.capture_manifest_ids.size() > policy.max_capture_manifests) {
        error = "FlyDelta candidate reference bound is invalid";
        return false;
    }
    if (candidate.cause != common_learning_cause::model_behavior ||
            (candidate.verification != common_learning_verification::host_verified &&
             candidate.verification != common_learning_verification::user_confirmed)) {
        error = "FlyDelta candidate is not host-verified model behavior";
        return false;
    }
    if (candidate.observed_occurrences < policy.min_observations ||
            candidate.verified_observations < policy.min_verified_observations ||
            candidate.verified_observations > candidate.observed_occurrences || candidate.contradictions != 0 ||
            !std::isfinite(candidate.confidence) || candidate.confidence < policy.min_confidence || candidate.confidence > 1.0f) {
        error = "FlyDelta candidate has insufficient or inconsistent qualification evidence";
        return false;
    }
    if (candidate.status != common_flydelta_candidate_status::eligible &&
            candidate.status != common_flydelta_candidate_status::approved) {
        error = "FlyDelta candidate requires eligibility or explicit approval";
        return false;
    }
    return true;
}

bool common_flydelta_candidate_from_transactions(
        const std::vector<common_learning_transaction> & transactions,
        const common_flydelta_candidate_policy & policy,
        common_flydelta_candidate & candidate,
        std::string & error) {
    error.clear();
    if (transactions.empty() || transactions.size() > policy.max_observations) {
        error = "FlyDelta candidate transaction bound is invalid";
        return false;
    }
    const auto & first = transactions.front().observation;
    if (!verified_model_observation(first)) {
        error = "FlyDelta source transaction is not host-verified model behavior";
        return false;
    }
    candidate = {};
    candidate.id = "learning://flydelta/candidate/" + transactions.front().id;
    candidate.scope = first.scope;
    std::optional<common_adaptation_evidence_source> candidate_source;
    candidate.cause = first.cause;
    candidate.verification = first.verification;
    candidate.status = common_flydelta_candidate_status::eligible;
    for (const auto & transaction : transactions) {
        std::string transaction_error;
        if (!common_learning_transaction_validate(transaction, 64, transaction_error)) {
            error = "FlyDelta source transaction is not eligible: " + transaction_error;
            return false;
        }
        const auto & observation = transaction.observation;
        if (!verified_model_observation(observation)) {
            error = "FlyDelta source transaction is not eligible: observation is not host-verified model behavior";
            return false;
        }
        if (observation.scope.namespace_id != candidate.scope.namespace_id ||
                observation.scope.project_id != candidate.scope.project_id ||
                observation.cause != candidate.cause) {
            error = "FlyDelta candidate mixes incompatible scopes or causes";
            return false;
        }
        if (candidate.verification != common_learning_verification::user_confirmed &&
                observation.verification == common_learning_verification::user_confirmed) {
            candidate.verification = observation.verification;
        }
        candidate.transaction_ids.push_back(transaction.id);
        for (const auto & signal : observation.signals) {
            const auto source = common_adaptation_evidence_source_for_signal(signal.type);
            if (!source) continue;
            if (candidate_source && *candidate_source != *source) {
                error = "FlyDelta candidate mixes incompatible adaptation sources";
                return false;
            }
            candidate_source = source;
        }
        ++candidate.observed_occurrences;
        ++candidate.verified_observations;
        for (const auto & signal : observation.signals) {
            if (candidate.tool_family.empty()) candidate.tool_family = signal.tool_family;
            if (candidate.provider_kind.empty()) candidate.provider_kind = signal.provider_kind;
            if (candidate.learning_domain.empty()) candidate.learning_domain = signal.tool_family.empty() ? "agent_behavior" : signal.tool_family;
        }
    }
    candidate.confidence = std::min(1.0f, static_cast<float>(candidate.verified_observations) /
        static_cast<float>(std::max<size_t>(1, candidate.observed_occurrences)));
    if (candidate_source) candidate.source = *candidate_source;
    if (!common_flydelta_candidate_validate(candidate, policy, error)) return false;
    return true;
}

bool common_flydelta_evaluation_report_validate(
        const common_flydelta_evaluation_report & report,
        std::string & error) {
    error.clear();
    if (report.schema_version != 1 || !nonempty_bounded(report.revision_id) ||
            !nonempty_bounded(report.candidate_id) || !nonempty_bounded(report.baseline_profile_id) ||
            !nonempty_bounded(report.candidate_profile_id) || !nonempty_bounded(report.test_suite_revision) ||
            report.baseline_profile_id == report.candidate_profile_id) {
        error = "FlyDelta evaluation identity is incomplete";
        return false;
    }
    if (report.evaluated_turns == 0 || report.baseline_successes > report.evaluated_turns ||
            report.candidate_successes > report.evaluated_turns || report.candidate_interventions > report.evaluated_turns ||
            report.false_interventions > report.candidate_interventions) {
        error = "FlyDelta evaluation counts are invalid";
        return false;
    }
    const bool passed = report.intended_behavior_passed && report.retention_passed && report.agent_regression_passed;
    if ((report.status == "passed") != passed || (report.status != "passed" && report.status != "failed")) {
        error = "FlyDelta evaluation status does not match gates";
        return false;
    }
    return true;
}

std::string common_flydelta_evaluation_report_to_json(
        const common_flydelta_evaluation_report & report) {
    return json{
        {"schema_version", report.schema_version},
        {"revision_id", report.revision_id},
        {"candidate_id", report.candidate_id},
        {"baseline_profile_id", report.baseline_profile_id},
        {"candidate_profile_id", report.candidate_profile_id},
        {"test_suite_revision", report.test_suite_revision},
        {"gates", {
            {"intended_behavior", report.intended_behavior_passed},
            {"retention", report.retention_passed},
            {"agent_regression", report.agent_regression_passed},
        }},
        {"evaluated_turns", report.evaluated_turns},
        {"baseline_successes", report.baseline_successes},
        {"candidate_successes", report.candidate_successes},
        {"candidate_interventions", report.candidate_interventions},
        {"false_interventions", report.false_interventions},
        {"status", report.status},
    }.dump();
}

bool common_flydelta_evaluation_report_from_json(
        const std::string & text,
        common_flydelta_evaluation_report & report,
        std::string & error) {
    error.clear();
    try {
        const auto value = json::parse(text);
        report = {};
        report.schema_version = value.value("schema_version", 0);
        report.revision_id = value.value("revision_id", "");
        report.candidate_id = value.value("candidate_id", "");
        report.baseline_profile_id = value.value("baseline_profile_id", "");
        report.candidate_profile_id = value.value("candidate_profile_id", "");
        report.test_suite_revision = value.value("test_suite_revision", "");
        const auto gates = value.value("gates", json::object());
        report.intended_behavior_passed = gates.value("intended_behavior", false);
        report.retention_passed = gates.value("retention", false);
        report.agent_regression_passed = gates.value("agent_regression", false);
        report.evaluated_turns = value.value("evaluated_turns", 0U);
        report.baseline_successes = value.value("baseline_successes", 0U);
        report.candidate_successes = value.value("candidate_successes", 0U);
        report.candidate_interventions = value.value("candidate_interventions", 0U);
        report.false_interventions = value.value("false_interventions", 0U);
        report.status = value.value("status", "failed");
    } catch (const std::exception & exception) {
        error = std::string("invalid FlyDelta evaluation JSON: ") + exception.what();
        return false;
    }
    return common_flydelta_evaluation_report_validate(report, error);
}
