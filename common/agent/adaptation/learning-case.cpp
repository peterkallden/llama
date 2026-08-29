#include "agent/adaptation/learning-case.h"

#include <nlohmann/json.hpp>

using json = nlohmann::ordered_json;

const char * common_learning_case_status_name(common_learning_case_status status) {
    switch (status) {
        case common_learning_case_status::draft: return "draft";
        case common_learning_case_status::redacted: return "redacted";
        case common_learning_case_status::approved: return "approved";
        case common_learning_case_status::revoked: return "revoked";
    }
    return "draft";
}

const char * common_learning_redaction_status_name(common_learning_redaction_status status) {
    switch (status) {
        case common_learning_redaction_status::not_evaluated: return "not_evaluated";
        case common_learning_redaction_status::caller_asserted: return "caller_asserted";
        case common_learning_redaction_status::policy_checked: return "policy_checked";
        case common_learning_redaction_status::rejected: return "rejected";
    }
    return "not_evaluated";
}

bool common_learning_case_validate(const common_learning_case & value, size_t max_evidence,
        size_t max_text_size, std::string & error) {
    error.clear();
    if (value.schema_version != 1) { error = "unsupported learning case schema"; return false; }
    if (value.id.empty() || value.observation_id.empty()) { error = "learning case requires source identity"; return false; }
    if (value.scope.namespace_id.empty() || value.scope.session_id.empty() || value.scope.turn_id.empty()) {
        error = "learning case requires complete scope"; return false;
    }
    if (value.evidence_ids.empty() || value.evidence_ids.size() > max_evidence) {
        error = "learning case evidence bound is invalid"; return false;
    }
    for (const auto & id : value.evidence_ids) if (id.empty()) { error = "learning case contains empty evidence id"; return false; }
    if (value.schema_fingerprint.empty() || value.input.empty() || value.rejected_action.empty() ||
            value.preferred_action.empty() || value.content_hash.empty()) {
        error = "learning case requires redacted content and schema fingerprint"; return false;
    }
    if (value.input.size() > max_text_size || value.rejected_action.size() > max_text_size ||
            value.preferred_action.size() > max_text_size) {
        error = "learning case exceeds text bound"; return false;
    }
    if (value.redaction_policy_id.empty() || value.redaction_method.empty() ||
            (value.redaction_status != common_learning_redaction_status::caller_asserted &&
             value.redaction_status != common_learning_redaction_status::policy_checked)) {
        error = "learning case lacks an accepted redaction result";
        return false;
    }
    if (value.status == common_learning_case_status::draft || value.status == common_learning_case_status::revoked) {
        error = "learning case is not exportable"; return false;
    }
    return true;
}

std::string common_learning_case_to_json(const common_learning_case & value) {
    return json{
        {"schema_version", value.schema_version}, {"id", value.id},
        {"observation_id", value.observation_id},
        {"scope", {{"namespace_id", value.scope.namespace_id}, {"session_id", value.scope.session_id},
                    {"project_id", value.scope.project_id}, {"turn_id", value.scope.turn_id}}},
        {"evidence_ids", value.evidence_ids}, {"schema_fingerprint", value.schema_fingerprint},
        {"input", value.input}, {"rejected_action", value.rejected_action},
        {"preferred_action", value.preferred_action},
        {"redaction", {{"policy_id", value.redaction_policy_id}, {"method", value.redaction_method},
                        {"status", common_learning_redaction_status_name(value.redaction_status)}}},
        {"status", common_learning_case_status_name(value.status)}, {"content_hash", value.content_hash}
    }.dump();
}
