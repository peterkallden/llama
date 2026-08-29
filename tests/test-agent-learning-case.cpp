#include "agent/adaptation/learning-case.h"

#include <cassert>

int main() {
    common_learning_case value;
    value.id = "learning://case/1";
    value.observation_id = "learning://observation/1";
    value.scope.namespace_id = "local";
    value.scope.session_id = "session-1";
    value.scope.turn_id = "turn-1";
    value.evidence_ids = {"evidence-1"};
    value.schema_fingerprint = "tool-schema:data.inspect:v1";
    value.input = "Inspect the attached table.";
    value.rejected_action = "invalid binding";
    value.preferred_action = "valid binding";
    value.redaction_policy_id = "stub:caller-asserted-v1";
    value.redaction_method = "caller_asserted";
    value.redaction_status = common_learning_redaction_status::caller_asserted;
    value.content_hash = "identity:fnv1a64:0123456789abcdef";
    value.learning_domain = "tool_use";
    value.tool_family = "diagnostics";
    value.provider_kind = "mcp";
    std::string error;
    assert(!common_learning_case_validate(value, 4, 128, error));
    value.status = common_learning_case_status::redacted;
    assert(common_learning_case_validate(value, 4, 128, error));
    assert(common_learning_case_to_json(value).find("rejected_action") != std::string::npos);
    value.status = common_learning_case_status::revoked;
    assert(!common_learning_case_validate(value, 4, 128, error));
    return 0;
}
