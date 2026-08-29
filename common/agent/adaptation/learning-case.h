#pragma once

#include "agent/agent-scope.h"

#include <cstddef>
#include <string>
#include <vector>

enum class common_learning_case_status { draft, redacted, approved, revoked };
enum class common_learning_redaction_status { not_evaluated, caller_asserted, policy_checked, rejected };

const char * common_learning_case_status_name(common_learning_case_status status);
const char * common_learning_redaction_status_name(common_learning_redaction_status status);

// Curator-facing bounded evidence. Runtime code must not populate this from
// raw tool output; callers provide explicitly redacted text.
struct common_learning_case {
    int schema_version = 1;
    std::string id;
    std::string observation_id;
    common_agent_scope scope;
    std::vector<std::string> evidence_ids;
    std::string schema_fingerprint;
    std::string input;
    std::string rejected_action;
    std::string preferred_action;
    std::string redaction_policy_id;
    std::string redaction_method;
    common_learning_redaction_status redaction_status = common_learning_redaction_status::not_evaluated;
    common_learning_case_status status = common_learning_case_status::draft;
    std::string content_hash;
    // Copied explicitly by the curator from host-owned observation metadata;
    // these are not inferred from the tool name during promotion.
    std::string learning_domain;
    std::string tool_family;
    std::string provider_kind;
};

bool common_learning_case_validate(const common_learning_case & value, size_t max_evidence,
        size_t max_text_size, std::string & error);
std::string common_learning_case_to_json(const common_learning_case & value);
