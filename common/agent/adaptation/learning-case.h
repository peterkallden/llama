#pragma once

#include "agent/agent-scope.h"

#include <cstddef>
#include <string>
#include <vector>

enum class common_learning_case_status { draft, redacted, approved, revoked };

const char * common_learning_case_status_name(common_learning_case_status status);

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
    common_learning_case_status status = common_learning_case_status::draft;
    std::string content_hash;
};

bool common_learning_case_validate(const common_learning_case & value, size_t max_evidence,
        size_t max_text_size, std::string & error);
std::string common_learning_case_to_json(const common_learning_case & value);
