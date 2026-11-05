#pragma once

#include <nlohmann/json.hpp>
#include <string>
#include <vector>

// Response data shared by runtime producers and transport-only clients.
struct common_agent_review_item {
    std::string id, description, correction;
    std::vector<std::string> evidence_ids;
};
struct common_agent_unverified_draft {
    std::string text, stop_reason, plan_id, draft_fingerprint, evidence_fingerprint;
    std::vector<common_agent_review_item> issues;
};

nlohmann::ordered_json common_agent_unverified_draft_json(const common_agent_unverified_draft &);
