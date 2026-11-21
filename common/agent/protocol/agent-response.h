#pragma once

#include <nlohmann/json.hpp>
#include <string>
#include <vector>

// Response data shared by runtime producers and transport-only clients.
struct common_agent_review_item {
    std::string id, description, correction;
    std::vector<std::string> evidence_ids;
};
// Model review of an exact excerpt, bound by the host to a successful read.
// A reviewed excerpt does not approve the rest of its resource or its coverage.
struct common_agent_review_part {
    std::string observation_id, excerpt, status, reason;
    std::vector<std::string> evidence_ids;
    std::string resource_uri, content_fingerprint, evidence_fingerprint;
};
struct common_agent_review_progress {
    std::string plan_id, draft_fingerprint, evidence_fingerprint;
    bool accepted = false, paused = false;
    std::vector<common_agent_review_part> parts;
    std::vector<common_agent_review_item> remaining;
    std::vector<std::string> follow_up;
    // A validated blocked task outcome; neither acceptance nor a budget pause.
    std::string blocked_reason;
};
struct common_agent_unverified_draft {
    std::string text, stop_reason, plan_id, draft_fingerprint, evidence_fingerprint;
    std::vector<common_agent_review_item> issues;
};

nlohmann::ordered_json common_agent_review_parts_json(const std::vector<common_agent_review_part> &);
bool common_agent_review_parts_from_json(const nlohmann::ordered_json &, std::vector<common_agent_review_part> &);
nlohmann::ordered_json common_agent_review_progress_json(const common_agent_review_progress &);
nlohmann::ordered_json common_agent_unverified_draft_json(const common_agent_unverified_draft &);
