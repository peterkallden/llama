#include "agent-response.h"
#include <utility>

using json = nlohmann::ordered_json;

json common_agent_unverified_draft_json(const common_agent_unverified_draft & draft) {
    auto issues = json::array();
    for (const auto & item : draft.issues) {
        issues.push_back({{"id", item.id}, {"description", item.description},
            {"correction", item.correction}, {"evidence_ids", item.evidence_ids}});
    }
    return {{"text",draft.text},{"review_status","unverified"},{"stop_reason",draft.stop_reason},
        {"plan_id",draft.plan_id},{"draft_fingerprint",draft.draft_fingerprint},
        {"evidence_fingerprint",draft.evidence_fingerprint},{"issues",std::move(issues)}};
}
