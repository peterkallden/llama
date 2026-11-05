#include "agent/protocol/agent-jsonl.h"

int main() {
    common_agent_jsonl_turn_result result;
    result.ok = false;
    common_agent_unverified_draft draft;
    draft.text = "Incomplete report";
    draft.stop_reason = "review_budget_exhausted";
    draft.issues.push_back({"review-1-1", "Missing citation", "Add a citation", {"source:1"}});
    result.unverified_draft = draft;
    const auto message = common_agent_jsonl_make_turn_result(result);
    if (message.at("ok") != false || !message.at("response").get<std::string>().empty()) return 1;
    const auto & serialized = message.at("unverified_draft");
    if (serialized.at("review_status") != "unverified" || serialized.at("text") != draft.text) return 2;
    if (serialized.at("issues").at(0).at("evidence_ids").at(0) != "source:1") return 3;
    result.unverified_draft.reset();
    if (common_agent_jsonl_make_turn_result(result).contains("unverified_draft")) return 4;
    return 0;
}
