#pragma once

#include "plan/plan-types.h"
#include "protocol/agent-response.h"
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <vector>

// Host lifecycle facts, independent of inference backend and research storage.
enum class common_agent_synthesis_work { pending, complete, bounded };
enum class common_agent_synthesis_evidence { unknown, sufficient, partial, insufficient };
enum class common_agent_continuation_outcome { transitioned, not_applicable, failed };
struct common_agent_synthesis_state {
    common_agent_synthesis_work work = common_agent_synthesis_work::pending;
    common_agent_synthesis_evidence evidence = common_agent_synthesis_evidence::unknown;
    bool mandatory_remaining = true;
    bool partial_allowed = false;
    std::string stop_reason;
    std::vector<std::string> evidence_ids;
    std::string continuation_reason, plan_id, evidence_fingerprint;
    uint64_t plan_version = 0;
};

struct common_agent_review_checklist {
    size_t round = 0;
    std::string draft_fingerprint, evidence_fingerprint;
    std::vector<common_agent_review_item> blocking;
    std::vector<std::string> advisory;
};

common_agent_synthesis_state common_agent_resolve_synthesis_state(const common_plan_state &,
    std::optional<common_agent_synthesis_state> research = std::nullopt);
std::string common_agent_synthesis_evidence_fingerprint(const common_plan_state &);
std::string common_agent_workflow_evidence_fingerprint(const common_plan_state &);
nlohmann::ordered_json common_agent_synthesis_state_json(const common_agent_synthesis_state &);
std::string common_agent_render_synthesis_state(const common_agent_synthesis_state &, const std::string & phase);
nlohmann::ordered_json common_agent_review_checklist_json(const common_agent_review_checklist &);
bool common_agent_review_checklist_from_json(const nlohmann::ordered_json &, common_agent_review_checklist &);
