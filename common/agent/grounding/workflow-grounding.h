#pragma once
#include "agent/contracts/agent-request.h"
#include "agent/thinking/research/research-verifier.h"

struct common_agent_grounding_result {
    bool passed = true;
    std::string summary;
    std::string clarification_question;
    nlohmann::ordered_json records = nlohmann::ordered_json::array();
};
bool common_agent_workflow_requires_claims(const common_plan_state &);
std::string common_agent_grounded_draft_schema();
std::string common_agent_grounded_draft_grammar(const nlohmann::ordered_json & handles);
common_agent_grounding_result common_agent_verify_workflow(
    const common_agent_request &, const common_plan_state &, common_plan_verification_phase,
    const std::string & step_id, const std::string & arguments, const std::string & result,
    const std::string & canonical_tool_name = {});

// Host-built bounded handles are the only model-selectable claim sources.
nlohmann::ordered_json common_agent_workflow_claim_handles(
    const common_agent_request &, const common_plan_state &);
bool common_agent_parse_grounded_draft(const std::string &, bool json_format,
    std::string & answer, nlohmann::ordered_json & claims, std::string & error);
common_agent_grounding_result common_agent_verify_declared_claims(
    const nlohmann::ordered_json & claims, const nlohmann::ordered_json & handles,
    std::vector<common_agent_research_verification_claim> & verified_claims);
