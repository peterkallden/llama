#pragma once

#include "agent/runtime/agent-reflection-engine.h"
#include <nlohmann/json.hpp>
#include <map>
#include <functional>

// Method progress is retained in ordinary plan observations, not a second store.
// The reviewer interprets completion; the host validates phase order and provenance.
bool common_agent_method_review(const common_plan_state &,
    const std::vector<common_agent_method_phase_review> &, nlohmann::ordered_json & progress,
    std::string & error);
std::string common_agent_render_method_progress(const common_plan_state &);
bool common_agent_method_has_remaining(const common_plan_state &);
nlohmann::ordered_json common_agent_method_ready_phases(const common_plan_state &);
bool common_agent_method_resolve_choice(const common_agent_request &, const std::string & handle,
    std::string & phase, std::string & error);

bool common_agent_method_parse_selection(const common_agent_request &, bool compact_dsl,
    std::string & text, std::string & phase, std::string & error);

bool common_agent_method_validate_progress(const common_plan_state &,
    const std::vector<common_plan_operation> &, const std::string & phase, const std::string & turn,
    const std::function<bool(const std::string &)> & is_read_only, std::string & error);

// Build ordinary additive plan operations from the existing planner's bounded
// phase slice. Fresh host IDs prevent collisions with prior slices on resume.
bool common_agent_method_append_slice(const common_plan_state &, const common_plan_state & slice,
    std::vector<common_plan_operation> &, std::string & error,
    const std::map<std::string, std::string> & repair_targets = {});
