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

// Resolve immutable checkpoint lineage without granting execution authority.
class common_plan_store;
std::optional<common_plan_state> common_agent_resolve_prior_work(
    common_plan_store &, const common_plan_state & successor,
    const nlohmann::ordered_json & checkpoint, std::string & error);

// Produce an inheritance decision without changing either plan. The caller must
// validate the referenced observations against current host-owned sources.
nlohmann::ordered_json common_agent_method_inheritance(
    const common_plan_state & parent, const common_plan_state & successor,
    const std::function<bool(const common_plan_observation &)> & evidence_is_current);

// Resolve one issued reference only within the validated ancestor chain.
std::optional<common_plan_observation> common_agent_resolve_prior_observation(
    common_plan_store &, const common_plan_state & successor,
    const nlohmann::ordered_json & checkpoint,
    const nlohmann::ordered_json & reference, std::string & error);

common_plan_state common_agent_prior_evidence_view(const common_plan_state & current,
    const common_plan_state & parent,
    const std::function<bool(const common_plan_observation &)> & evidence_is_current);
