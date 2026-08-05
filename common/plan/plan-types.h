#pragma once

#include "runtime/runtime-state.h"

#include <cstdint>
#include <algorithm>
#include <optional>
#include "resource/resource-contract.h"
#include <string>
#include <vector>

enum class common_plan_scope { turn, session, project, global };
enum class common_plan_kind { task, blueprint };
enum class common_plan_status { proposed, active, completed, blocked, failed, cancelled };
enum class common_plan_step_status { pending, active, completed, blocked, skipped, failed };
// A plan step either invokes one registered tool, asks the model for a bounded
// intermediate result, or produces the user-visible final response.
enum class common_plan_step_mode { tool, reasoning, final_response };

struct common_plan_constraint { std::string id; std::string description; bool hard = true; };
struct common_plan_assumption { std::string id; std::string statement; float confidence = 0.5f; bool valid = true; std::vector<std::string> evidence_ids; };
struct common_plan_observation {
    std::string id;
    std::string source;
    std::string summary;
    float confidence = 0.5f;
    std::vector<std::string> evidence_ids;
    std::vector<common_runtime_resource_ref> resource_refs;
    int64_t created_at = 0;
};

inline bool common_plan_chunk_observations_valid(
        const std::vector<common_plan_observation> & observations,
        std::string & error) {
    std::string parent_uri;
    size_t chunk_count = 0;
    std::vector<size_t> indexes;
    for (const auto & observation : observations) {
        if (observation.source != "resource_chunk") {
            continue;
        }
        if (observation.resource_refs.size() != 1 ||
                observation.resource_refs.front().lineage.parent_uri.empty()) {
            error = "resource_chunk observation requires exactly one parent-linked resource";
            return false;
        }
        const auto & lineage = observation.resource_refs.front().lineage;
        if (!common_runtime_resource_lineage_is_valid(lineage, error)) {
            return false;
        }
        if (parent_uri.empty()) {
            parent_uri = lineage.parent_uri;
            chunk_count = lineage.chunk_count;
        } else if (parent_uri != lineage.parent_uri || chunk_count != lineage.chunk_count) {
            error = "resource_chunk observations must share one parent and chunk_count";
            return false;
        }
        if (std::find(indexes.begin(), indexes.end(), lineage.chunk_index) != indexes.end()) {
            error = "resource_chunk observations must not contain duplicate chunk indexes";
            return false;
        }
        indexes.push_back(lineage.chunk_index);
    }
    error.clear();
    return true;
}

enum class common_plan_chunk_synthesis_status { complete, incomplete, conflict };

struct common_plan_chunk_synthesis_input {
    common_plan_chunk_synthesis_status status = common_plan_chunk_synthesis_status::incomplete;
    std::string parent_uri;
    size_t chunk_count = 0;
    std::vector<size_t> completed_chunk_indexes;
    std::vector<size_t> missing_chunk_indexes;
    std::vector<std::string> summaries;
};

// Build a deterministic, ordered synthesis view from existing plan evidence.
// The original resource remains authoritative; summaries are bounded working
// evidence and must not be treated as a replacement resource.
inline bool common_plan_chunk_synthesis_from_observations(
        const std::vector<common_plan_observation> & observations,
        const std::string & parent_uri,
        common_plan_chunk_synthesis_input & out,
        std::string & error) {
    out = {};
    out.parent_uri = parent_uri;
    if (!common_plan_chunk_observations_valid(observations, error)) {
        out.status = common_plan_chunk_synthesis_status::conflict;
        return false;
    }
    struct item { size_t index; std::string summary; };
    std::vector<item> items;
    for (const auto & observation : observations) {
        if (observation.source != "resource_chunk" || observation.resource_refs.size() != 1) continue;
        const auto & lineage = observation.resource_refs.front().lineage;
        if (!parent_uri.empty() && lineage.parent_uri != parent_uri) continue;
        if (out.parent_uri.empty()) out.parent_uri = lineage.parent_uri;
        if (out.chunk_count == 0) out.chunk_count = lineage.chunk_count;
        items.push_back({lineage.chunk_index, observation.summary});
    }
    std::sort(items.begin(), items.end(), [](const item & left, const item & right) {
        return left.index < right.index;
    });
    for (const auto & item : items) {
        out.completed_chunk_indexes.push_back(item.index);
        out.summaries.push_back(item.summary);
    }
    for (size_t index = 0; index < out.chunk_count; ++index) {
        if (std::find(out.completed_chunk_indexes.begin(), out.completed_chunk_indexes.end(), index) ==
                out.completed_chunk_indexes.end()) {
            out.missing_chunk_indexes.push_back(index);
        }
    }
    out.status = out.missing_chunk_indexes.empty()
        ? common_plan_chunk_synthesis_status::complete
        : common_plan_chunk_synthesis_status::incomplete;
    error.clear();
    return true;
}
// Data proposed by a plan; execution is owned by the agent tool registry.
struct common_plan_tool_call { std::string name; std::string arguments_json = "{}"; };
struct common_plan_step {
    std::string id, title, objective, intended_contribution;
    common_plan_step_status status = common_plan_step_status::pending;
    std::vector<std::string> depends_on, blocked_by, required_evidence, source_memory_ids;
    std::optional<std::string> selected_tool, result_summary;
    std::optional<common_plan_tool_call> tool_call;
    common_plan_step_mode mode = common_plan_step_mode::final_response;
    bool optional = false, generated_from_memory = false;
    int64_t created_at = 0, updated_at = 0;
};

inline common_plan_step_mode common_plan_step_effective_mode(const common_plan_step & step) {
    return step.tool_call ? common_plan_step_mode::tool : step.mode;
}
struct common_plan_state {
    std::string id, session_id;
    // Runtime-owned identity boundary.  These mirror memory identity so a
    // persisted plan can be selected or resumed only in its original scope.
    std::string namespace_id = "local", project_id, turn_id;
    common_plan_kind kind = common_plan_kind::task;
    std::optional<std::string> derived_from_plan_id;
    common_plan_scope scope = common_plan_scope::turn;
    common_plan_status status = common_plan_status::proposed;
    // Purpose is the stable user-facing reason for the work. Goal and success
    // criteria describe the mutable execution target for this particular plan.
    std::string purpose, goal, success_criteria;
    std::vector<common_plan_step> steps;
    // Semantic host capabilities required to execute this plan. These are
    // requirements, not tool bindings; the host resolves them per turn.
    std::vector<std::string> required_capabilities;
    std::vector<common_plan_constraint> constraints;
    std::vector<common_plan_assumption> assumptions;
    std::vector<common_plan_observation> observations;
    std::optional<std::string> active_step_id, next_action;
    uint64_t version = 0;
    int64_t created_at = 0, updated_at = 0;
};

inline common_agent_state_descriptor describe_common_plan(
        const common_plan_state & plan) {
    common_agent_state_descriptor descriptor;
    descriptor.state_id = plan.id;
    descriptor.state_type = "plan";
    descriptor.state_class = common_agent_state_class::durable_domain;
    descriptor.lifetime = plan.scope == common_plan_scope::turn
        ? common_agent_state_lifetime::turn
        : plan.scope == common_plan_scope::session
            ? common_agent_state_lifetime::session
            : plan.scope == common_plan_scope::project
                ? common_agent_state_lifetime::project
                : common_agent_state_lifetime::durable;
    descriptor.persistence = common_agent_state_persistence::persistent;
    descriptor.identity.namespace_id = plan.namespace_id;
    descriptor.identity.project_id = plan.project_id;
    descriptor.identity.session_id = plan.session_id;
    descriptor.identity.turn_id = plan.turn_id;
    descriptor.owner = "common_plan_store";
    descriptor.source_of_truth = "plan store";
    return descriptor;
}
enum class common_plan_operation_kind { create_plan, revise_goal, add_step, revise_step, replace_step, remove_step, activate_step, reset_step, complete_step, block_step, unblock_step, fail_step, skip_step, add_dependency, remove_dependency, add_constraint, add_assumption, invalidate_assumption, record_observation, set_next_action, request_replan, complete_plan, fail_plan };
struct common_plan_operation {
    common_plan_operation_kind kind = common_plan_operation_kind::add_step;
    std::string plan_id;
    uint64_t expected_version = 0;
    std::optional<std::string> step_id, target_id, value;
    std::optional<common_plan_step> step;
    std::optional<common_plan_constraint> constraint;
    std::optional<common_plan_assumption> assumption;
    std::optional<common_plan_observation> observation;
    std::string reason_summary;
    std::vector<std::string> evidence_ids;
};
struct common_plan_event { uint64_t sequence = 0, prior_version = 0, new_version = 0; common_plan_operation operation; bool accepted = false; std::string reason_summary; int64_t created_at = 0; };

const char * common_plan_operation_kind_name(common_plan_operation_kind kind);
bool common_plan_scope_matches(const common_plan_state & plan, common_plan_scope scope,
    const std::string & namespace_id, const std::string & session_id,
    const std::string & project_id, const std::string & turn_id);
