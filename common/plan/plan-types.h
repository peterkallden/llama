#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

enum class common_plan_scope { turn, session, project, global };
enum class common_plan_status { proposed, active, completed, blocked, failed, cancelled };
enum class common_plan_step_status { pending, active, completed, blocked, skipped, failed };

struct common_plan_constraint { std::string id; std::string description; bool hard = true; };
struct common_plan_assumption { std::string id; std::string statement; float confidence = 0.5f; bool valid = true; std::vector<std::string> evidence_ids; };
struct common_plan_observation { std::string id; std::string source; std::string summary; float confidence = 0.5f; std::vector<std::string> evidence_ids; int64_t created_at = 0; };
struct common_plan_step {
    std::string id, title, objective;
    common_plan_step_status status = common_plan_step_status::pending;
    std::vector<std::string> depends_on, blocked_by, required_evidence;
    std::optional<std::string> selected_tool, result_summary;
    bool optional = false, generated_from_memory = false;
    int64_t created_at = 0, updated_at = 0;
};
struct common_plan_state {
    std::string id, session_id;
    common_plan_scope scope = common_plan_scope::turn;
    common_plan_status status = common_plan_status::proposed;
    std::string goal, success_criteria;
    std::vector<common_plan_step> steps;
    std::vector<common_plan_constraint> constraints;
    std::vector<common_plan_assumption> assumptions;
    std::vector<common_plan_observation> observations;
    std::optional<std::string> active_step_id, next_action;
    uint64_t version = 0;
    int64_t created_at = 0, updated_at = 0;
};
enum class common_plan_operation_kind { create_plan, revise_goal, add_step, revise_step, remove_step, activate_step, complete_step, block_step, unblock_step, fail_step, skip_step, add_dependency, remove_dependency, add_constraint, add_assumption, invalidate_assumption, record_observation, set_next_action, request_replan, complete_plan, fail_plan };
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
