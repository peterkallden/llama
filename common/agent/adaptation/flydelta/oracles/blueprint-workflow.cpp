#include "agent/adaptation/flydelta/oracles/blueprint-workflow.h"

#include <nlohmann/json.hpp>

#include <utility>

using json = nlohmann::ordered_json;

namespace {

bool valid_terminal(const std::string & tool) {
    return tool == "data.query" || tool == "data.filter" ||
        tool == "data.aggregate" || tool == "data.join" ||
        tool == "data.transform" || tool == "statistics.describe" ||
        tool == "statistics.outliers" || tool == "statistics.value_counts";
}

bool valid_request(const common_flydelta_dataset_blueprint_request & request) {
    return request.schema_version == 1 && !request.proposal_id.empty() &&
        request.blueprint_ref.rfind("blueprint://dataset-", 0) == 0 &&
        !request.blueprint_revision.empty() && !request.graph_revision.empty() &&
        !request.workflow_ref.empty() && !request.workflow_revision.empty() &&
        !request.dataset_ref.empty() && !request.dataset_name.empty() &&
        (request.terminal_tool != "data.join" ||
            (!request.second_dataset_ref.empty() && !request.second_dataset_name.empty())) &&
        valid_terminal(request.terminal_tool) &&
        !request.terminal_arguments_json.empty() && request.max_expansions > 0 &&
        request.max_path_length >= 2 && request.max_path_length <= 8;
}

std::string workflow_step_id(const std::string & plan_id, size_t index) {
    return plan_id + ":astar:" + std::to_string(index);
}

} // namespace

bool common_flydelta_propose_dataset_blueprint_workflow(
        const common_flydelta_dataset_blueprint_request & request,
        common_flydelta_workflow_proposal & result,
        std::string & error) {
    error.clear();
    result = {};
    if (!valid_request(request)) {
        error = "dataset blueprint request is incomplete or outside the bounded contract";
        result.status = common_flydelta_astar_status::invalid_request;
        return false;
    }

    common_flydelta_workflow_proposal_request proposal_request;
    proposal_request.proposal_id = request.proposal_id;
    proposal_request.blueprint_ref = request.blueprint_ref;
    proposal_request.blueprint_revision = request.blueprint_revision;
    proposal_request.workflow_ref = request.workflow_ref;
    proposal_request.workflow_revision = request.workflow_revision;
    proposal_request.graph_revision = request.graph_revision;
    proposal_request.start_state = "source-unselected";
    proposal_request.goal_state = "operation-resolved";
    proposal_request.max_expansions = request.max_expansions;
    proposal_request.max_path_length = request.max_path_length;
    proposal_request.is_goal = [](const std::string & state) {
        return state == "operation-resolved";
    };
    proposal_request.expand = [request](
            const std::string & state,
            std::vector<common_flydelta_astar_successor> & successors,
            std::string & expand_error) {
        successors.clear();
        expand_error.clear();
        if (state == "source-unselected") {
            successors.push_back({"source-selected", "select-dataset", 1.0f, 0.0f});
        } else if (state == "source-selected") {
            if (request.terminal_tool == "data.join") {
                successors.push_back({"second-source-selected", "select-second-dataset", 1.0f, 0.0f});
            } else if (request.schema_known) {
                successors.push_back({"operation-resolved", "execute-operation", 1.0f, 0.0f});
            } else {
                successors.push_back({"schema-known", "inspect-dataset", 1.0f, 0.0f});
            }
        } else if (state == "second-source-selected") {
            if (request.schema_known && request.second_schema_known) {
                successors.push_back({"operation-resolved", "execute-operation", 1.0f, 0.0f});
            } else if (!request.schema_known) {
                successors.push_back({"left-schema-known", "inspect-dataset", 1.0f, 0.0f});
            } else {
                successors.push_back({"both-schemas-known", "inspect-second-dataset", 1.0f, 0.0f});
            }
        } else if (state == "left-schema-known") {
            if (request.second_schema_known) {
                successors.push_back({"operation-resolved", "execute-operation", 1.0f, 0.0f});
            } else {
                successors.push_back({"both-schemas-known", "inspect-second-dataset", 1.0f, 0.0f});
            }
        } else if (state == "both-schemas-known") {
            successors.push_back({"operation-resolved", "execute-operation", 1.0f, 0.0f});
        } else if (state == "schema-known") {
            successors.push_back({"operation-resolved", "execute-operation", 1.0f, 0.0f});
        }
        return true;
    };
    proposal_request.materialize = [request](
            const std::vector<std::string> &,
            const std::vector<std::string> & actions,
            std::vector<common_tool_workflow_step_view> & steps,
            std::string & materialize_error) {
        steps.clear();
        materialize_error.clear();
        for (const auto & action : actions) {
            if (action == "select-dataset") {
                steps.push_back({"dataset.select", json{{"name", request.dataset_name}}.dump()});
            } else if (action == "inspect-dataset") {
                steps.push_back({"dataset.inspect", json{{"dataset", request.dataset_ref}}.dump()});
            } else if (action == "select-second-dataset") {
                steps.push_back({"dataset.select", json{{"name", request.second_dataset_name}}.dump()});
            } else if (action == "inspect-second-dataset") {
                steps.push_back({"dataset.inspect", json{{"dataset", request.second_dataset_ref}}.dump()});
            } else if (action == "execute-operation") {
                steps.push_back({request.terminal_tool, request.terminal_arguments_json});
            } else {
                materialize_error = "dataset blueprint contains an unknown transition";
                steps.clear();
                return false;
            }
        }
        return !steps.empty();
    };
    if (!common_flydelta_propose_workflow_path(proposal_request, result, error)) {
        return false;
    }
    if (result.proposed) {
        result.binding_refs = {request.dataset_ref};
        if (request.terminal_tool == "data.join") {
            result.binding_refs.push_back(request.second_dataset_ref);
        }
    }
    return true;
}

bool common_flydelta_materialize_workflow_proposal_plan(
        const common_flydelta_workflow_proposal & proposal,
        const common_plan_state & blueprint_instance,
        common_plan_state & materialized,
        std::string & error) {
    error.clear();
    if (!proposal.proposed || proposal.status != common_flydelta_astar_status::found ||
            proposal.canonical_steps.empty() || blueprint_instance.kind != common_plan_kind::task ||
            blueprint_instance.id.empty() || blueprint_instance.session_id.empty()) {
        error = "cannot materialize an unverified or incomplete blueprint workflow proposal";
        return false;
    }
    if (proposal.state_refs.size() != proposal.transition_refs.size() + 1 ||
            proposal.canonical_steps.size() != proposal.transition_refs.size() ||
            proposal.path_fingerprint.empty()) {
        error = "blueprint workflow proposal provenance is incomplete";
        return false;
    }

    common_plan_state out = blueprint_instance;
    out.kind = common_plan_kind::task;
    out.steps.clear();
    out.active_step_id.reset();
    out.next_action.reset();
    out.status = common_plan_status::active;
    out.version = 0;

    std::string previous_id;
    for (size_t index = 0; index < proposal.canonical_steps.size(); ++index) {
        const auto & source = proposal.canonical_steps[index];
        if (source.tool_name.empty() || source.arguments_json.empty()) {
            error = "blueprint workflow proposal contains an incomplete tool step";
            return false;
        }
        common_plan_step step;
        step.id = workflow_step_id(blueprint_instance.id, index);
        step.title = source.tool_name;
        step.objective = "Execute host-verified workflow step: " + source.tool_name;
        step.intended_contribution = step.objective;
        step.status = index == 0
            ? common_plan_step_status::active
            : common_plan_step_status::pending;
        step.mode = common_plan_step_mode::tool;
        step.selected_tool = source.tool_name;
        step.tool_call = common_plan_tool_call{source.tool_name, source.arguments_json};
        if (!previous_id.empty()) step.depends_on.push_back(previous_id);
        step.created_at = blueprint_instance.created_at;
        step.updated_at = blueprint_instance.updated_at;
        step.semantic_alias = proposal.transition_refs[index];
        out.steps.push_back(std::move(step));
        previous_id = out.steps.back().id;
    }

    common_plan_step answer;
    answer.id = blueprint_instance.id + ":astar:answer";
    answer.title = "Answer";
    answer.objective = blueprint_instance.goal.empty()
        ? "Answer from the verified workflow result."
        : blueprint_instance.goal;
    answer.intended_contribution = blueprint_instance.success_criteria.empty()
        ? answer.objective : blueprint_instance.success_criteria;
    answer.mode = common_plan_step_mode::final_response;
    answer.status = common_plan_step_status::pending;
    answer.depends_on.push_back(previous_id);
    answer.created_at = blueprint_instance.created_at;
    answer.updated_at = blueprint_instance.updated_at;
    out.steps.push_back(std::move(answer));
    out.active_step_id = out.steps.front().id;
    out.next_action = out.steps.front().id;
    materialized = std::move(out);
    return true;
}

common_blueprint_instance_materializer
common_flydelta_make_dataset_blueprint_plan_materializer(
        common_flydelta_dataset_blueprint_request_provider request_provider,
        common_flydelta_workflow_proposal_verifier verifier) {
    return [request_provider = std::move(request_provider), verifier = std::move(verifier)](
            const common_agent_request & request,
            const common_plan_state & blueprint_instance,
            common_plan_state & materialized,
            std::string & error) {
        error.clear();
        if (!request_provider) return common_blueprint_materialization_outcome::not_applicable;

        common_flydelta_dataset_blueprint_request dataset_request;
        std::string provider_error;
        if (!request_provider(request, blueprint_instance, dataset_request, provider_error)) {
            if (!provider_error.empty()) {
                error = provider_error;
                return common_blueprint_materialization_outcome::failed_safely;
            }
            return common_blueprint_materialization_outcome::not_applicable;
        }

        common_flydelta_workflow_proposal proposal;
        if (!common_flydelta_propose_dataset_blueprint_workflow(
                dataset_request, proposal, error)) {
            return common_blueprint_materialization_outcome::failed_safely;
        }
        if (!verifier) {
            error = "dataset blueprint planner requires a host Oracle verifier";
            return common_blueprint_materialization_outcome::failed_safely;
        }
        std::string verification_error;
        if (!verifier(proposal, verification_error)) {
            error = verification_error.empty()
                ? "dataset blueprint workflow proposal was not admitted by the host Oracle"
                : verification_error;
            return common_blueprint_materialization_outcome::failed_safely;
        }
        if (!common_flydelta_materialize_workflow_proposal_plan(
                proposal, blueprint_instance, materialized, error)) {
            return common_blueprint_materialization_outcome::failed_safely;
        }
        return common_blueprint_materialization_outcome::applied;
    };
}
