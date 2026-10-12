#include "agent-resource-workflow-continuation.h"

#include "agent/adaptation/flydelta/oracles/workflow-proposal.h"

#include <algorithm>
#include <nlohmann/json.hpp>

namespace {

bool is_dataset_operation(const std::string & name) {
    return name == "data.query" || name == "data.filter" ||
        name == "data.aggregate" || name == "data.join" ||
        name == "data.transform" || name == "statistics.describe" ||
        name == "statistics.outliers" || name == "statistics.value_counts";
}

} // namespace

common_agent_workflow_continuation_provider
make_agent_resource_document_workflow_continuation(
        common_agent_route_candidate route,
        std::vector<common_agent_tool_argument_binding> bindings) {
    return [route = std::move(route), bindings = std::move(bindings)](
            const common_plan_state & plan,
            const std::string & completed_step_id,
            const common_plan_observation & observation,
            std::vector<common_plan_step> & continuation_steps,
            std::string & error) {
        continuation_steps.clear();
        error.clear();
        if (route.kind != common_agent_route_kind::blueprint_workflow ||
                !route.workflow || route.workflow->workflow_ref !=
                    "workflow://resource/document-analysis" ||
                !plan.route_binding || plan.route_binding->route_id != route.id ||
                !plan.workflow_definition ||
                plan.workflow_definition->workflow_ref != route.workflow->workflow_ref ||
                plan.workflow_definition->workflow_revision != route.workflow->workflow_revision) return true;

        const auto table_step = std::find_if(plan.steps.begin(), plan.steps.end(),
            [](const auto & step) {
                return step.status == common_plan_step_status::completed && step.tool_call &&
                    step.tool_call->name == "document.table";
            });
        if (table_step == plan.steps.end() || observation.id.empty()) return true;

        const auto binding = std::find_if(bindings.begin(), bindings.end(), [&](const auto & value) {
            return is_dataset_operation(value.tool_name) && value.tool_name != "data.join" &&
                std::find(route.resolved_tools.begin(), route.resolved_tools.end(), value.tool_name) !=
                    route.resolved_tools.end();
        });
        if (binding == bindings.end() || binding->tool_name == observation.source) return true;
        const bool already_planned = std::any_of(plan.steps.begin(), plan.steps.end(), [&](const auto & step) {
            return step.tool_call && step.tool_call->name == binding->tool_name &&
                step.status != common_plan_step_status::failed &&
                step.status != common_plan_step_status::skipped;
        });
        if (already_planned) return true;

        auto arguments = nlohmann::ordered_json::parse(binding->arguments_json, nullptr, false);
        if (!arguments.is_object()) {
            error = "host-bound document analysis arguments are invalid JSON";
            return false;
        }
        arguments["dataset"] = nlohmann::ordered_json{
            {"$from_step", table_step->id}, {"$json_pointer", "/dataset"}};
        common_flydelta_workflow_proposal_request proposal_request;
        proposal_request.proposal_id = "proposal://document-continuation/" +
            plan.id + "/" + std::to_string(plan.version);
        proposal_request.blueprint_ref = "blueprint://" + route.blueprint_logical_id;
        proposal_request.blueprint_revision = route.blueprint_revision;
        proposal_request.workflow_ref = route.workflow->workflow_ref;
        proposal_request.workflow_revision = route.workflow->workflow_revision;
        proposal_request.graph_revision = route.graph_revision;
        proposal_request.start_state = "document-table-ready";
        proposal_request.goal_state = "requested-operation-ready";
        proposal_request.max_expansions = 4;
        proposal_request.max_path_length = 2;
        proposal_request.is_goal = [](const std::string & state) {
            return state == "requested-operation-ready";
        };
        proposal_request.expand = [](const std::string & state,
                std::vector<common_flydelta_astar_successor> & successors,
                std::string & expand_error) {
            successors.clear();
            expand_error.clear();
            if (state == "document-table-ready") {
                successors.push_back({"requested-operation-ready", "run-bound-operation", 1.0f, 0.0f});
            }
            return true;
        };
        const std::string operation_tool_name = binding->tool_name;
        proposal_request.materialize = [operation_tool_name, arguments](
                const std::vector<std::string> &,
                const std::vector<std::string> & actions,
                std::vector<common_tool_workflow_step_view> & steps,
                std::string & materialize_error) {
            steps.clear();
            materialize_error.clear();
            if (actions.size() != 1 || actions.front() != "run-bound-operation") {
                materialize_error = "document continuation path is outside its bounded contract";
                return false;
            }
            steps.push_back({operation_tool_name, arguments.dump()});
            return true;
        };
        common_flydelta_workflow_proposal proposal;
        if (!common_flydelta_propose_workflow_path(proposal_request, proposal, error) ||
                !proposal.proposed || proposal.canonical_steps.size() != 1) {
            if (error.empty()) error = "A* could not resolve the requested document continuation";
            return false;
        }
        const auto & proposed = proposal.canonical_steps.front();
        if (std::find(route.resolved_tools.begin(), route.resolved_tools.end(), proposed.tool_name) ==
                route.resolved_tools.end()) {
            error = "A* document continuation proposed a tool outside the selected route";
            return false;
        }
        common_plan_step step;
        step.id = plan.id + ":astar-cont:" + std::to_string(plan.version);
        step.title = proposed.tool_name;
        step.objective = "Continue document analysis with the host-bound operation.";
        step.intended_contribution = step.objective;
        step.status = common_plan_step_status::pending;
        step.depends_on = {completed_step_id};
        step.mode = common_plan_step_mode::tool;
        step.selected_tool = proposed.tool_name;
        step.tool_call = common_plan_tool_call{proposed.tool_name, proposed.arguments_json};
        step.semantic_alias = "astar:" + proposal.path_fingerprint;
        continuation_steps.push_back(std::move(step));
        return true;
    };
}
