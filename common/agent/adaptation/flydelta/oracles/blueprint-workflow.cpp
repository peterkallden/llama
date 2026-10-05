#include "agent/adaptation/flydelta/oracles/blueprint-workflow.h"

#include <nlohmann/json.hpp>

#include <utility>

using json = nlohmann::ordered_json;

namespace {

bool valid_terminal(const std::string & tool) {
    return tool == "data.aggregate" || tool == "data.filter" ||
        tool == "statistics.describe";
}

bool valid_request(const common_flydelta_dataset_blueprint_request & request) {
    return request.schema_version == 1 &&
        request.proposal_id == "proposal://dataset-inspect-summarize" &&
        request.blueprint_ref == "blueprint://dataset-inspect-summarize" &&
        !request.blueprint_revision.empty() && !request.graph_revision.empty() &&
        !request.workflow_ref.empty() && !request.workflow_revision.empty() &&
        !request.dataset_ref.empty() && valid_terminal(request.terminal_tool) &&
        !request.terminal_arguments_json.empty() && request.max_expansions > 0 &&
        request.max_path_length >= 2 && request.max_path_length <= 8;
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
            if (request.schema_known) {
                successors.push_back({"operation-resolved", "execute-operation", 1.0f, 0.0f});
            } else {
                successors.push_back({"schema-known", "inspect-dataset", 1.0f, 0.0f});
            }
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
                steps.push_back({"dataset.select", json{{"dataset", request.dataset_ref}}.dump()});
            } else if (action == "inspect-dataset") {
                steps.push_back({"dataset.inspect", json{{"dataset", request.dataset_ref}}.dump()});
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
    if (result.proposed) result.binding_refs = {request.dataset_ref};
    return true;
}
