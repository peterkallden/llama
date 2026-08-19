#include "agent/thinking/research/research-workspace-factory.h"

#include "agent/thinking/research/research-workspace.h"

#include <algorithm>

bool common_agent_research_create_workspace(
        const common_agent_request & request,
        common_agent_research_workspace & workspace,
        std::string & error) {
    if (request.prompt.empty()) {
        error = "research workspace requires a non-empty prompt";
        return false;
    }

    workspace = {};
    workspace.workspace_id = request.turn_id.empty() ? "research:turn" : "research:" + request.turn_id;
    workspace.request_id = request.turn_id;
    workspace.turn_id = request.turn_id;
    workspace.session_id = request.session_id;
    workspace.plan_id = request.plan_id.value_or("");
    workspace.scope = common_agent_scope_from_request(request);

    workspace.objective.objective_id = workspace.workspace_id + ":objective";
    workspace.objective.question = request.prompt;
    workspace.objective.purpose = request.objective && !request.objective->purpose.empty()
        ? request.objective->purpose
        : request.prompt;
    workspace.objective.expected_output = request.objective && !request.objective->desired_outcome.empty()
        ? request.objective->desired_outcome
        : request.prompt;
    if (request.objective) {
        workspace.objective.success_criteria = request.objective->success_criteria;
        workspace.objective.constraints = request.objective->constraints;
    }

    workspace.budget.max_iterations = request.deliberation_policy.max_research_iterations > 0
        ? request.deliberation_policy.max_research_iterations
        : static_cast<int>(request.max_iterations);
    workspace.budget.max_tasks = static_cast<int>(request.max_tool_batches);
    workspace.budget.max_tool_calls = static_cast<int>(request.max_tool_batches);
    workspace.budget.minimum_coverage = 1.0;
    workspace.budget.minimum_sources = request.deliberation_policy.require_source_cross_check ? 2 : 1;

    for (size_t index = 0; index < request.input_resources.size(); ++index) {
        const auto & input = request.input_resources[index];
        if (input.resource.uri.empty()) {
            error = "research input resource requires a URI";
            return false;
        }
        common_agent_research_source source;
        source.source_id = workspace.workspace_id + ":input:" + std::to_string(index + 1);
        source.title = input.resource.name.empty() ? input.resource.uri : input.resource.name;
        source.origin = "user";
        source.authority = "user-supplied";
        source.kind = common_agent_research_source_kind::user_supplied;
        source.resource_ref = input.resource;
        // A URI identifies the stored object, but is not a content hash.
        source.content_hash.clear();
        source.quality_score = 1.0;
        source.primary_source = input.role == "primary_source";
        source.role = input.role;
        source.required = input.required;
        if (!common_agent_research_add_source(workspace, std::move(source), error)) return false;
    }

    for (size_t index = 0; index < request.memories.size(); ++index) {
        const auto & hit = request.memories[index];
        common_agent_research_source source;
        source.source_id = workspace.workspace_id + ":memory:" + std::to_string(index + 1);
        source.memory_id = hit.memory.id;
        source.content_hash = hit.memory.id;
        source.title = hit.memory.summary.empty() ? hit.memory.id : hit.memory.summary;
        source.origin = "memory";
        source.authority = "host-memory-store";
        source.kind = common_agent_research_source_kind::memory;
        source.quality_score = hit.final_score > 0.0f ? std::min(1.0f, hit.final_score) : hit.memory.confidence;
        source.primary_source = false;
        if (!common_agent_research_add_source(workspace, std::move(source), error)) return false;
    }

    if (!request.objective || request.objective->success_criteria.empty()) {
        return common_agent_research_add_gap(workspace, {
            workspace.workspace_id + ":gap:objective",
            request.prompt,
            "research objective",
            "evidence must directly address the research question",
            1}, error);
    }
    for (size_t index = 0; index < request.objective->success_criteria.size(); ++index) {
        const auto & criterion = request.objective->success_criteria[index];
        if (!common_agent_research_add_gap(workspace, {
                workspace.workspace_id + ":gap:criterion:" + std::to_string(index + 1),
                criterion,
                "research success criterion",
                criterion,
                static_cast<int>(request.objective->success_criteria.size() - index)}, error)) {
            return false;
        }
    }
    error.clear();
    return true;
}
