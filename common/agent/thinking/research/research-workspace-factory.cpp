#include "agent/thinking/research/research-workspace-factory.h"

#include "agent/thinking/research/research-workspace.h"

#include <algorithm>
#include <limits>

bool common_agent_research_create_workspace(
        const common_agent_request & request,
        common_agent_research_workspace & workspace,
        std::string & error) {
    if (request.prompt.empty()) {
        error = "research workspace requires a non-empty prompt";
        return false;
    }

    const bool resumed = request.research_workspace_seed.has_value();
    if (resumed && !common_agent_research_workspace_validate(*request.research_workspace_seed, error)) return false;
    workspace = resumed ? *request.research_workspace_seed : common_agent_research_workspace{};
    workspace.workspace_id = request.turn_id.empty() ? "research:turn" : "research:" + request.turn_id;
    if (request.research_draft) {
        workspace.research_id = request.research_draft->research_id;
        workspace.draft_id = request.research_draft->draft_id;
        workspace.base_revision = request.research_draft->base_revision;
        workspace.revision = request.research_draft->draft_revision;
    } else if (workspace.research_id.empty()) {
        workspace.research_id = workspace.workspace_id;
    }
    workspace.request_id = request.turn_id;
    workspace.turn_id = request.turn_id;
    workspace.session_id = request.session_id;
    workspace.plan_id = request.plan_id.value_or("");
    workspace.scope = common_agent_scope_from_request(request);

    if (!resumed) {
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
    }

    // Counters and tasks are retained as research history. Each new execution
    // gets only the current request's additional budget, rather than inheriting
    // an already exhausted ceiling or discarding earlier work.
    const auto grant_budget = [&](size_t used, size_t granted, int & ceiling) {
        const auto maximum = static_cast<size_t>(std::numeric_limits<int>::max());
        if (used > maximum || granted > maximum - used) {
            error = "research execution budget exceeds its supported counter range";
            return false;
        }
        ceiling = static_cast<int>(used + granted);
        return true;
    };
    const size_t iterations = request.deliberation_policy.max_research_iterations > 0
        ? static_cast<size_t>(request.deliberation_policy.max_research_iterations) : request.max_iterations;
    if (!grant_budget(static_cast<size_t>(workspace.iterations_completed), iterations, workspace.budget.max_iterations) ||
            !grant_budget(workspace.tasks.size(), request.max_tool_batches, workspace.budget.max_tasks) ||
            !grant_budget(static_cast<size_t>(workspace.tool_calls), request.max_tool_batches, workspace.budget.max_tool_calls))
        return false;
    workspace.no_progress_iterations = 0;
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
        const bool already_present = std::any_of(workspace.sources.begin(), workspace.sources.end(), [&](const auto & existing) {
            return existing.resource_ref && existing.resource_ref->uri == input.resource.uri && existing.role == input.role;
        });
        if (already_present) continue;
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
        source.memory_kind = hit.memory.kind;
        source.availability = common_agent_research_source_availability::materialized_context;
        source.role = hit.memory.kind == common_memory_kind::procedure ||
                hit.memory.kind == common_memory_kind::constraint ||
                hit.memory.kind == common_memory_kind::decision
            ? "method_context" : "memory_context";
        source.quality_score = hit.final_score > 0.0f ? std::min(1.0f, hit.final_score) : hit.memory.confidence;
        source.primary_source = false;
        if (std::any_of(workspace.sources.begin(), workspace.sources.end(), [&](const auto & existing) {
                return existing.kind == common_agent_research_source_kind::memory && existing.memory_id == hit.memory.id;
            })) continue;
        if (!common_agent_research_add_source(workspace, std::move(source), error)) return false;
    }

    if (resumed) {
        const bool known = std::any_of(workspace.gaps.begin(), workspace.gaps.end(), [&](const auto & gap) {
            return gap.question == request.prompt;
        });
        if (!known && request.prompt != workspace.objective.question &&
                !common_agent_research_add_gap(workspace, {
                    workspace.workspace_id + ":gap:turn", request.prompt,
                    "follow-up research question", "evidence must address the current chat request", 1,
                    common_agent_research_gap_status::open, common_agent_research_gap_assessment::unknown, {}, {}}, error)) {
            return false;
        }
    } else if (!request.objective || request.objective->success_criteria.empty()) {
        return common_agent_research_add_gap(workspace, {
            workspace.workspace_id + ":gap:objective",
            request.prompt,
            "research objective",
            "evidence must directly address the research question",
            1,
            common_agent_research_gap_status::open, common_agent_research_gap_assessment::unknown, {}, {}}, error);
    }
    if (!resumed && request.objective) {
        for (size_t index = 0; index < request.objective->success_criteria.size(); ++index) {
            const auto & criterion = request.objective->success_criteria[index];
            if (!common_agent_research_add_gap(workspace, {
                    workspace.workspace_id + ":gap:criterion:" + std::to_string(index + 1),
                    criterion,
                    "research success criterion",
                    criterion,
                    static_cast<int>(request.objective->success_criteria.size() - index),
                    common_agent_research_gap_status::open, common_agent_research_gap_assessment::unknown, {}, {}}, error)) {
                return false;
            }
        }
    }
    error.clear();
    return true;
}
