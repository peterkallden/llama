#include "agent/agent-runtime-context.h"

#include "agent/thinking/research/research-workspace-factory.h"

#include <algorithm>
#include <iomanip>
#include <sstream>

namespace {

bool bridge_research_result_to_plan(
        common_agent_runtime_turn_context & context,
        const common_agent_research_result & research_result) {
    if (context.plan_store == nullptr || context.outer_plan == nullptr ||
            context.outer_plan->id.empty()) {
        context.error = "research completion requires the active outer plan";
        return false;
    }

    const std::string observation_id =
        "research:completion:" + research_result.workspace_id;
    if (std::find_if(
            context.outer_plan->observations.begin(),
            context.outer_plan->observations.end(),
            [&](const auto & observation) { return observation.id == observation_id; }) !=
            context.outer_plan->observations.end()) {
        return true;
    }

    common_plan_observation observation;
    observation.id = observation_id;
    observation.source = "research_workspace";
    std::ostringstream summary;
    summary << "Research workspace completed with coverage="
            << std::fixed << std::setprecision(3) << research_result.coverage.objective_coverage
            << ", sources=" << research_result.sources.size()
            << ", evidence=" << research_result.evidence.size()
            << ", comparisons=" << research_result.comparisons.size();
    observation.summary = summary.str();
    observation.confidence = static_cast<float>(std::clamp(
        research_result.coverage.evidence_quality, 0.0, 1.0));
    for (const auto & evidence : research_result.evidence) {
        if (observation.evidence_ids.size() >= 32) break;
        observation.evidence_ids.push_back(evidence.evidence_id);
        for (const auto & source : research_result.sources) {
            if (source.source_id != evidence.source_id || !source.resource_ref) continue;
            if (std::find_if(
                    observation.resource_refs.begin(), observation.resource_refs.end(),
                    [&](const auto & ref) { return ref.uri == source.resource_ref->uri; }) ==
                    observation.resource_refs.end() && observation.resource_refs.size() < 32) {
                observation.resource_refs.push_back(*source.resource_ref);
            }
            break;
        }
    }

    common_plan_operation operation;
    operation.kind = common_plan_operation_kind::record_observation;
    operation.plan_id = context.outer_plan->id;
    operation.expected_version = context.outer_plan->version;
    operation.reason_summary = "research workspace completion evidence";
    operation.evidence_ids = observation.evidence_ids;
    operation.observation = std::move(observation);
    if (!context.plan_store->apply(operation, *context.outer_plan, context.error)) return false;

    context.emit_event(
        common_agent_event_type::plan_updated,
        "research completion bridged to outer plan evidence");
    context.emit_trace(
        common_runtime_trace_stage::observation,
        common_runtime_trace_kind::recorded,
        "research completion bridged to outer plan evidence",
        context.outer_plan->id,
        observation_id);
    return true;
}

} // namespace

common_agent_research_lifecycle_sink make_common_agent_research_lifecycle_sink(
        common_agent_runtime_turn_context & context) {
    return [&context](const common_agent_research_lifecycle_event & event) {
        common_agent_event_type type = common_agent_event_type::research_gap_opened;
        switch (event.type) {
            case common_agent_research_lifecycle_event_type::gap_opened:
                type = common_agent_event_type::research_gap_opened;
                break;
            case common_agent_research_lifecycle_event_type::task_scheduled:
                type = common_agent_event_type::research_task_scheduled;
                break;
            case common_agent_research_lifecycle_event_type::task_started:
                type = common_agent_event_type::research_task_started;
                break;
            case common_agent_research_lifecycle_event_type::task_completed:
                type = common_agent_event_type::research_task_completed;
                break;
            case common_agent_research_lifecycle_event_type::task_failed:
                type = common_agent_event_type::research_task_failed;
                break;
            case common_agent_research_lifecycle_event_type::iteration_completed:
                type = common_agent_event_type::research_iteration_completed;
                break;
            case common_agent_research_lifecycle_event_type::sources_compared:
                type = common_agent_event_type::research_sources_compared;
                break;
        }
        std::string detail = "research";
        if (!event.gap_id.empty()) detail += " gap=" + event.gap_id;
        if (!event.task_id.empty()) detail += " task=" + event.task_id;
        if (event.iteration > 0) detail += " iteration=" + std::to_string(event.iteration);
        if (event.retry) detail += " retry=true";
        context.emit_event(type, detail);

        common_runtime_trace_kind trace_kind = common_runtime_trace_kind::recorded;
        switch (event.type) {
            case common_agent_research_lifecycle_event_type::gap_opened:
            case common_agent_research_lifecycle_event_type::task_started:
                trace_kind = common_runtime_trace_kind::started;
                break;
            case common_agent_research_lifecycle_event_type::task_completed:
                trace_kind = common_runtime_trace_kind::completed;
                break;
            case common_agent_research_lifecycle_event_type::task_failed:
                trace_kind = common_runtime_trace_kind::failed;
                break;
            case common_agent_research_lifecycle_event_type::task_scheduled:
            case common_agent_research_lifecycle_event_type::sources_compared:
                trace_kind = common_runtime_trace_kind::decided;
                break;
            case common_agent_research_lifecycle_event_type::iteration_completed:
                trace_kind = common_runtime_trace_kind::recorded;
                break;
        }
        context.emit_trace(
            common_runtime_trace_stage::research,
            trace_kind,
            detail,
            context.request.plan_id.value_or(""),
            event.task_id.empty() ? event.gap_id : event.task_id);
    };
}

bool run_common_agent_research_phase(
        common_agent_runtime_turn_context & context) {
    if (context.request.deliberation_policy.mode != common_agent_thinking_mode::research) {
        return true;
    }
    if (context.tools == nullptr) {
        context.result.error = "research mode requires a host-approved tool runtime";
        return false;
    }

    context.research_workspace.emplace();
    if (!common_agent_research_create_workspace(
            context.request, *context.research_workspace, context.error)) {
        context.result.error = context.error;
        return false;
    }

    context.emit_event(
        common_agent_event_type::research_started,
        "research acquisition started plan_id=" + context.request.plan_id.value_or(""));
    context.emit_trace(
        common_runtime_trace_stage::research,
        common_runtime_trace_kind::started,
        "research acquisition started",
        context.request.plan_id.value_or(""),
        context.research_workspace->workspace_id);

    context.research_assessor = std::make_unique<common_agent_research_bounded_assessor>();
    context.research_adapter = std::make_unique<common_agent_research_runtime_adapter>(
        *context.tools, context.research_assessor.get());
    context.research_runner = std::make_unique<common_agent_research_runner>();

    const auto research_result = context.research_runner->run(
        *context.research_workspace,
        *context.research_adapter,
        context.error,
        context.request.research_should_stop,
        context.request.research_stop_reason,
        make_common_agent_research_lifecycle_sink(context));
    context.result.research_result = research_result;

    if (!research_result.complete) {
        std::string checkpoint_error;
        context.result.research_workspace_checkpoint =
            make_common_agent_research_workspace_checkpoint(
                *context.research_workspace,
                static_cast<size_t>(context.research_workspace->iterations_completed) + 1,
                checkpoint_error);
        if (!checkpoint_error.empty()) {
            context.result.error = "research checkpoint creation failed safely: " + checkpoint_error;
            return false;
        }
        context.emit_trace(
            common_runtime_trace_stage::research,
            common_runtime_trace_kind::recorded,
            "research workspace checkpoint created",
            context.request.plan_id.value_or(""),
            context.research_workspace->workspace_id);
    }

    for (const auto & source : context.research_workspace->sources) {
        context.emit_event(
            common_agent_event_type::research_source_recorded,
            "research source recorded: " + source.source_id);
        context.emit_trace(
            common_runtime_trace_stage::research,
            common_runtime_trace_kind::recorded,
            "research source recorded source=" + source.source_id,
            context.request.plan_id.value_or(""),
            source.source_id);
    }
    for (const auto & evidence : context.research_workspace->evidence) {
        context.emit_event(
            common_agent_event_type::research_evidence_recorded,
            "research evidence recorded: " + evidence.evidence_id);
        context.emit_trace(
            common_runtime_trace_stage::research,
            common_runtime_trace_kind::recorded,
            "research evidence recorded evidence=" + evidence.evidence_id +
                " source=" + evidence.source_id,
            context.request.plan_id.value_or(""),
            evidence.evidence_id);
    }
    for (const auto & comparison : context.research_workspace->comparisons) {
        context.emit_trace(
            common_runtime_trace_stage::research,
            common_runtime_trace_kind::decided,
            "research sources compared comparison=" + comparison.comparison_id +
                " sources=" + std::to_string(comparison.source_ids.size()),
            context.request.plan_id.value_or(""),
            comparison.comparison_id);
    }

    if (!context.error.empty() || !research_result.complete) {
        context.emit_event(
            common_agent_event_type::research_incomplete,
            "research stopped before sufficient evidence coverage");
        context.emit_trace(
            common_runtime_trace_stage::research,
            common_runtime_trace_kind::failed,
            "research incomplete",
            {},
            context.research_workspace->workspace_id);
        context.result.error = context.error.empty()
            ? "research did not meet its evidence coverage requirement"
            : context.error;
        return false;
    }

    context.emit_event(
        common_agent_event_type::research_completed,
        "research completed with sufficient evidence coverage");
    context.emit_trace(
        common_runtime_trace_stage::research,
        common_runtime_trace_kind::completed,
        "research completed",
        {},
        context.research_workspace->workspace_id);
    context.research_synthesis_context = research_result.synthesis_context;
    if (!bridge_research_result_to_plan(context, research_result)) {
        context.result.error = context.error;
        return false;
    }
    return true;
}
