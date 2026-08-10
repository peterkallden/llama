#include "agent-runtime-context.h"

#include "research/research-workspace-factory.h"

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
        context.emit_event(type, std::move(detail));
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
    }
    for (const auto & evidence : context.research_workspace->evidence) {
        context.emit_event(
            common_agent_event_type::research_evidence_recorded,
            "research evidence recorded: " + evidence.evidence_id);
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
    return true;
}
