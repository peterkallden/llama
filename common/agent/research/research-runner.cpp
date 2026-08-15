#include "agent/research/research-runner.h"

#include "agent/research/research-workspace.h"

#include <algorithm>
#include <chrono>
#include <nlohmann/json.hpp>

namespace {

using json = nlohmann::ordered_json;

bool build_call(
        const std::string & tool_name,
        const std::string & instruction,
        const common_agent_research_workspace & workspace,
        common_agent_tool_call & call) {
    call.name = tool_name;
    if (tool_name == "development.build" || tool_name == "development.test") {
        json arguments = {
            {"target", instruction},
            {"configuration", "Debug"},
        };
        if (tool_name == "development.test") arguments["timeout_ms"] = 120000;
        json resources = json::array();
        for (const auto & source : workspace.sources) {
            if (source.resource_ref && resources.size() < 32) {
                resources.push_back(source.resource_ref->uri);
            }
        }
        if (!resources.empty()) arguments["resource_refs"] = std::move(resources);
        call.arguments_json = std::move(arguments).dump();
        return true;
    }
    if (tool_name == "repository.search") {
        call.arguments_json = json{{"query", instruction}, {"max_results", 8}}.dump();
        return true;
    }
    if (tool_name == "web_search") {
        call.arguments_json = json{{"query", instruction}, {"limit", 5}}.dump();
        return true;
    }
    if (tool_name == "resource_read") {
        call.arguments_json = json{{"uri", instruction}, {"max_bytes", 16384}}.dump();
        return true;
    }
    if (tool_name == "memory_get") {
        call.arguments_json = json{{"id", instruction}}.dump();
        return true;
    }
    if (tool_name == "web_fetch") {
        call.arguments_json = json{{"url", instruction}, {"max_bytes", 32768}, {"extract", "text"}}.dump();
        return true;
    }
    return false;
}

std::string bounded_summary(const common_tool_execution_result & result) {
    std::string summary = !result.content_summary.empty()
        ? result.content_summary
        : (!result.safe_summary.empty() ? result.safe_summary : result.output);
    if (summary.size() > 2048) summary.resize(2048);
    return summary;
}

std::string existing_source_id_for_resource(
        const common_agent_research_workspace & workspace,
        const std::vector<common_runtime_resource_ref> & resources,
        const std::string & requested_uri) {
    for (const auto & source : workspace.sources) {
        if (source.resource_ref && source.resource_ref->uri == requested_uri) return source.source_id;
    }
    for (const auto & resource : resources) {
        for (const auto & source : workspace.sources) {
            if (source.resource_ref && source.resource_ref->uri == resource.uri) return source.source_id;
        }
    }
    return {};
}

std::string existing_source_id_for_memory(
        const common_agent_research_workspace & workspace,
        const std::string & memory_id) {
    for (const auto & source : workspace.sources) {
        if (source.kind == common_agent_research_source_kind::memory && source.memory_id == memory_id) {
            return source.source_id;
        }
    }
    return {};
}

} // namespace

common_agent_research_runtime_adapter::common_agent_research_runtime_adapter(
        const common_agent_tool_runtime & tools,
        const common_agent_research_assessor * assessor)
    : tools(tools), assessor(assessor) {}

bool common_agent_research_runtime_adapter::execute(
        const common_agent_research_action & action,
        common_agent_research_workspace & workspace,
        common_agent_research_event & event,
        std::string & error) const {
    event = {};
    event.task_id = action.task_id;
    event.gap_id = action.gap_id;

    std::string last_failure_code;
    std::string last_failure_summary;
    bool last_failure_retryable = false;
    bool executed_failure = false;
    for (const auto & tool_name : action.preferred_tools) {
        common_agent_tool_call call;
        if (!build_call(tool_name, action.instruction, workspace, call)) continue;
        std::string validation_error;
        if (!tools.validate(call, validation_error)) {
            if (!executed_failure) {
                last_failure_code = "tool.invalid_arguments";
                last_failure_summary = validation_error;
                last_failure_retryable = false;
            }
            continue;
        }

        const auto result = tools.execute(call);
        if (!result.ok) {
            executed_failure = true;
            last_failure_code = result.failure_code;
            last_failure_summary = result.safe_summary;
            last_failure_retryable = result.retryable;
            continue;
        }

        const std::string existing_source_id = existing_source_id_for_resource(
            workspace, result.resource_refs, action.instruction);
        const std::string memory_source_id = tool_name == "memory_get"
            ? existing_source_id_for_memory(workspace, action.instruction)
            : std::string();
        const std::string source_id = !memory_source_id.empty()
            ? memory_source_id
            : (existing_source_id.empty() ? action.task_id + ":source" : existing_source_id);
        if (!existing_source_id.empty() || !memory_source_id.empty()) {
            common_agent_research_evidence evidence;
            evidence.evidence_id = action.task_id + ":evidence";
            evidence.source_id = source_id;
            evidence.claim_id = action.gap_id;
            evidence.statement = bounded_summary(result);
            if (evidence.statement.empty()) evidence.statement = "Research resource was read successfully.";
            evidence.source_location = result.resource_refs.empty()
                ? action.instruction
                : result.resource_refs.front().uri;
            evidence.relation = common_agent_research_evidence_relation::supports;
            evidence.origin = common_agent_research_evidence_origin::normalized_tool_result;
            evidence.relevance = 0.75;
            evidence.confidence = 0.9;
            if (!common_agent_research_record_evidence(workspace, evidence, error)) return false;
            const auto gap = std::find_if(workspace.gaps.begin(), workspace.gaps.end(),
                [&](const auto & candidate) { return candidate.gap_id == action.gap_id; });
            event.type = common_agent_research_event_type::task_completed;
            event.evidence_count = 1;
            event.evidence_ids = {evidence.evidence_id};
            event.gap_sufficiently_answered = !evidence.statement.empty();
            event.gap_confidence = evidence.confidence;
            event.assessment_summary = event.gap_sufficiently_answered
                ? "bounded resource evidence addresses the gap criterion"
                : "resource evidence did not address the gap criterion";
            if (assessor && gap != workspace.gaps.end()) {
                event.assessment = assessor->assess(*gap, evidence);
                event.gap_sufficiently_answered =
                    event.assessment.status != common_agent_research_assessment_status::insufficient &&
                    event.assessment.status != common_agent_research_assessment_status::contradicted;
                event.assessment_summary = event.assessment.summary;
            }
            event.assessment = {
                event.assessment.status == common_agent_research_assessment_status::inconclusive
                    ? (event.gap_sufficiently_answered
                        ? common_agent_research_assessment_status::inconclusive
                        : common_agent_research_assessment_status::insufficient)
                    : event.assessment.status,
                event.assessment.confidence > 0.0 ? event.assessment.confidence : evidence.confidence,
                event.assessment.summary.empty() ? event.assessment_summary : event.assessment.summary};
            error.clear();
            return true;
        }
        common_agent_research_source source;
        source.source_id = source_id;
        source.title = tool_name;
        source.origin = tool_name;
        source.authority = "host-tool-runtime";
        source.kind = tool_name == "web_search" || tool_name == "web_fetch"
            ? common_agent_research_source_kind::web_page
            : (!result.resource_refs.empty() &&
                result.resource_refs.front().uri.rfind("artifact://", 0) == 0
                ? common_agent_research_source_kind::remote_agent_result
                : common_agent_research_source_kind::repository_file);
        source.quality_score = result.resource_refs.empty() ? 0.5 : 0.75;
        source.primary_source = true;
        if (!result.resource_refs.empty()) {
            source.resource_ref = result.resource_refs.front();
            source.content_hash = result.resource_refs.front().uri;
        } else {
            source.content_hash = tool_name + ":" + action.instruction;
        }
        if (!common_agent_research_add_source(workspace, source, error)) return false;

        common_agent_research_evidence evidence;
        evidence.evidence_id = action.task_id + ":evidence";
        evidence.source_id = source_id;
        evidence.claim_id = action.gap_id;
        evidence.statement = bounded_summary(result);
        if (evidence.statement.empty()) evidence.statement = "Research tool returned a successful result.";
        evidence.source_location = result.resource_refs.empty()
            ? action.instruction
            : result.resource_refs.front().uri;
        evidence.relation = common_agent_research_evidence_relation::supports;
        evidence.origin = common_agent_research_evidence_origin::normalized_tool_result;
        evidence.relevance = 0.75;
        evidence.confidence = source.quality_score;
        if (!common_agent_research_record_evidence(workspace, evidence, error)) return false;

        event.type = common_agent_research_event_type::task_completed;
        event.evidence_count = 1;
        event.evidence_ids = {evidence.evidence_id};
        // The v1 adapter exposes an explicit bounded assessment seam. A
        // future model-backed assessor can replace this decision without
        // changing controller or workspace contracts.
        const auto gap = std::find_if(workspace.gaps.begin(), workspace.gaps.end(),
            [&](const auto & candidate) { return candidate.gap_id == action.gap_id; });
        event.gap_sufficiently_answered = !evidence.statement.empty();
        event.gap_confidence = evidence.confidence;
        event.assessment_summary = event.gap_sufficiently_answered
            ? "bounded tool evidence addresses the gap criterion"
            : "tool evidence did not address the gap criterion";
        if (assessor && gap != workspace.gaps.end()) {
            event.assessment = assessor->assess(*gap, evidence);
            event.gap_sufficiently_answered =
                event.assessment.status != common_agent_research_assessment_status::insufficient &&
                event.assessment.status != common_agent_research_assessment_status::contradicted;
            event.assessment_summary = event.assessment.summary;
        }
        event.assessment = {
            event.assessment.status == common_agent_research_assessment_status::inconclusive
                ? (event.gap_sufficiently_answered
                    ? common_agent_research_assessment_status::inconclusive
                    : common_agent_research_assessment_status::insufficient)
                : event.assessment.status,
            event.assessment.confidence > 0.0 ? event.assessment.confidence : evidence.confidence,
            event.assessment.summary.empty() ? event.assessment_summary : event.assessment.summary};
        error.clear();
        return true;
    }

    event.type = common_agent_research_event_type::task_failed;
    event.retryable = last_failure_retryable;
    event.failure_code = (!executed_failure && last_failure_code.empty())
        ? "research.acquisition_failed" : last_failure_code;
    event.failure_summary = last_failure_summary.empty()
        ? "All host-approved acquisition tools failed."
        : last_failure_summary;
    error.clear();
    return true;
}

common_agent_research_result common_agent_research_runner::run(
        common_agent_research_workspace & workspace,
        const common_agent_research_tool_executor & executor,
        std::string & error,
        const std::function<bool()> & should_stop,
        const std::function<common_agent_research_stop_reason()> & stop_reason,
        const common_agent_research_lifecycle_sink & lifecycle_sink) const {
    common_agent_research_controller controller;
    const auto emit_lifecycle = [&](common_agent_research_lifecycle_event event) {
        if (lifecycle_sink) lifecycle_sink(event);
    };
    for (const auto & gap : workspace.gaps) {
        emit_lifecycle({common_agent_research_lifecycle_event_type::gap_opened, gap.gap_id, {}, 0});
    }
    const auto finalize_with_reason = [&](common_agent_research_stop_reason reason) {
        auto result = controller.finalize(workspace);
        result.stop_reason = reason;
        result.complete = false;
        return result;
    };
    const auto started_at = std::chrono::steady_clock::now();
    auto action = controller.begin(workspace, error);
    if (!error.empty()) return controller.finalize(workspace);

    int iterations = 0;
    while (action.kind == common_agent_research_action_kind::schedule_task) {
        if (should_stop && should_stop()) {
            return finalize_with_reason(
                stop_reason ? stop_reason() : common_agent_research_stop_reason::cancelled);
        }
        if (iterations++ >= workspace.budget.max_iterations) {
            auto result = controller.finalize(workspace);
            result.stop_reason = common_agent_research_stop_reason::budget_exhausted;
            result.complete = false;
            return result;
        }
        if (workspace.tool_calls >= workspace.budget.max_tool_calls) {
            auto result = controller.finalize(workspace);
            result.stop_reason = common_agent_research_stop_reason::budget_exhausted;
            result.complete = false;
            return result;
        }
        if (workspace.budget.max_elapsed_ms > 0 &&
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - started_at).count() >= workspace.budget.max_elapsed_ms) {
            auto result = controller.finalize(workspace);
            result.stop_reason = common_agent_research_stop_reason::deadline_exceeded;
            result.complete = false;
            return result;
        }
        bool retry = false;
        for (const auto & task : workspace.tasks) {
            if (task.task_id == action.task_id) {
                retry = task.attempt > 0;
                break;
            }
        }
        emit_lifecycle({common_agent_research_lifecycle_event_type::task_scheduled,
            action.gap_id, action.task_id, iterations, retry});
        emit_lifecycle({common_agent_research_lifecycle_event_type::task_started,
            action.gap_id, action.task_id, iterations});
        common_agent_research_event event;
        ++workspace.tool_calls;
        if (!executor.execute(action, workspace, event, error)) return controller.finalize(workspace);
        emit_lifecycle({
            event.type == common_agent_research_event_type::task_failed
                ? common_agent_research_lifecycle_event_type::task_failed
                : common_agent_research_lifecycle_event_type::task_completed,
            action.gap_id,
            action.task_id,
            iterations});
        emit_lifecycle({common_agent_research_lifecycle_event_type::iteration_completed,
            action.gap_id, action.task_id, iterations});
        if (should_stop && should_stop()) {
            return finalize_with_reason(
                stop_reason ? stop_reason() : common_agent_research_stop_reason::cancelled);
        }
        ++workspace.iterations_completed;
        const size_t comparisons_before = workspace.comparisons.size();
        action = controller.advance(workspace, event, error);
        if (workspace.comparisons.size() > comparisons_before) {
            emit_lifecycle({common_agent_research_lifecycle_event_type::sources_compared,
                action.gap_id.empty() ? event.gap_id : action.gap_id,
                action.task_id.empty() ? event.task_id : action.task_id,
                iterations});
        }
        if (!error.empty()) return controller.finalize(workspace);
    }

    auto result = controller.finalize(workspace);
    result.stop_reason = action.stop_reason;
    result.complete = action.stop_reason == common_agent_research_stop_reason::sufficient_coverage ||
        action.stop_reason == common_agent_research_stop_reason::success_criteria_met;
    return result;
}
