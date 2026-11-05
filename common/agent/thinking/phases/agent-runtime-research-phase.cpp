#include "agent/agent-runtime-context.h"

#include "agent/thinking/research/research-workspace-factory.h"
#include "agent/thinking/research/research-workspace.h"
#include "agent/thinking/research/research-assessor.h"

#include <algorithm>
#include <iomanip>
#include <sstream>
#include <nlohmann/json.hpp>

namespace {

const char * research_stop_reason_name(common_agent_research_stop_reason reason) {
    switch (reason) {
        case common_agent_research_stop_reason::success_criteria_met: return "success_criteria_met";
        case common_agent_research_stop_reason::sufficient_coverage: return "sufficient_coverage";
        case common_agent_research_stop_reason::budget_exhausted: return "budget_exhausted";
        case common_agent_research_stop_reason::deadline_exceeded: return "deadline_exceeded";
        case common_agent_research_stop_reason::cancelled: return "cancelled";
        case common_agent_research_stop_reason::no_progress: return "no_progress";
        case common_agent_research_stop_reason::source_access_blocked: return "source_access_blocked";
        case common_agent_research_stop_reason::policy_blocked: return "policy_blocked";
    }
    return "unknown";
}

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
    summary << (research_result.complete ? "Research workspace completed with coverage=" : "Research workspace incomplete with coverage=")
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

bool seed_research_from_workflow_resources(
        common_agent_runtime_turn_context & context,
        common_agent_research_workspace & workspace) {
    if (context.outer_plan == nullptr || !context.outer_plan->route_binding) return true;

    // Reuse only host-validated compact findings. The helper deliberately
    // leaves multi-gap research unseeded rather than guessing claim mapping.
    if (!common_agent_research_seed_chunk_observations(
            workspace, context.outer_plan->observations, context.error)) return false;

    const bool has_host_chunk_observations = std::any_of(
        context.outer_plan->observations.begin(), context.outer_plan->observations.end(),
        [](const auto & observation) {
            return observation.source == "resource_chunk" &&
                observation.resource_refs.size() == 1 &&
                common_runtime_resource_lineage_is_chunk(observation.resource_refs.front().lineage);
        });

    for (const auto & step : context.outer_plan->steps) {
        if (step.status != common_plan_step_status::completed || !step.tool_call) continue;
        const std::string observation_prefix = "tool:" + step.id + ":";
        // Collection identities are host-projected discovery candidates, not
        // findings. Preserve their provider identity when a workflow hands
        // off to research; never select a candidate on the model's behalf.
        for (const auto & observation : context.outer_plan->observations) {
            if (observation.id.rfind(observation_prefix, 0) != 0 || observation.confidence <= 0.0f ||
                    observation.resource_refs.empty()) continue;
            const auto & ref = observation.resource_refs.front();
            const auto & provider = ref.metadata.source_provider;
            const auto & operation = ref.metadata.source_operation;
            const std::string suffix = "." + operation;
            if (provider.empty() || operation.empty() || step.tool_call->name.size() <= suffix.size() ||
                    step.tool_call->name.compare(step.tool_call->name.size() - suffix.size(), suffix.size(), suffix) != 0)
                continue;
            const auto payload = nlohmann::ordered_json::parse(observation.summary, nullptr, false);
            if (!payload.is_object() || (payload.contains("result_truncated") && payload["result_truncated"] == true)) continue;
            const auto & data = payload.contains("result") && payload["result"].is_object() ? payload["result"] : payload;
            if (!data.contains("host_item_parameter") || !data["host_item_parameter"].is_string() ||
                    !data.contains("host_item_candidates") || !data["host_item_candidates"].is_array()) continue;
            const auto key = data["host_item_parameter"].get<std::string>();
            if (key.empty() || key.size() > 128) continue;
            size_t index = 0;
            for (const auto & item : data["host_item_candidates"]) {
                if (++index > 5 || !item.is_object() || !item.contains(key) || !item[key].is_string()) continue;
                const auto id = item[key].get<std::string>();
                if (id.empty() || id.size() > 2048 || std::any_of(workspace.sources.begin(), workspace.sources.end(),
                        [&](const auto & source) { return source.role == "api_discovery_candidate" &&
                            source.authority == provider && source.origin == id; })) continue;
                if (common_agent_research_candidate_count(workspace) >=
                        static_cast<size_t>(workspace.budget.max_candidates)) break;
                common_agent_research_source candidate;
                candidate.source_id = workspace.workspace_id + ":workflow-candidate:" + step.id + ":" + std::to_string(index);
                candidate.title = id;
                for (const char * name : {"display_name", "title"})
                    if (item.contains(name) && item[name].is_string() && !item[name].get_ref<const std::string &>().empty()) {
                        candidate.title = item[name].get<std::string>().substr(0, 512); break;
                    }
                candidate.origin = id;
                candidate.authority = provider;
                candidate.kind = common_agent_research_source_kind::api_result;
                candidate.role = "api_discovery_candidate";
                if (!common_agent_research_add_source(workspace, std::move(candidate), context.error)) return false;
            }
        }
        if (
                (step.tool_call->name != "openalex.getWork" &&
                 step.tool_call->name != "web.fetch" &&
                 step.tool_call->name != "resource.read")) continue;
        // Chunk observations already carry the host-read text evidence and
        // provenance for this workflow. Do not schedule the fetched PDF or
        // OCR representation for a second, whole-resource research read.
        if (has_host_chunk_observations &&
                (step.tool_call->name == "web.fetch" || step.tool_call->name == "resource.read")) continue;
        for (const auto & observation : context.outer_plan->observations) {
            if (observation.id.rfind(observation_prefix, 0) != 0) continue;
            for (size_t index = 0; index < observation.resource_refs.size(); ++index) {
                const auto & ref = observation.resource_refs[index];
                if (ref.uri.empty() || std::any_of(workspace.sources.begin(), workspace.sources.end(),
                        [&](const auto & source) {
                            return source.resource_ref && source.resource_ref->uri == ref.uri;
                        })) continue;
                common_agent_research_source source;
                source.source_id = workspace.workspace_id + ":workflow:" + step.title + ":" +
                    std::to_string(index + 1);
                source.title = ref.name.empty() ? step.tool_call->name : ref.name;
                source.origin = step.tool_call->name;
                source.authority = "host-tool-runtime";
                source.kind = step.tool_call->name.rfind("openalex.", 0) == 0
                    ? common_agent_research_source_kind::api_result
                    : step.tool_call->name == "web.fetch"
                        ? common_agent_research_source_kind::web_page
                        : common_agent_research_source_kind::documentation;
                source.resource_ref = ref;
                source.quality_score = step.tool_call->name == "openalex.getWork" ? 0.85 : 0.75;
                // An OpenAlex record reports article metadata and an abstract;
                // it is not the article's full-text primary source.
                source.primary_source = step.tool_call->name != "openalex.getWork";
                source.role = "workflow_evidence";
                auto source_id = source.source_id;
                const auto payload = nlohmann::ordered_json::parse(observation.summary, nullptr, false);
                const auto & record = payload.is_object() && payload.contains("result") && payload["result"].is_object()
                    ? payload["result"] : payload;
                const auto record_id = record.is_object() && record.contains("id") && record["id"].is_string()
                    ? record["id"].get<std::string>() : std::string{};
                const auto candidate = std::find_if(workspace.sources.begin(), workspace.sources.end(), [&](const auto & existing) {
                    return existing.role == "api_discovery_candidate" && !record_id.empty() && existing.origin == record_id;
                });
                if (candidate != workspace.sources.end()) {
                    source_id = candidate->source_id;
                    if (!common_agent_research_materialize_candidate(workspace, *candidate, ref, context.error)) return false;
                    candidate->quality_score = source.quality_score;
                    candidate->primary_source = source.primary_source;
                    candidate->role = source.role;
                } else if (!common_agent_research_add_source(workspace, std::move(source), context.error)) {
                    return false;
                }
                // The API preview is a host-produced view of the persisted
                // record. Keep its abstract separate from a full-text claim;
                // do not mark the objective answered merely by finding it.
                if (step.tool_call->name == "openalex.getWork" && workspace.gaps.size() == 1) {
                    if (payload.is_object()) {
                        const auto & record = payload.contains("result") && payload["result"].is_object()
                            ? payload["result"] : payload;
                        if (record.contains("abstract_text") && record["abstract_text"].is_string() &&
                                !record["abstract_text"].get_ref<const std::string &>().empty()) {
                            common_agent_research_evidence evidence;
                            evidence.evidence_id = source_id + ":abstract";
                            evidence.source_id = source_id;
                            evidence.claim_id = workspace.gaps.front().gap_id;
                            evidence.statement = "OpenAlex abstract preview: " + record["abstract_text"].get<std::string>();
                            if ((record.contains("abstract_truncated") && record["abstract_truncated"] == true) ||
                                    (record.contains("abstract_text_truncated") && record["abstract_text_truncated"] == true) ||
                                    (record.contains("view_truncated") && record["view_truncated"] == true) ||
                                    (payload.contains("result_truncated") && payload["result_truncated"] == true))
                                evidence.statement += " [Preview truncated; full abstract not established.]";
                            evidence.source_location = ref.uri + "#/abstract_inverted_index";
                            evidence.relation = common_agent_research_evidence_relation::contextualizes;
                            evidence.origin = common_agent_research_evidence_origin::normalized_tool_result;
                            evidence.relevance = 0.75;
                            evidence.confidence = 0.75;
                            if (!common_agent_research_record_evidence(workspace, std::move(evidence), context.error)) return false;
                        }
                    }
                }
            }
        }
    }
    return true;
}

} // namespace

void common_agent_apply_research_synthesis_state(common_agent_request & request,
        const common_agent_research_result & result) {
    common_agent_synthesis_state synthesis;
    synthesis.evidence=result.complete ? common_agent_synthesis_evidence::sufficient :
        result.partial_answer_available ? common_agent_synthesis_evidence::partial : common_agent_synthesis_evidence::insufficient;
    synthesis.partial_allowed=result.partial_answer_available;
    synthesis.stop_reason=research_stop_reason_name(result.stop_reason);
    synthesis.evidence_ids=result.critical_evidence_ids;
    request.synthesis_state=std::move(synthesis);
}

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

bool checkpoint_common_agent_research_workspace(
        common_agent_runtime_turn_context & context) {
    if (!context.research_workspace) return true;
    size_t sequence = static_cast<size_t>(context.research_workspace->iterations_completed) + 1;
    if (context.result.research_workspace_checkpoint)
        sequence = std::max(sequence, context.result.research_workspace_checkpoint->sequence + 1);
    std::string checkpoint_error;
    auto checkpoint = make_common_agent_research_workspace_checkpoint(
        *context.research_workspace, sequence, checkpoint_error);
    if (!checkpoint_error.empty()) {
        const auto detail = "research checkpoint creation failed safely: " + checkpoint_error;
        if (context.result.error.empty()) context.result.error = detail;
        else context.result.error += "; " + detail;
        return false;
    }
    context.result.research_workspace_checkpoint = std::move(checkpoint);
    context.emit_trace(common_runtime_trace_stage::research, common_runtime_trace_kind::recorded,
        "research workspace checkpoint created", context.request.plan_id.value_or(""),
        context.research_workspace->workspace_id);
    return true;
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
    if (context.tools->has_execution_envelope() &&
            !context.tools->set_execution_phase(common_agent_execution_phase::research)) {
        context.error = "research phase is not permitted by the route execution envelope";
        context.result.error = context.error;
        return false;
    }
    const std::string transition_source =
        context.request.cognitive_state.mode == common_agent_cognitive_mode::reflect
        ? "reflection" : "host";
    if (!common_agent_runtime_transition_cognitive_mode(
            context, common_agent_cognitive_mode::explore,
            "bounded research evidence acquisition", transition_source,
            common_runtime_trace_stage::research)) {
        context.error = "invalid cognitive-mode transition into exploration";
        context.result.error = context.error;
        return false;
    }

    context.research_workspace.emplace();
    if (!common_agent_research_create_workspace(
            context.request, *context.research_workspace, context.error)) {
        context.result.error = context.error;
        return false;
    }
    if (!seed_research_from_workflow_resources(
            context, *context.research_workspace)) {
        context.result.error = context.error;
        return false;
    }

    context.research_assessor = std::make_unique<common_agent_research_bounded_assessor>(
        [&context](const common_agent_research_gap & gap, const common_agent_research_evidence & evidence) {
            common_agent_research_assessment assessment;
            auto & workspace = *context.research_workspace;
            constexpr int reservation = 192;
            if (!context.request.research_assess_evidence ||
                    (context.request.research_should_stop && context.request.research_should_stop()) ||
                    (workspace.budget.max_tokens > 0 &&
                     workspace.generation_tokens_reserved + reservation > workspace.budget.max_tokens)) {
                assessment.summary = "Semantic assessment was unavailable or its budget was exhausted.";
                return assessment;
            }
            workspace.generation_tokens_reserved += reservation;
            std::string error;
            // Reserve before inference so interruption cannot make the same
            // budget available again after recovery.
            if (context.request.research_progress_sink &&
                    !context.request.research_progress_sink(workspace, error)) {
                assessment.summary = "Could not checkpoint the research assessment budget.";
                return assessment;
            }
            int used = 0;
            if (!context.request.research_assess_evidence(gap, evidence, assessment, used, error)) {
                assessment = {};
                assessment.summary = "Semantic assessment did not produce a valid result: " + error;
            }
            return assessment;
        },
        [&context](const common_agent_research_gap & gap,
                const std::vector<common_agent_research_evidence> & evidence,
                common_agent_research_source_comparison & comparison) {
            auto & workspace = *context.research_workspace;
            constexpr int reservation = 192;
            if (!context.request.research_compare_evidence ||
                    (context.request.research_should_stop && context.request.research_should_stop()) ||
                    (workspace.budget.max_tokens > 0 &&
                     workspace.generation_tokens_reserved + reservation > workspace.budget.max_tokens)) return false;
            workspace.generation_tokens_reserved += reservation;
            std::string error;
            if (context.request.research_progress_sink &&
                    !context.request.research_progress_sink(workspace, error)) return false;
            int used = 0;
            return context.request.research_compare_evidence(gap, evidence, comparison, used, error);
        });
    // When the model selects research operations, resolve current user source
    // constraints before existing evidence can establish coverage. The
    // selector will still receive existing observations during the task.
    if (!context.request.research_select_tool && !common_agent_research_assess_materialized_evidence(
            *context.research_workspace, *context.research_assessor, context.error)) {
        context.result.error = "materialized research evidence assessment failed: " + context.error;
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

    context.research_adapter = std::make_unique<common_agent_research_runtime_adapter>(
        *context.tools, context.research_assessor.get(), context.request.research_should_stop,
        context.request.resource_chunk_page_bytes, context.request.resource_chunk_overlap_bytes,
        context.request.resource_extract_chunk, context.request.research_progress_sink, context.request.resource_plan_chunks,
        context.request.research_select_tool,
        [&context](const std::string & tool, const std::string & outcome) {
            context.emit_trace(common_runtime_trace_stage::research, common_runtime_trace_kind::decided,
                "research tool selection outcome=" + outcome + (tool.empty() ? std::string{} : " tool=" + tool),
                context.request.plan_id.value_or(""), context.research_workspace->workspace_id);
        });
    context.research_runner = std::make_unique<common_agent_research_runner>();

    auto research_result = context.research_runner->run(
        *context.research_workspace,
        *context.research_adapter,
        context.error,
        context.request.research_should_stop,
        context.request.research_stop_reason,
        make_common_agent_research_lifecycle_sink(context),
        context.request.research_progress_sink);
    const bool bounded_stop = research_result.stop_reason == common_agent_research_stop_reason::no_progress ||
        research_result.stop_reason == common_agent_research_stop_reason::budget_exhausted ||
        research_result.stop_reason == common_agent_research_stop_reason::source_access_blocked;
    if (!research_result.complete && context.error.empty() && bounded_stop) {
        research_result.partial_answer_available = std::any_of(research_result.evidence.begin(), research_result.evidence.end(),
            [&](const auto & evidence) {
                if (evidence.origin == common_agent_research_evidence_origin::model_inference || evidence.statement.empty()) return false;
                return std::any_of(research_result.sources.begin(), research_result.sources.end(), [&](const auto & source) {
                    return source.source_id == evidence.source_id && source.resource_ref &&
                        source.kind != common_agent_research_source_kind::memory;
                });
            });
        if (research_result.partial_answer_available) {
            research_result.synthesis_context = "Research is incomplete. Provide only a partial answer grounded in the listed sources. "
                "Explicitly identify unresolved questions and source-access limitations; do not claim unread full text was reviewed.\n" +
                research_result.synthesis_context;
            if (context.outer_plan) {
                for (const auto & handled : context.outer_plan->observations) {
                    if (handled.source != "workflow_handled_failure") continue;
                    for (const auto & id : handled.evidence_ids) {
                        const auto failure = std::find_if(context.outer_plan->observations.begin(), context.outer_plan->observations.end(),
                            [&](const auto & observation) { return observation.id == id; });
                        if (failure == context.outer_plan->observations.end()) continue;
                        const auto payload = nlohmann::ordered_json::parse(failure->summary, nullptr, false);
                        if (!payload.is_object() || !payload.contains("failure") || !payload["failure"].is_object()) continue;
                        const auto & detail = payload["failure"];
                        research_result.synthesis_context += "Source access limitation: " + failure->source;
                        if (detail.contains("http_status") && detail["http_status"].is_number_integer())
                            research_result.synthesis_context += " HTTP " + std::to_string(detail["http_status"].get<int>());
                        research_result.synthesis_context += "; this attempt did not obtain the source.\n";
                    }
                }
            }
        }
    }
    if (!research_result.complete && context.error.empty() && !research_result.no_action_message.empty()) {
        // No evidence is required merely to ask a clarification or answer from
        // already available conversation context. Do not label this complete.
        research_result.partial_answer_available = true;
        research_result.synthesis_context = "Research is incomplete. Use only established context; "
            "if clarification was requested, ask that question and do not invent source findings.\n" +
            research_result.synthesis_context;
    }
    context.result.research_result = research_result;
    common_agent_apply_research_synthesis_state(context.request,research_result);

    // Both complete and partial research must reach daemon persistence.
    if (!checkpoint_common_agent_research_workspace(context)) return false;

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

    if (!context.error.empty() || (!research_result.complete && !research_result.partial_answer_available)) {
        context.emit_event(
            common_agent_event_type::research_incomplete,
            "research stopped before sufficient evidence coverage");
        context.emit_trace(
            common_runtime_trace_stage::research,
            common_runtime_trace_kind::failed,
            "research incomplete",
            {},
            context.research_workspace->workspace_id);
        std::ostringstream failure;
        failure << (context.error.empty()
                ? "research did not meet its evidence coverage requirement"
                : context.error)
                << " [stop_reason=" << research_stop_reason_name(research_result.stop_reason)
                << " coverage=" << research_result.coverage.answered_gaps << "/"
                << research_result.coverage.total_gaps
                << " evidence=" << research_result.evidence.size()
                << " sources=" << research_result.sources.size()
                << " iterations=" << context.research_workspace->iterations_completed
                << " tool_calls=" << context.research_workspace->tool_calls << "]";
        context.result.error = failure.str();
        return false;
    }

    if (research_result.partial_answer_available) {
        const bool no_action = !research_result.no_action_message.empty();
        context.emit_event(common_agent_event_type::research_incomplete,
            no_action ? "research stopped without another tool action" :
                "research stopped with bounded source-backed partial evidence");
        context.emit_trace(common_runtime_trace_stage::research, common_runtime_trace_kind::decided,
            no_action ? "research incomplete; clarification or context-only response allowed" :
                "research incomplete; partial synthesis allowed with explicit limitations",
            context.request.plan_id.value_or(""), context.research_workspace->workspace_id);
        context.research_synthesis_context = research_result.synthesis_context;
        if (!bridge_research_result_to_plan(context, research_result)) {
            context.result.error = context.error;
            return false;
        }
        return common_agent_runtime_transition_cognitive_mode(context, common_agent_cognitive_mode::frame,
            "bounded partial research evidence is ready for synthesis", "host", common_runtime_trace_stage::research);
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
    if (!common_agent_runtime_transition_cognitive_mode(
            context, common_agent_cognitive_mode::frame,
            "research evidence met the bounded completion criteria", "host",
            common_runtime_trace_stage::research)) {
        context.error = "invalid cognitive-mode transition after research";
        context.result.error = context.error;
        return false;
    }
    return true;
}
