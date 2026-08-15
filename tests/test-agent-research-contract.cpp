#include "agent/research/research-contract.h"
#include "agent/research/research-controller.h"
#include "agent/research/research-workspace-factory.h"
#include "agent/research/research-workspace.h"
#include "agent/research/research-assessor.h"
#include "agent/input-resources.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>

int main() {
    common_agent_request factory_request;
    factory_request.prompt = "Research the bounded runtime contract";
    factory_request.turn_id = "factory-turn";
    factory_request.session_id = "factory-session";
    factory_request.namespace_id = "factory-namespace";
    factory_request.max_iterations = 3;
    factory_request.max_tool_batches = 2;
    factory_request.deliberation_policy = make_common_agent_deliberation_policy(common_agent_thinking_mode::research);
    common_agent_input_resource input_resource;
    input_resource.resource.uri = "agent-resource://factory-turn/upload-1";
    input_resource.resource.name = "notes.txt";
    input_resource.resource.mime_type = "text/plain";
    input_resource.role = "reference";
    input_resource.required = true;
    factory_request.input_resources.push_back(input_resource);
    common_memory_hit memory_hit;
    memory_hit.memory.id = "memory-procedure-1";
    memory_hit.memory.kind = common_memory_kind::procedure;
    memory_hit.memory.summary = "Use the bounded runtime contract first.";
    memory_hit.memory.confidence = 0.9f;
    memory_hit.final_score = 0.9f;
    factory_request.memories.push_back(memory_hit);
    common_agent_research_workspace factory_workspace;
    std::string factory_error;
    assert(common_agent_research_create_workspace(factory_request, factory_workspace, factory_error));
    assert(factory_workspace.workspace_id == "research:factory-turn");
    assert(factory_workspace.gaps.size() == 1);
    assert(factory_workspace.budget.max_iterations == 4);
    assert(factory_workspace.budget.max_tool_calls == 2);
    assert(factory_workspace.sources.size() == 2);
    assert(factory_workspace.sources.front().kind == common_agent_research_source_kind::user_supplied);
    assert(factory_workspace.sources.front().resource_ref &&
        factory_workspace.sources.front().resource_ref->uri == input_resource.resource.uri);
    assert(factory_workspace.sources.front().role == "reference" &&
        factory_workspace.sources.front().required &&
        !factory_workspace.sources.front().primary_source &&
        factory_workspace.sources.front().content_hash.empty());
    assert(factory_workspace.sources.back().kind == common_agent_research_source_kind::memory);
    assert(factory_workspace.sources.back().memory_id == memory_hit.memory.id);
    const auto input_context = common_agent_render_input_resource_context(factory_request.input_resources);
    assert(input_context.find("id=r1") != std::string::npos);
    assert(input_context.find("agent-resource://factory-turn/upload-1") == std::string::npos);
    common_agent_research_controller factory_controller;
    const auto input_action = factory_controller.begin(factory_workspace, factory_error);
    assert(input_action.kind == common_agent_research_action_kind::schedule_task);
    assert(input_action.preferred_tools.size() == 1 && input_action.preferred_tools.front() == "resource_read");
    assert(input_action.instruction == input_resource.resource.uri);

    factory_workspace.gaps.front().status = common_agent_research_gap_status::open;
    factory_workspace.tasks.clear();
    factory_workspace.evidence.clear();
    factory_workspace.coverage = {};
    factory_workspace.sources.front().resource_ref.reset();
    const auto memory_action = factory_controller.begin(factory_workspace, factory_error);
    assert(memory_action.kind == common_agent_research_action_kind::schedule_task);
    assert(memory_action.preferred_tools.size() == 1 && memory_action.preferred_tools.front() == "memory_get");
    assert(memory_action.instruction == memory_hit.memory.id);

    common_agent_research_workspace workspace;
    workspace.workspace_id = "research-1";
    workspace.request_id = "request-1";
    workspace.turn_id = "turn-1";
    workspace.session_id = "session-1";
    workspace.plan_id = "plan-1";
    workspace.plan_step_id = "research-step";
    workspace.scope.namespace_id = "project";
    workspace.scope.session_id = "session-1";
    workspace.objective.objective_id = "objective-1";
    workspace.objective.question = "What scheduling gaps remain?";
    workspace.objective.success_criteria = {"claims are source-backed"};
    std::string error;
    assert(common_agent_research_add_gap(workspace, {"gap-1", "Is cancellation bounded?", "The runtime contract is unclear.", "evidence establishes whether cancellation is bounded", 1}, error));
    assert(common_agent_research_add_task(workspace, {"task-1", "gap-1", "Find cancellation paths", common_agent_research_task_kind::repository_inspection, {"repository.search"}, {}, 1, 0, 1}, error));
    common_runtime_resource_ref resource;
    resource.uri = "agent-resource://turn-1/source-1";
    resource.scope = common_runtime_resource_scope::turn;
    assert(common_agent_research_add_source(workspace, {"source-1", "runtime-operation.h", "repository", "repository", "now", "hash-1", common_agent_research_source_kind::repository_file, resource, 0.95, true}, error));
    common_agent_research_evidence evidence;
    evidence.evidence_id = "evidence-1";
    evidence.source_id = "source-1";
    evidence.claim_id = "claim-1";
    evidence.statement = "Cancellation is represented by a bounded operation state.";
    evidence.source_location = "runtime-operation.h:42";
    evidence.relation = common_agent_research_evidence_relation::supports;
    evidence.origin = common_agent_research_evidence_origin::direct_source;
    evidence.relevance = 0.95;
    evidence.confidence = 0.9;
    evidence.directly_observed = true;
    assert(common_agent_research_record_evidence(workspace, evidence, error));
    assert(common_agent_research_update_coverage(workspace, {1, 1, 0, 0, 1.0, 0.9, 1.0}, error));
    assert(common_agent_research_workspace_validate(workspace, error));
    assert(!common_agent_research_record_evidence(workspace, workspace.evidence.front(), error));
    assert(error.find("already exists") != std::string::npos);
    workspace.evidence.push_back(workspace.evidence.front());
    assert(!common_agent_research_workspace_validate(workspace, error));
    assert(error.find("duplicate evidence") != std::string::npos);

    common_agent_research_controller controller;
    workspace.evidence.pop_back();
    workspace.gaps.front().status = common_agent_research_gap_status::open;
    workspace.tasks.clear();
    workspace.coverage = {};
    const auto first_action = controller.begin(workspace, error);
    assert(first_action.kind == common_agent_research_action_kind::schedule_task);
    const auto final_action = controller.advance(workspace, {
        common_agent_research_event_type::task_completed,
        first_action.task_id,
        first_action.gap_id,
        1,
        {"evidence-1"},
        true,
        0.9}, error);
    assert(final_action.kind == common_agent_research_action_kind::complete);
    assert(final_action.stop_reason == common_agent_research_stop_reason::sufficient_coverage);
    const auto research_result = controller.finalize(workspace);
    assert(research_result.complete && research_result.workspace_id == "research-1");
    assert(research_result.established_claim_ids.size() == 1);
    assert(research_result.established_claim_ids.front() == "gap-1");
    assert(research_result.sources.size() == 1 && research_result.evidence.size() == 1);

    auto assessment_workspace = workspace;
    assessment_workspace.gaps.front().status = common_agent_research_gap_status::open;
    assessment_workspace.gaps.front().evidence_ids.clear();
    assessment_workspace.tasks.clear();
    assessment_workspace.coverage = {};
    const auto assessment_start = controller.begin(assessment_workspace, error);
    assert(assessment_start.kind == common_agent_research_action_kind::schedule_task);
    const auto assessment_action = controller.advance(assessment_workspace, {
        common_agent_research_event_type::task_completed,
        assessment_start.task_id,
        assessment_start.gap_id,
        1,
        {"evidence-1"},
        true,
        0.9,
        "tool completed but criterion was not assessed",
        {common_agent_research_assessment_status::insufficient, 0.2,
            "evidence is not sufficient for the completion criterion"}}, error);
    assert(assessment_action.kind == common_agent_research_action_kind::schedule_task);
    assert(assessment_workspace.gaps.front().status == common_agent_research_gap_status::investigating);

    common_agent_research_bounded_assessor assessor;
    const auto sufficient_assessment = assessor.assess(
        {"gap", "", "", "repository result", 1},
        {"evidence", "source", "gap", "repository search returned a match", "repository", common_agent_research_evidence_relation::supports,
            common_agent_research_evidence_origin::normalized_tool_result, 0.8, 0.9});
    assert(sufficient_assessment.status == common_agent_research_assessment_status::sufficient);
    const auto insufficient_assessment = assessor.assess(
        {"gap", "", "", "web result", 1},
        {"evidence-2", "source", "gap", "repository search returned a match", "repository", common_agent_research_evidence_relation::supports,
            common_agent_research_evidence_origin::normalized_tool_result, 0.8, 0.9});
    assert(insufficient_assessment.status == common_agent_research_assessment_status::insufficient);

    auto criteria_request = factory_request;
    criteria_request.input_resources.clear();
    criteria_request.memories.clear();
    criteria_request.objective = common_agent_objective{};
    criteria_request.objective->success_criteria = {"verify source", "record limitation"};
    common_agent_research_workspace criteria_workspace;
    assert(common_agent_research_create_workspace(criteria_request, criteria_workspace, factory_error));
    assert(criteria_workspace.gaps.size() == 2);
    assert(criteria_workspace.gaps[0].completion_criterion == "verify source");
    assert(criteria_workspace.gaps[1].completion_criterion == "record limitation");

    const auto cancelled_action = controller.advance(workspace, {
        common_agent_research_event_type::cancelled,
        {}, {}, 0, {}}, error);
    assert(cancelled_action.kind == common_agent_research_action_kind::complete);
    assert(cancelled_action.stop_reason == common_agent_research_stop_reason::cancelled);

    auto budget_limited_workspace = workspace;
    budget_limited_workspace.budget.max_iterations = 0;
    const auto budget_action = controller.begin(budget_limited_workspace, error);
    assert(budget_action.kind == common_agent_research_action_kind::complete);
    assert(budget_action.stop_reason == common_agent_research_stop_reason::budget_exhausted);

    auto retry_workspace = workspace;
    retry_workspace.gaps.front().status = common_agent_research_gap_status::open;
    retry_workspace.tasks.clear();
    retry_workspace.evidence.clear();
    retry_workspace.coverage = {};
    retry_workspace.budget.max_iterations = 2;
    const auto retry_start = controller.begin(retry_workspace, error);
    assert(retry_start.kind == common_agent_research_action_kind::schedule_task);
    const auto retry_action = controller.advance(retry_workspace, {
        common_agent_research_event_type::task_failed,
        retry_start.task_id,
        retry_start.gap_id,
        0,
        {}}, error);
    assert(retry_action.kind == common_agent_research_action_kind::schedule_task);
    assert(retry_action.task_id != retry_start.task_id);
    assert(retry_action.instruction != retry_start.instruction);
    assert(retry_action.preferred_tools != retry_start.preferred_tools);

    auto confidence_workspace = workspace;
    confidence_workspace.gaps.front().status = common_agent_research_gap_status::open;
    confidence_workspace.gaps.front().evidence_ids.clear();
    confidence_workspace.budget.minimum_confidence = 0.95;
    confidence_workspace.coverage = {};
    const auto confidence_start = controller.begin(confidence_workspace, error);
    assert(confidence_start.kind == common_agent_research_action_kind::schedule_task);
    const auto confidence_action = controller.advance(confidence_workspace, {
        common_agent_research_event_type::task_completed,
        confidence_start.task_id,
        confidence_start.gap_id,
        1,
        {"evidence-1"}}, error);
    assert(confidence_action.kind == common_agent_research_action_kind::schedule_task);

    auto insufficient_workspace = workspace;
    insufficient_workspace.gaps.front().status = common_agent_research_gap_status::open;
    insufficient_workspace.gaps.front().evidence_ids.clear();
    insufficient_workspace.coverage = {};
    const auto insufficient_start = controller.begin(insufficient_workspace, error);
    assert(insufficient_start.kind == common_agent_research_action_kind::schedule_task);
    const auto insufficient_action = controller.advance(insufficient_workspace, {
        common_agent_research_event_type::task_completed,
        insufficient_start.task_id,
        insufficient_start.gap_id,
        1,
        {"evidence-1"},
        false,
        0.9}, error);
    assert(insufficient_action.kind == common_agent_research_action_kind::schedule_task);
    assert(insufficient_workspace.gaps.front().status != common_agent_research_gap_status::sufficiently_answered);
    return 0;
}
