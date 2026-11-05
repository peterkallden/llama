#pragma once

#include "plan/plan-store.h"
#include "plan/plan-scheduler.h"
#include <stdexcept>

inline void require_route_entry(bool condition, const std::string & message) {
    if (!condition) throw std::runtime_error(message);
}

inline void write_atomic_route_entry(common_plan_store & store) {
    std::string error;
    common_plan_state plan;
    plan.id = "atomic-route-entry";
    plan.status = common_plan_status::active;
    common_plan_step source;
    source.id = "source"; source.title = "Source";
    source.status = common_plan_step_status::completed;
    source.tool_call = common_plan_tool_call{"catalog.search", "{}"};
    common_plan_step answer;
    answer.id = "answer"; answer.title = "Answer";
    answer.mode = common_plan_step_mode::final_response;
    answer.status = common_plan_step_status::active;
    answer.depends_on = {source.id};
    plan.steps = {source, answer}; plan.active_step_id = answer.id;
    common_plan_observation observed;
    observed.id = "tool:source:catalog.search"; observed.source = "catalog.search";
    observed.summary = R"({"result":{"items":[{"id":"W1"},{"id":"W2"}]}})";
    plan.observations.push_back(observed);
    require_route_entry(store.create(plan, error), error);

    common_plan_operation op;
    op.kind = common_plan_operation_kind::request_route_transition;
    op.plan_id = plan.id; op.expected_version = plan.version; op.host_authorized = true;
    common_plan_workflow_definition workflow;
    workflow.workflow_ref = "workflow://test/evidence";
    workflow.workflow_revision = "v1"; workflow.graph_revision = "graph-v1";
    workflow.allowed_tools = {"catalog.get"};
    op.workflow_definition = workflow;
    op.route_binding = common_plan_route_binding{"route:evidence", "", "", workflow.workflow_ref,
        workflow.workflow_revision, workflow.graph_revision, "envelope", "policy"};
    op.route_binding->continuation_source_fingerprint = "fingerprint";
    op.workflow_continuation_record = common_plan_workflow_continuation_record{
        "search@done->workflow://test/evidence@v1", "fingerprint", "workflow://test/search", "v1", "done",
        workflow.workflow_ref, workflow.workflow_revision, observed.id};
    common_plan_step entry;
    entry.id = "entry"; entry.title = "Get W2";
    entry.mode = common_plan_step_mode::tool;
    entry.selected_tool = "catalog.get";
    entry.tool_call = common_plan_tool_call{"catalog.get", R"({"id":"W2"})"};
    entry.depends_on = {source.id};
    op.route_steps = {entry};
    common_plan_observation handoff;
    handoff.id = "workflow_handoff:fingerprint:workflow://test/evidence";
    handoff.source = "workflow_handoff";
    handoff.summary = R"({"source_fingerprint":"fingerprint","target_revision":"v1","value":{"items":[{"id":"W1"},{"id":"W2"}]}})";
    handoff.evidence_ids = {observed.id};
    op.route_handoff = handoff;

    auto invalid = op;
    auto last = entry; last.id = "invalid"; last.depends_on = {"absent"};
    invalid.route_steps.push_back(last);
    common_plan_state updated;
    require_route_entry(!store.apply(invalid, updated, error), "invalid atomic route entry was accepted");
    auto unchanged = store.get(plan.id, error);
    require_route_entry(unchanged && unchanged->version == plan.version && unchanged->steps.size() == 2 &&
        !unchanged->route_binding && unchanged->workflow_continuation_history.empty() && unchanged->observations.size() == 1,
        "rejected route entry changed persisted state");
    invalid = op;
    invalid.route_handoff->summary = R"({"source_fingerprint":42,"target_revision":"v1","value":{}})";
    require_route_entry(!store.apply(invalid, updated, error), "malformed handoff metadata was accepted");
    invalid = op;
    invalid.route_handoff->evidence_ids = {"unrelated-observation"};
    require_route_entry(!store.apply(invalid, updated, error), "unrelated handoff evidence was accepted");
    require_route_entry(store.apply(op, updated, error), error);
    require_route_entry(updated.version == plan.version + 1, "route entry used multiple plan operations");
    require_route_entry(!store.apply(op, updated, error), "stale route entry replay was accepted");
}

inline void verify_atomic_route_entry(common_plan_store & store) {
    std::string error;
    const auto plan = store.get("atomic-route-entry", error);
    require_route_entry(plan && plan->steps.size() == 3 && plan->route_binding &&
        plan->route_binding->workflow_step_start_index == 2 && plan->observations.size() == 2 &&
        plan->workflow_continuation_history.size() == 1 &&
        plan->workflow_continuation_history.front().source_observation_id == "tool:source:catalog.search",
        "atomic route entry did not survive store reopen");
    require_route_entry(plan->steps[1].status == common_plan_step_status::pending && !plan->active_step_id &&
        plan->steps[1].depends_on.back() == "entry" && !common_plan_step_dependencies_ready(*plan, plan->steps[1]),
        "final answer bypassed the committed route entry");
    require_route_entry(plan->steps[2].tool_call && plan->steps[2].tool_call->arguments_json == R"({"id":"W2"})" &&
        common_plan_step_dependencies_ready(*plan, plan->steps[2]), "committed candidate identity or readiness was lost");
}
