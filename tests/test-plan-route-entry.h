#include "agent/method-phases.h"
#include "agent/synthesis-state.h"
#include <stdexcept>
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

inline void write_method_progress(common_plan_store & store) {
    common_plan_state plan;plan.id="method-progress-roundtrip";plan.source_revision="method@v2";
    plan.phase_review_required=true;plan.supersedes_blueprints={"method-v1"};
    common_plan_step step;step.id="lookup";step.title="Lookup";plan.steps={step};
    plan.observations.push_back({"method-outline","blueprint_outline",
        R"([{"id":"orient","depends_on":[]},{"id":"knowledge","depends_on":["orient"]}])",0.0f});
    plan.observations.push_back({"method-review","method_phase_review",
        R"({"schema":"method-phase-review.v1","method_revision":"method@v2","phases":{"orient":{"status":"satisfied","reason":"Inspected local source","evidence_ids":[]}}})",0.0f});
    std::string error;require_route_entry(store.create(plan,error),error);
}
inline void verify_method_progress(common_plan_store & store) {
    std::string error;const auto plan=store.get("method-progress-roundtrip",error);
    require_route_entry(plan && plan->phase_review_required && plan->source_revision=="method@v2" &&
        plan->supersedes_blueprints==std::vector<std::string>{"method-v1"} && plan->observations.size()==2 &&
        plan->observations.back().source=="method_phase_review" &&
        plan->observations.back().summary.find("Inspected local source")!=std::string::npos,
        "method metadata or phase progress did not survive store restart");
}

// Checkpoint references and inherited phase decisions use ordinary plan codecs.
inline void write_prior_work_lineage(common_plan_store & store) {
    std::string error;
    common_plan_state source;source.id="prior-work-source";source.namespace_id="local";
    source.session_id="lineage-session";source.project_id="lineage-project";
    source.status=common_plan_status::active;
    source.observations.push_back({"prior-source-read","repository.read",R"({"ok":true,"result":{"path":"header.h","sha256":"sha256:known"}})",1.0f});
    if (!store.create(source,error)) throw std::runtime_error(error);
    source=*store.get(source.id,error);
    const nlohmann::ordered_json reference={{"source_plan_id",source.id},{"source_plan_version",source.version},{"observation_id","prior-source-read"}};
    common_plan_state child;child.id="prior-work-child";child.namespace_id=source.namespace_id;
    child.session_id=source.session_id;child.project_id=source.project_id;child.status=common_plan_status::active;
    child.observations.push_back({"prior-work-link","prior_work_checkpoint",
        nlohmann::ordered_json{{"schema","prior-work.v2"},{"source_plan_id",source.id},{"source_plan_version",source.version},
            {"evidence_fingerprint",common_agent_synthesis_evidence_fingerprint(source)}}.dump(),1.0f});
    child.observations.push_back({"inherited-phases","method_phase_review",
        nlohmann::ordered_json{{"schema","method-phase-review.v1"},{"method_revision","v1"},
            {"phases",{{"inspect",{{"status","satisfied"},{"reason","Inspected source"},
                {"evidence_ids",{"prior-source-read"}},{"evidence_refs",nlohmann::ordered_json::array({reference})}}}}}}.dump(),0.0f});
    if (!store.create(child,error)) throw std::runtime_error(error);
}

inline void verify_prior_work_lineage(common_plan_store & store) {
    std::string error;
    const auto child=store.get("prior-work-child",error),source=store.get("prior-work-source",error);
    if (!child || !source) throw std::runtime_error("prior-work plans lost on restart: "+error);
    const auto checkpoint=nlohmann::ordered_json::parse(child->observations.front().summary);
    const auto progress=nlohmann::ordered_json::parse(child->observations.back().summary);
    const auto reference=progress["phases"]["inspect"]["evidence_refs"].front();
    const auto resolved=common_agent_resolve_prior_observation(store,*child,checkpoint,reference,error);
    if (!resolved || resolved->id!="prior-source-read") throw std::runtime_error("prior-work evidence lost on restart: "+error);
    common_plan_state grandchild=*child;grandchild.id="prior-work-grandchild";grandchild.observations.clear();
    const nlohmann::ordered_json next={{"schema","prior-work.v2"},{"source_plan_id",child->id},
        {"source_plan_version",child->version},{"evidence_fingerprint",common_agent_synthesis_evidence_fingerprint(*child)}};
    if (!common_agent_resolve_prior_observation(store,grandchild,next,reference,error))
        throw std::runtime_error("second continuation lost ancestor evidence: "+error);
}
