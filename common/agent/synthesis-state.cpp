#include "synthesis-state.h"
#include "plan/plan-workflow-state.h"
#include "hash/hash.h"
#include <algorithm>

using json = nlohmann::ordered_json;
namespace {
json items_json(const std::vector<common_agent_review_item> & items) {
    auto out = json::array();
    for (const auto & item : items) out.push_back({{"id",item.id},{"description",item.description},
        {"correction",item.correction},{"evidence_ids",item.evidence_ids}});
    return out;
}
}
std::string common_agent_synthesis_evidence_fingerprint(const common_plan_state & plan) {
    auto evidence=json::array();
    for(const auto & observation:plan.observations) {
        if(observation.source=="response_review" || observation.source=="reflection_hint" ||
            observation.source=="workflow_continuation_resolution" || observation.source=="unverified_draft") continue;
        evidence.push_back({{"id",observation.id},{"source",observation.source},{"summary",observation.summary}});
    }
    const auto text=evidence.dump(); return hash_sha256_hex(text.data(),text.size());
}
std::string common_agent_workflow_evidence_fingerprint(const common_plan_state & plan) {
    auto projected=plan;
    projected.observations.erase(std::remove_if(projected.observations.begin(),projected.observations.end(),[](const auto & item) {
        return item.id.rfind("tool:",0)!=0 && item.source.rfind("resource_chunk",0)!=0 &&
            item.source!="workflow_terminal_choice" && item.source!="workflow_handled_failure" &&
            item.source!="workflow_handoff" && item.source!="workflow_arguments" && item.source!="workflow_verification";
    }),projected.observations.end());
    return common_agent_synthesis_evidence_fingerprint(projected);
}
common_agent_synthesis_state common_agent_resolve_synthesis_state(const common_plan_state & plan,
        std::optional<common_agent_synthesis_state> research) {
    auto state = research.value_or(common_agent_synthesis_state{});
    state.plan_id=plan.id;state.plan_version=plan.version;
    state.evidence_fingerprint=common_agent_synthesis_evidence_fingerprint(plan);
    state.mandatory_remaining = std::any_of(plan.steps.begin(), plan.steps.end(), [&](const auto & step) {
        return !step.optional && common_plan_step_effective_mode(step) != common_plan_step_mode::final_response &&
            step.status != common_plan_step_status::completed && step.status != common_plan_step_status::skipped &&
            !common_plan_workflow_failure_handled(plan,step);
    });
    if (plan.workflow_definition && !plan.workflow_definition->transitions.empty()) {
        const bool terminal = common_plan_workflow_is_terminal(plan,*plan.workflow_definition);
        state.mandatory_remaining = state.mandatory_remaining || !terminal;
        if (terminal && !plan.workflow_definition->continuation_links.empty()) {
            bool resolved = false;
            const auto workflow_fingerprint=common_agent_workflow_evidence_fingerprint(plan);
            for (auto item = plan.observations.rbegin(); item != plan.observations.rend(); ++item) {
                if (item->source != "workflow_continuation_resolution") continue;
                const auto value = json::parse(item->summary,nullptr,false);
                if (!value.is_object() || value.value("evidence_fingerprint",json{}) != workflow_fingerprint ||
                    value.value("workflow",json{}) != plan.workflow_definition->workflow_ref ||
                    value.value("revision",json{}) != plan.workflow_definition->workflow_revision ||
                    value.value("segment",json{}) != (plan.route_binding ? plan.route_binding->workflow_step_start_index : 0) ||
                    value.value("terminal_state",json{}) != common_plan_workflow_completed_state(plan,*plan.workflow_definition)) continue;
                state.continuation_reason = value.value("reason",std::string{});
                resolved = value.value("outcome",json{}) == "not_applicable";
                if(std::find(state.evidence_ids.begin(),state.evidence_ids.end(),item->id)==state.evidence_ids.end())
                    state.evidence_ids.push_back(item->id);
                break;
            }
            state.mandatory_remaining = state.mandatory_remaining || !resolved;
        }
    }
    if (state.mandatory_remaining) { state.work=common_agent_synthesis_work::pending; state.partial_allowed=false; }
    else if (research && research->partial_allowed) state.work=common_agent_synthesis_work::bounded;
    else state.work=common_agent_synthesis_work::complete;
    // Without research, a declared unavailable terminal can still finish as
    // a bounded answer. This is execution disposition, never a coverage score.
    if(!research && !state.mandatory_remaining && plan.workflow_definition &&
            !state.continuation_reason.empty()) {
        const auto terminal=common_plan_workflow_completed_state(plan,*plan.workflow_definition);
        if(std::any_of(plan.workflow_definition->transitions.begin(),plan.workflow_definition->transitions.end(),
            [&](const auto & transition){return !transition.failure_terminal_state.empty() && transition.failure_terminal_state==terminal;})) {
            state.work=common_agent_synthesis_work::bounded;state.partial_allowed=true;
        }
    }
    return state;
}
json common_agent_synthesis_state_json(const common_agent_synthesis_state & state) {
    const char * work = state.work==common_agent_synthesis_work::pending ? "pending" :
        state.work==common_agent_synthesis_work::bounded ? "bounded_complete" : "complete";
    const char * evidence = state.evidence==common_agent_synthesis_evidence::sufficient ? "sufficient" :
        state.evidence==common_agent_synthesis_evidence::partial ? "partial" :
        state.evidence==common_agent_synthesis_evidence::insufficient ? "insufficient" : "unknown";
    return {{"work",work},{"evidence",evidence},{"mandatory_remaining",state.mandatory_remaining},
        {"partial_allowed",state.partial_allowed},{"stop_reason",state.stop_reason},{"continuation_reason",state.continuation_reason},
        {"evidence_ids",state.evidence_ids},{"plan_id",state.plan_id},{"plan_version",state.plan_version},
        {"evidence_fingerprint",state.evidence_fingerprint}};
}
std::string common_agent_render_synthesis_state(const common_agent_synthesis_state & state,const std::string & phase) {
    auto value=common_agent_synthesis_state_json(state); value["phase"]=phase;
    return "\n[Host execution and answer contract]\n"+value.dump()+
        "\nWork completion and evidence completeness are separate. When partial_allowed is true and mandatory_remaining is false, "
        "a correctly qualified partial report is acceptable; unavailable evidence alone is not a blocking defect. "
        "Reject unsupported claims, false access claims or omitted material limitations. Advisory wording preferences do not require revision.\n";
}
json common_agent_review_checklist_json(const common_agent_review_checklist & c) {
    return {{"round",c.round},{"draft_fingerprint",c.draft_fingerprint},{"evidence_fingerprint",c.evidence_fingerprint},
        {"blocking",items_json(c.blocking)},{"advisory",c.advisory}};
}
bool common_agent_review_checklist_from_json(const json & value,common_agent_review_checklist & out) {
    try {
        common_agent_review_checklist next;
        next.round=value.at("round").get<size_t>();
        next.draft_fingerprint=value.at("draft_fingerprint").get<std::string>();
        next.evidence_fingerprint=value.at("evidence_fingerprint").get<std::string>();
        const auto & items=value.at("blocking"); if (!items.is_array() || items.size()>4) return false;
        for (const auto & item:items) {
            common_agent_review_item issue{item.at("id").get<std::string>(),
                item.at("description").get<std::string>(),item.at("correction").get<std::string>(),
                item.at("evidence_ids").get<std::vector<std::string>>()};
            if(issue.id.empty() || issue.id.size()>64 || issue.description.size()>512 || issue.correction.size()>512 ||
                issue.evidence_ids.size()>8 || std::any_of(issue.evidence_ids.begin(),issue.evidence_ids.end(),
                    [](const auto & id){return id.empty() || id.size()>1024;}) ||
                std::any_of(next.blocking.begin(),next.blocking.end(),[&](const auto & prior){return prior.id==issue.id;})) return false;
            next.blocking.push_back(std::move(issue));
        }
        next.advisory=value.value("advisory",std::vector<std::string>{}); if(next.advisory.size()>4)return false;
        if(std::any_of(next.advisory.begin(),next.advisory.end(),[](const auto & text){return text.size()>512;})) return false;
        out=std::move(next); return true;
    } catch(const json::exception &) {return false;}
}
