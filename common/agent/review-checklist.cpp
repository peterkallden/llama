#include "review-checklist.h"
#include "continuation-control.h"
#include "method-phases.h"
#include "agent/runtime/agent-reflection-engine.h"
#include "plan/plan-context.h"
#include "plan/plan-workflow-state.h"
#include "hash/hash.h"
#include <algorithm>
#include <set>
#include <limits>

namespace {
bool validate_review_part(const common_plan_state & plan,common_agent_review_part & part,std::string & error) {
    const auto observed=std::find_if(plan.observations.begin(),plan.observations.end(),
        [&](const auto & item){return item.id==part.observation_id;});
    if(observed==plan.observations.end() || !common_plan_workflow_observation_is_usable(*observed)) {
        error="excerpt review requires a successful saved read";return false;
    }
    const auto * read=common_plan_find_tool_observation_step(plan,*observed);
    if(!read || !read->tool_call || read->status!=common_plan_step_status::completed || read->tool_call->name!=observed->source) {
        error="excerpt review must refer to a completed tool read";return false;
    }
    const auto payload=nlohmann::ordered_json::parse(observed->summary,nullptr,false);
    const auto body=payload.is_object()?payload.value("result",payload):nlohmann::ordered_json{};
    const auto content=body.is_object()?body.value("content",nlohmann::ordered_json{}):nlohmann::ordered_json{};
    if (!body.is_object() || !body.contains("content") || !body["content"].is_string() ||
            !body.contains("resource") || !body["resource"].is_object()) {
        error="excerpt parts require resource content with a host resource identity; omit parts for repository/workspace reads and assess them through phase_reviews or revision_guidance";
        return false;
    }
    if(!content.is_string() || part.excerpt.empty() || part.excerpt.size()>512 ||
        content.get_ref<const std::string &>().find(part.excerpt)==std::string::npos ||
        part.reason.empty() || part.reason.size()>512 || part.evidence_ids.size()>8 ||
        (part.status!="reviewed" && part.status!="requires_work" && part.status!="unreviewed")) {
        error="excerpt review must identify exact inspected text and a bounded verdict";return false;
    }
    const auto resource=body.value("resource",nlohmann::ordered_json::object());
    part.resource_uri=resource.is_object() && resource.contains("uri") && resource["uri"].is_string()
        ? resource["uri"].get<std::string>() : std::string{};
    if(part.resource_uri.empty()) {error="excerpt review lacks a host resource identity";return false;}
    if(part.status=="reviewed" && part.evidence_ids.empty()) {error="reviewed excerpt requires evidence";return false;}
    for(const auto & id:part.evidence_ids) if(std::none_of(plan.observations.begin(),plan.observations.end(),
        [&](const auto & item){return item.id==id && common_plan_workflow_observation_is_usable(item);})) {
        error="excerpt review cites unknown or failed evidence";return false;
    }
    const auto & text=content.get_ref<const std::string &>();
    part.content_fingerprint=hash_sha256_hex(text.data(),text.size());
    auto evidence=nlohmann::ordered_json::array({{{"id",observed->id},{"summary",observed->summary}}});
    auto ids=part.evidence_ids;std::sort(ids.begin(),ids.end());ids.erase(std::unique(ids.begin(),ids.end()),ids.end());
    for(const auto & id:ids) for(const auto & item:plan.observations) if(item.id==id) {
        evidence.push_back({{"id",id},{"summary",item.summary}});break;
    }
    const auto serialized=evidence.dump();
    part.evidence_fingerprint=hash_sha256_hex(serialized.data(),serialized.size());
    return true;
}
}

bool common_agent_review_step_allows_retry(const common_plan_state & plan, const std::string & step_id) {
    for (auto it = plan.observations.rbegin(); it != plan.observations.rend(); ++it) {
        const auto * observed = common_plan_find_tool_observation_step(plan, *it);
        if (!observed || observed->id != step_id) continue;
        const auto value = nlohmann::ordered_json::parse(it->summary, nullptr, false);
        if (!value.is_object() || !value.contains("failure") || !value["failure"].is_object()) continue;
        const auto retryable = value["failure"].find("retryable");
        // Retain old saved records without a retryability declaration. New
        // host-declared failures share the bounded unchanged-retry contract.
        return retryable == value["failure"].end() || common_agent_continuation_allows_same_call(plan, step_id);
    }
    return true;
}

bool common_agent_advance_review_checklist(const common_agent_review_checklist & previous,
        const common_reflection_result & review,const common_plan_state & plan,const std::string & draft,
        common_agent_review_checklist & next,std::string & error) {
    error.clear();
    if(previous.round==std::numeric_limits<size_t>::max()) {error="invalid saved review round";return false;}
    common_agent_review_checklist candidate;
    candidate.round=previous.round+1;
    candidate.draft_fingerprint=hash_sha256_hex(draft.data(),draft.size());
    candidate.evidence_fingerprint=common_agent_synthesis_evidence_fingerprint(plan);
    candidate.advisory=review.advisory_notes;
    if(review.parts.size()>8) {error="excerpt review exceeds eight parts";return false;}
    for(auto part:review.parts) {
        if(!validate_review_part(plan,part,error)) return false;
        if(std::any_of(candidate.parts.begin(),candidate.parts.end(),[&](const auto & prior){
                return prior.resource_uri==part.resource_uri && prior.excerpt==part.excerpt;})) {
            error="duplicate reviewed excerpt";return false;
        }
        candidate.parts.push_back(std::move(part));
    }
    // A different chat draft does not invalidate an unchanged report excerpt.
    // Keep only assessments whose exact read and supporting records still match.
    for(const auto & prior:previous.parts) {
        if(candidate.parts.size()==8) break;
        if(std::any_of(candidate.parts.begin(),candidate.parts.end(),[&](const auto & current){
                return current.resource_uri==prior.resource_uri && current.excerpt==prior.excerpt;})) continue;
        auto validated=prior;std::string ignored;
        if(validate_review_part(plan,validated,ignored) && !prior.evidence_fingerprint.empty() &&
                validated.resource_uri==prior.resource_uri && validated.content_fingerprint==prior.content_fingerprint &&
                validated.evidence_fingerprint==prior.evidence_fingerprint) candidate.parts.push_back(std::move(validated));
    }
    std::set<std::string> accounted;
    const auto known=[&](const std::string & id) {
        return std::any_of(previous.blocking.begin(),previous.blocking.end(),[&](const auto & item){return item.id==id;});
    };
    const auto deny=[&](const std::string & reason){error=reason;return false;};
    nlohmann::ordered_json phase_progress;
    if (!common_agent_method_review(plan,review.phase_reviews,phase_progress,error)) return false;
    if (plan.phase_review_required && review.decision==common_reflection_decision::accept) {
        auto projected=plan;
        projected.observations.push_back({"projected-phase-review","method_phase_review",
            nlohmann::ordered_json{{"schema","method-phase-review.v1"},{"phases",phase_progress}}.dump(),0.0f});
        if (common_agent_method_has_remaining(projected)) return deny("method phases remain unresolved; continue the existing plan or record a bounded limitation for each phase");
        const bool blocked=std::any_of(phase_progress.begin(),phase_progress.end(),[](const auto & phase){return phase.value("status",std::string{})=="blocked";});
        if (blocked && review.task_outcome!="blocked") return deny("blocked method phases cannot establish a satisfied user outcome");
    }
    if (review.task_outcome == "remaining" &&
            (review.decision == common_reflection_decision::accept ||
             review.next_action == common_agent_reflection_next_action::revise_response))
        return deny("unfinished task work requires plan repair or an authorized evidence phase, not answer acceptance or wording repair");
    for (const auto & operation : review.proposed_plan_operations) {
        if (operation.kind != common_plan_operation_kind::reset_step &&
                operation.kind != common_plan_operation_kind::activate_step &&
                operation.kind != common_plan_operation_kind::unblock_step) continue;
        const auto step_id = operation.step_id.value_or(operation.step ? operation.step->id : std::string{});
        const auto step = std::find_if(plan.steps.begin(), plan.steps.end(), [&](const auto & item) { return item.id == step_id; });
        if (step == plan.steps.end() || step->status != common_plan_step_status::failed) continue;
        if (!common_agent_review_step_allows_retry(plan, step_id))
            return deny("non-retryable failed tool call cannot be reset unchanged; use a supported corrected call or stop bounded");
    }
    for(const auto & id:review.resolved_issue_ids)
        if(!known(id) || !accounted.insert(id).second) return deny("unknown or duplicate resolved issue ID");
    for(const auto & issue:review.issues) {
        if(!issue.previous_issue_id.empty() && (!known(issue.previous_issue_id) || !accounted.insert(issue.previous_issue_id).second))
            return deny("unknown or duplicate previous issue ID");
        if(issue.description.empty() || issue.description.size()>512 || issue.correction.size()>512 ||
            (review.structured_review && issue.correction.empty()) || issue.evidence_ids.size()>8)
            return deny("blocking issue is empty or exceeds its bounds");
        for(const auto & id:issue.evidence_ids)
            if(std::none_of(plan.observations.begin(),plan.observations.end(),[&](const auto & observation){return observation.id==id;}))
                return deny("issue cites an unknown evidence ID");
        candidate.blocking.push_back({issue.previous_issue_id.empty() ?
            "review-"+std::to_string(candidate.round)+"-"+std::to_string(candidate.blocking.size()+1) : issue.previous_issue_id,
            issue.description,issue.correction,issue.evidence_ids});
    }
    // Omission never resolves a host-owned blocker. Retain it across both
    // structured and legacy reviews; explicit resolution remains mandatory.
    for (const auto & issue : previous.blocking)
        if (!accounted.count(issue.id)) candidate.blocking.push_back(issue);
    if(review.decision==common_reflection_decision::accept && !candidate.blocking.empty())
        return deny("accept cannot retain blockers; resolve prior issues explicitly");
    if(review.structured_review) {
        if(review.next_action==common_agent_reflection_next_action::revise_response && candidate.blocking.empty())
            return deny("revise_response requires a blocking correction, not advisory preferences");
    } else if(review.decision!=common_reflection_decision::accept) {
        for(const auto & text:review.revision_guidance) {
            if(candidate.blocking.size()>=4) break;
            candidate.blocking.push_back({"review-"+std::to_string(candidate.round)+"-"+
                std::to_string(candidate.blocking.size()+1),text,text,{}});
        }
    }
    if(candidate.blocking.size()>4 || candidate.advisory.size()>4) return deny("review checklist exceeds four entries per category");
    if(std::any_of(candidate.advisory.begin(),candidate.advisory.end(),[](const auto & text){return text.size()>512;}))
        return deny("advisory note exceeds its text bound");
    if(review.next_action==common_agent_reflection_next_action::revise_response && candidate.blocking.empty() &&
        !candidate.advisory.empty()) return deny("advisory preferences alone cannot require response revision");
    std::vector<common_agent_review_part> validated_parts;
    if(!common_agent_review_parts_from_json(common_agent_review_parts_json(candidate.parts),validated_parts))
        return deny("excerpt review exceeds its bounded contract");
    next=std::move(candidate);return true;
}
