#include "review-checklist.h"
#include "agent/runtime/agent-reflection-engine.h"
#include "hash/hash.h"
#include <algorithm>
#include <set>
#include <limits>

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
    std::set<std::string> accounted;
    const auto known=[&](const std::string & id) {
        return std::any_of(previous.blocking.begin(),previous.blocking.end(),[&](const auto & item){return item.id==id;});
    };
    const auto deny=[&](const std::string & reason){error=reason;return false;};
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
    if(review.structured_review) {
        if(accounted.size()!=previous.blocking.size()) return deny("account for each prior blocker as resolved or retained");
        if(review.decision==common_reflection_decision::accept && !candidate.blocking.empty())
            return deny("accept cannot retain blockers; use advisory_notes for nonblocking suggestions");
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
    next=std::move(candidate);return true;
}
