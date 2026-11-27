#pragma once
#include "agent/synthesis-state.h"
struct common_reflection_result;
// A missing retryability declaration retains legacy behavior. Explicit host
// non-retryable failures must not be reset unchanged.
bool common_agent_review_step_allows_retry(const common_plan_state &, const std::string & step_id);
// Identity/bounds validation only. The reviewer owns semantic judgments.
bool common_agent_advance_review_checklist(const common_agent_review_checklist & previous,
    const common_reflection_result &,const common_plan_state &,const std::string & draft,
    common_agent_review_checklist & next,std::string & error, bool budget_finalization = false);
