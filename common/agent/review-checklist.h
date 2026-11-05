#pragma once
#include "agent/synthesis-state.h"
struct common_reflection_result;
// Identity/bounds validation only. The reviewer owns semantic judgments.
bool common_agent_advance_review_checklist(const common_agent_review_checklist & previous,
    const common_reflection_result &,const common_plan_state &,const std::string & draft,
    common_agent_review_checklist & next,std::string & error);
