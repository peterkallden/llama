#pragma once
#include "plan/plan-types.h"
struct common_plan_context_config { size_t char_budget = 2048; };
std::string common_plan_escape_context_text(const std::string & text);
std::string common_plan_render_context(const common_plan_state & plan, const common_plan_context_config & config = {});
std::string common_plan_render_tool_observations(
        const common_plan_state & plan,
        const common_plan_context_config & config = {});
std::string common_plan_render_step_context(const common_plan_state & plan, const common_plan_step & step, const common_plan_context_config & config = {});
