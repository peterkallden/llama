#pragma once
#include <cstddef>

// Both configured review rounds and plan revisions require execution capacity.
size_t common_agent_continuation_execution_limit(int reflection_rounds, int plan_revisions);
