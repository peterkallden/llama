#pragma once

#include "memory/memory-types.h"
#include "plan/plan-types.h"

#include <vector>

// Retains only provenance from actually retrieved procedure memories. Model
// output may suggest IDs, but never gains authority to invent evidence.
void common_plan_bind_memory_provenance(
    common_plan_operation & operation,
    const std::vector<common_memory_hit> & memories);
