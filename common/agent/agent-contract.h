// Full agent contract umbrella.
//
// This header intentionally remains as a stable compatibility and public API
// entrypoint. New entries belong in the narrow headers under agent/contracts/;
// implementation code should include those headers directly when it only
// consumes one contract family.
#pragma once

#include "agent/contracts/agent-events.h"
#include "agent/contracts/agent-failures.h"
#include "agent/contracts/agent-learning.h"
#include "agent/contracts/agent-request.h"
#include "agent/contracts/agent-result.h"
