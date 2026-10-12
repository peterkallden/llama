#pragma once

// Compatibility umbrella for the agent runtime contracts. New code should
// include the narrow header under agent/runtime/ that matches its seam;
// existing consumers may continue including this stable public entrypoint.
#include "agent/runtime/agent-action-executor.h"
#include "agent/runtime/agent-planner.h"
#include "agent/runtime/agent-reflection-engine.h"
#include "agent/runtime/agent-runtime.h"
#include "agent/runtime/agent-tool-runtime.h"
