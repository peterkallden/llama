#pragma once

#include "agent/agent-inference.h"
#include "agent/runtime/model-catalog.h"

#include <memory>

std::unique_ptr<common_agent_inference> make_agent_http_inference(
    const common_agent_model_selection & selection);
