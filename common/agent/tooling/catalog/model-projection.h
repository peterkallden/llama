#pragma once

#include "agent/tooling/catalog/tool-catalog.h"

#include <string>

// Conservative model-facing projections derived from a full host definition.
// Explicit model schemas remain the override for tools with richer semantics.
std::string common_tool_default_model_input_projection(
        const common_tool_definition & definition);

std::string common_tool_default_model_result_projection(
        const common_tool_definition & definition);
