#pragma once

#include "agent/dataset-contracts.h"
#include "agent/tooling/catalog/tool-catalog.h"

#include <string>

// Conservative model-facing projections derived from a full host definition.
// Explicit model schemas remain the override for tools with richer semantics.
std::string common_tool_default_model_input_projection(
        const common_tool_definition & definition);

std::string common_tool_default_model_result_projection(
        const common_tool_definition & definition);

// Enrich a model schema from an authoritative turn snapshot. Dataset-reference
// fields use dN handles; x-agent-column-of names their source dataset field.
// Column enums are emitted only when every candidate's schema is known.
bool common_agent_project_dataset_model_schema(
        nlohmann::ordered_json & schema,
        const std::vector<common_agent_dataset_descriptor> & datasets,
        const nlohmann::ordered_json & bound_arguments,
        std::string & error);

// Validate resolved canonical arguments against the selected dataset, since a
// union used during model generation cannot establish per-dataset membership.
bool common_agent_validate_dataset_column_arguments(
        const nlohmann::ordered_json & schema,
        const nlohmann::ordered_json & arguments,
        const std::vector<common_agent_dataset_descriptor> & datasets,
        std::string & error);
