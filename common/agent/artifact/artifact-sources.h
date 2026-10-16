#pragma once

#include "artifact-contract.h"

// Project already captured provenance. This never fetches a source or invents
// a request from a provider name or an operation identifier.
std::vector<common_agent_artifact_source_citation> common_agent_artifact_collect_sources(
        const common_agent_artifact_snapshot & snapshot);

std::string common_agent_artifact_render_source(const common_agent_artifact_source_citation & source);
