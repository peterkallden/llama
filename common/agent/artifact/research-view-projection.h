#pragma once

#include "agent/artifact/artifact-contract.h"
#include "agent/visualization/visualization-contract.h"

#include <string>
#include <vector>

bool common_agent_research_report_evidence_graph(
    const common_agent_artifact_document & document,
    const common_agent_artifact_snapshot & snapshot,
    const std::vector<std::string> & selected_block_ids,
    common_agent_visualization_relationship & out,
    std::string & error);

bool common_agent_research_workspace_evidence_graph(
    const common_agent_research_workspace & workspace,
    common_agent_visualization_relationship & out,
    std::string & error);
