#pragma once

#include "agent/thinking/research/research-contract.h"

bool common_agent_research_add_gap(
        common_agent_research_workspace & workspace,
        common_agent_research_gap gap,
        std::string & error);

bool common_agent_research_add_task(
        common_agent_research_workspace & workspace,
        common_agent_research_task task,
        std::string & error);

bool common_agent_research_add_source(
        common_agent_research_workspace & workspace,
        common_agent_research_source source,
        std::string & error);

bool common_agent_research_record_evidence(
        common_agent_research_workspace & workspace,
        common_agent_research_evidence evidence,
        std::string & error);

bool common_agent_research_add_comparison(
        common_agent_research_workspace & workspace,
        common_agent_research_source_comparison comparison,
        std::string & error);

bool common_agent_research_update_coverage(
        common_agent_research_workspace & workspace,
        common_agent_research_coverage coverage,
        std::string & error);

bool common_agent_research_transition_gap(
        common_agent_research_workspace & workspace,
        const std::string & gap_id,
        common_agent_research_gap_status target,
        std::string & error);

bool common_agent_research_transition_task(
        common_agent_research_workspace & workspace,
        const std::string & task_id,
        common_agent_research_task_status target,
        std::string & error);
