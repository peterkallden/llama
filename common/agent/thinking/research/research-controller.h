#pragma once

#include "agent/thinking/research/research-workspace.h"

enum class common_agent_research_event_type {
    task_completed,
    task_failed,
    cancelled,
};

enum class common_agent_research_assessment_status {
    inconclusive,
    sufficient,
    insufficient,
    contradicted,
};

struct common_agent_research_assessment {
    common_agent_research_assessment_status status =
        common_agent_research_assessment_status::inconclusive;
    double confidence = 0.0;
    std::string summary;
};

struct common_agent_research_event {
    common_agent_research_event_type type = common_agent_research_event_type::task_completed;
    std::string task_id;
    std::string gap_id;
    int evidence_count = 0;
    std::vector<std::string> evidence_ids;
    bool gap_sufficiently_answered = false;
    double gap_confidence = 0.0;
    std::string assessment_summary;
    common_agent_research_assessment assessment;
    bool retryable = false;
    std::string failure_code;
    std::string failure_summary;
};

enum class common_agent_research_action_kind {
    schedule_task,
    complete,
};

struct common_agent_research_action {
    common_agent_research_action_kind kind = common_agent_research_action_kind::complete;
    std::string task_id;
    std::string gap_id;
    std::string instruction;
    std::vector<std::string> preferred_tools;
    common_agent_research_stop_reason stop_reason = common_agent_research_stop_reason::budget_exhausted;
};

class common_agent_research_controller {
public:
    common_agent_research_action begin(
            common_agent_research_workspace & workspace,
            std::string & error) const;

    common_agent_research_action advance(
            common_agent_research_workspace & workspace,
            const common_agent_research_event & event,
            std::string & error) const;

    common_agent_research_result finalize(
            const common_agent_research_workspace & workspace) const;
};
