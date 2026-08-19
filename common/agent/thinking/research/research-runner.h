#pragma once

#include "agent/thinking/research/research-controller.h"
#include "agent/thinking/research/research-assessor.h"
#include "agent/agent-runtime.h"

class common_agent_research_tool_executor {
public:
    virtual ~common_agent_research_tool_executor() = default;
    virtual bool execute(
            const common_agent_research_action & action,
            common_agent_research_workspace & workspace,
            common_agent_research_event & event,
            std::string & error) const = 0;
};

class common_agent_research_runtime_adapter final : public common_agent_research_tool_executor {
public:
    explicit common_agent_research_runtime_adapter(
            const common_agent_tool_runtime & tools,
            const common_agent_research_assessor * assessor = nullptr);

    bool execute(
            const common_agent_research_action & action,
            common_agent_research_workspace & workspace,
            common_agent_research_event & event,
            std::string & error) const override;

private:
    const common_agent_tool_runtime & tools;
    const common_agent_research_assessor * assessor;
};

class common_agent_research_runner {
public:
    common_agent_research_result run(
            common_agent_research_workspace & workspace,
            const common_agent_research_tool_executor & executor,
            std::string & error,
            const std::function<bool()> & should_stop = {},
            const std::function<common_agent_research_stop_reason()> & stop_reason = {},
            const common_agent_research_lifecycle_sink & lifecycle_sink = {}) const;
};
