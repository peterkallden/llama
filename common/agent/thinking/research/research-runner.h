#pragma once

#include "agent/thinking/research/research-controller.h"
#include "agent/thinking/research/research-assessor.h"
#include "agent/agent-runtime.h"
#include <chrono>

class common_agent_research_tool_executor {
public:
    virtual ~common_agent_research_tool_executor() = default;
    virtual const common_agent_research_assessor * evidence_assessor() const { return nullptr; }
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
            const common_agent_research_assessor * assessor = nullptr,
            std::function<bool()> should_stop = {},
            size_t chunk_max_bytes = 0,
            size_t chunk_overlap_bytes = 0,
            std::function<bool(const std::string &, const common_agent_generation_resource_chunk &,
                common_agent_resource_chunk_extraction &, int &, std::string &)> extract_chunk = {},
            std::function<bool(const common_agent_research_workspace &, std::string &)> progress_sink = {},
            std::function<bool(const std::string &, common_agent_resource_chunk_plan &, std::string &)> plan_chunks = {},
            decltype(common_agent_request::research_select_tool) select_tool = {},
            std::function<void(const std::string &, const std::string &)> selection_trace = {});

    bool execute(
            const common_agent_research_action & action,
            common_agent_research_workspace & workspace,
            common_agent_research_event & event,
            std::string & error) const override;
    const common_agent_research_assessor * evidence_assessor() const override { return assessor; }

private:
    const common_agent_tool_runtime & tools;
    const common_agent_research_assessor * assessor;
    std::function<bool()> should_stop;
    size_t chunk_max_bytes;
    size_t chunk_overlap_bytes;
    std::function<bool(const std::string &, const common_agent_generation_resource_chunk &,
        common_agent_resource_chunk_extraction &, int &, std::string &)> extract_chunk;
    std::function<bool(const common_agent_research_workspace &, std::string &)> progress_sink;
    std::function<bool(const std::string &, common_agent_resource_chunk_plan &, std::string &)> plan_chunks;
    decltype(common_agent_request::research_select_tool) select_tool;
    std::function<void(const std::string &, const std::string &)> selection_trace;
    std::chrono::steady_clock::time_point started_at;
};

class common_agent_research_runner {
public:
    common_agent_research_result run(
            common_agent_research_workspace & workspace,
            const common_agent_research_tool_executor & executor,
            std::string & error,
            const std::function<bool()> & should_stop = {},
            const std::function<common_agent_research_stop_reason()> & stop_reason = {},
            const common_agent_research_lifecycle_sink & lifecycle_sink = {},
            const std::function<bool(const common_agent_research_workspace &, std::string &)> & progress_sink = {}) const;
};
