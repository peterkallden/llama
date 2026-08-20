#pragma once

#include "agent-runtime-session-host.h"

#include <future>
#include <memory>
#include <string>

class common_agent_inference_capacity_gate;

class common_agent_runtime_inference_task {
public:
    virtual ~common_agent_runtime_inference_task() = default;

    virtual bool poll(
        bool & ready,
        common_agent_runtime_session_host_turn_result & result,
        std::string & error) = 0;

    virtual bool cancel(std::string & error) = 0;
};

class common_agent_runtime_inference_executor {
public:
    virtual ~common_agent_runtime_inference_executor() = default;

    virtual std::shared_ptr<common_agent_runtime_inference_task> submit(
        common_agent_runtime_session_host * host,
        common_agent_runtime_session_host_turn_request request,
        const std::shared_ptr<common_agent_inference_capacity_gate> & inference_gate,
        std::string lease_id,
        std::string & error) = 0;
};

std::shared_ptr<common_agent_runtime_inference_executor>
make_common_agent_runtime_async_inference_executor();
