#include "agent-inference-executor.h"

#include "agent-inference-capacity-gate.h"

#include <cstdio>
#include <chrono>
#include <cstdlib>
#include <utility>

namespace {

bool agent_async_trace_enabled() {
    const char * value = std::getenv("LLAMA_AGENT_RESIDENT_TRACE");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

void agent_async_trace(const char * event, const char * detail = "") {
    if (!agent_async_trace_enabled()) {
        return;
    }
    std::fprintf(stderr, "agent async trace: event=%s %s\n", event, detail);
    std::fflush(stderr);
}

class async_inference_task final : public common_agent_runtime_inference_task {
public:
    async_inference_task(
            common_agent_runtime_session_host * host,
            common_agent_runtime_session_host_turn_request request,
            std::shared_ptr<common_agent_inference_capacity_gate> inference_gate,
            std::string lease_id)
        : result(std::make_shared<common_agent_runtime_session_host_turn_result>()),
          error(std::make_shared<std::string>()),
          request(std::move(request)),
          inference_gate(std::move(inference_gate)),
          lease_id(std::move(lease_id)) {
        future = std::async(
            std::launch::async,
            [this, host]() {
                agent_async_trace("worker-start");
                const auto release = [&]() {
                    if (this->inference_gate) {
                        this->inference_gate->release(this->lease_id);
                    }
                };
                try {
                    agent_async_trace("before-host-run-turn");
                    const bool ok = host->run_turn(this->request, *this->result, *this->error);
                    agent_async_trace("after-host-run-turn", ok ? "ok" : "failed");
                    release();
                    return ok;
                } catch (...) {
                    agent_async_trace("host-run-turn-threw");
                    release();
                    throw;
                }
            });
    }

    bool poll(
            bool & ready,
            common_agent_runtime_session_host_turn_result & output,
            std::string & output_error) override {
        ready = false;
        if (future.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready) {
            output_error.clear();
            return true;
        }
        try {
            const bool ok = future.get();
            output = *result;
            output_error = *error;
            ready = true;
            return ok || output.cancelled || output_error.empty();
        } catch (...) {
            output = {};
            output_error = "asynchronous inference executor failed";
            ready = true;
            return false;
        }
    }

    bool cancel(std::string & output_error) override {
        if (!request.execution_control.cancellation) {
            output_error = "asynchronous host turn has no cancellation state";
            return false;
        }
        request.execution_control.cancellation->request_cancel(
            request.execution_control.stop_reason().empty()
                ? "turn cancelled by host"
                : request.execution_control.stop_reason());
        output_error.clear();
        return true;
    }

private:
    std::shared_ptr<common_agent_runtime_session_host_turn_result> result;
    std::shared_ptr<std::string> error;
    std::future<bool> future;
    std::shared_ptr<common_agent_inference_capacity_gate> inference_gate;
    std::string lease_id;
    common_agent_runtime_session_host_turn_request request;
};

class async_inference_executor final : public common_agent_runtime_inference_executor {
public:
    std::shared_ptr<common_agent_runtime_inference_task> submit(
            common_agent_runtime_session_host * host,
            common_agent_runtime_session_host_turn_request request,
            const std::shared_ptr<common_agent_inference_capacity_gate> & inference_gate,
            std::string lease_id,
            std::string & error) override {
        if (host == nullptr) {
            error = "inference executor host is missing";
            return nullptr;
        }
        error.clear();
        return std::make_shared<async_inference_task>(
            host, std::move(request), inference_gate, std::move(lease_id));
    }
};

} // namespace

std::shared_ptr<common_agent_runtime_inference_executor>
make_common_agent_runtime_async_inference_executor() {
    return std::make_shared<async_inference_executor>();
}
