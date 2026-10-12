#include "agent/runtime/agent-event-queue.h"
#include "tools/agent/runtime/agent-runtime-control.h"

#include <cassert>
#include <chrono>
#include <thread>

static common_agent_event event(common_agent_event_type type, const char * detail) {
    common_agent_event result;
    result.type = type;
    result.detail = detail;
    return result;
}

int main() {
    common_agent_event_queue queue;
    assert(queue.push(event(common_agent_event_type::plan_created, "first")));
    assert(queue.push(event(common_agent_event_type::tool_executed, "second")));
    assert(queue.size() == 2);

    common_agent_event first;
    common_agent_event second;
    assert(queue.try_pop(first));
    assert(queue.try_pop(second));
    assert(first.detail == "first");
    assert(second.detail == "second");
    assert(!queue.try_pop(first));

    std::thread producer([&queue] {
        queue.push(event(common_agent_event_type::response_revised, "worker"));
    });
    assert(queue.wait_pop(first, std::chrono::seconds(1)));
    producer.join();
    assert(first.detail == "worker");

    queue.close();
    assert(queue.closed());
    assert(!queue.push(event(common_agent_event_type::plan_updated, "late")));
    assert(!queue.wait_pop(first, std::chrono::milliseconds(1)));

    const auto cancellation = std::make_shared<common_agent_runtime_cancellation_state>();
    common_agent_runtime_execution_control control;
    control.cancellation = cancellation;
    assert(!control.should_stop());
    assert(cancellation->request_cancel("Android UI cancelled turn"));
    assert(control.should_stop());
    assert(control.stop_reason() == "Android UI cancelled turn");
    assert(!cancellation->request_cancel("ignored"));
    assert(cancellation->reason() == "Android UI cancelled turn");
    return 0;
}
