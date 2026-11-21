#include "common/agent/protocol/agent-jsonl.h"

#include <cassert>
#include <cstdio>
#include <string>

int main() {
    common_agent_jsonl_turn_request request;
    request.prompt = "inspect the attachment";
    request.session_id = "android";
    request.namespace_id = "default";
    request.project_id = "project-1";
    request.turn_id = "turn-1";
    request.mode = "agent";
    request.resource_refs = {"resource://letter"};
    request.include_summary = true;
    request.tool_timeout_ms = 2500;

    const auto encoded_request = common_agent_jsonl_make_turn_request(request);
    assert(encoded_request.value("command", "") == "run_turn");
    assert(encoded_request.value("mode", "") == "agent");
    assert(encoded_request["resource_refs"].at(0) == "resource://letter");
    assert(encoded_request.value("tool_timeout_ms", 0U) == 2500U);

    FILE * stream = std::tmpfile();
    assert(stream != nullptr);
    std::string error;
    assert(common_agent_jsonl_write_message(stream, encoded_request, error));
    std::rewind(stream);
    nlohmann::ordered_json decoded_request;
    assert(common_agent_jsonl_read_message(stream, decoded_request, error));
    assert(decoded_request == encoded_request);
    std::fclose(stream);

    common_agent_jsonl_event_entry event;
    event.type = "tool.completed";
    event.event_type = "tool.completed";
    event.turn_id = "turn-1";
    event.sequence = 7;
    event.detail = "resource read";
    const auto encoded_event = common_agent_jsonl_make_event_message(event);
    assert(encoded_event.value("message_type", "") == "event");
    assert(encoded_event["event"].value("event_type", "") == "tool.completed");
    assert(encoded_event["event"].value("sequence", 0U) == 7U);

    common_agent_jsonl_turn_result result;
    result.request_id = "3";
    result.ok = true;
    result.response = "done";
    result.plan_id = "plan-1";
    result.event_count = 2;
    const auto encoded_result = common_agent_jsonl_make_turn_result(result);
    assert(encoded_result.value("message_type", "") == "response");
    assert(encoded_result.value("request_id", std::string()) == "3");
    assert(encoded_result.value("response", "") == "done");

    std::puts("agent_jsonl_wire_contract=ok");
    return 0;
}
