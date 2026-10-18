#include "tools/agent/daemon/agent-daemon-event-json.h"
#include "tools/agent/daemon/agent-daemon-jsonl-protocol.h"
#include "agent/protocol/agent-sse.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>

int main() {
    common_agent_event_context context;
    context.namespace_id = "web";
    context.session_id = "web-session";
    context.project_id = "web-example";
    common_agent_event_stream_delivery delivery;
    delivery.kind = common_agent_event_stream_delivery_kind::event;
    delivery.event = make_common_agent_daemon_event(
        common_agent_daemon_event_type::resource_created, "request", "turn", "saved", 7, context);
    delivery.event.resource_uri = "agent-resource://report";
    delivery.event.resource_name = "Research report.json";
    delivery.event.resource_mime_type = "application/vnd.llama-agent.report+json";
    const auto wire = make_agent_daemon_jsonl_event_message("browser", delivery);
    assert(wire["event"] == common_agent_daemon_event_to_json(delivery.event));
    assert(wire["event"]["resource_uri"] == delivery.event.resource_uri);
    assert(wire["event"]["resource_mime_type"] == delivery.event.resource_mime_type);
    assert(wire["event"]["session_id"] == context.session_id);
    assert(wire["event"]["turn_id"] == "turn");
    const auto sse = common_agent_sse_format_message(wire);
    assert(sse.find("id: 7\n") != std::string::npos);
    assert(sse.find("data: " + wire.dump()) != std::string::npos);

    delivery.event.event_type = common_agent_daemon_event_type::tool_completed;
    delivery.event.app_context_available = true;
    delivery.event.provider_id = "catalog";
    delivery.event.server_tool_name = "dataset.sample";
    delivery.event.tool_input_json = R"({"dataset":"dataset://sample"})";
    delivery.event.tool_result_json = R"({"isError":false,"structuredContent":{"rows":[{"value":4}]}})";
    const auto app_wire = make_agent_daemon_jsonl_event_message("browser", delivery);
    assert(app_wire["event"] == common_agent_daemon_event_to_json(delivery.event));
    assert(app_wire["event"]["app_context_available"] == true);
    assert(app_wire["event"]["tool_input"]["dataset"] == "dataset://sample");
    assert(app_wire["event"]["tool_result"]["structuredContent"]["rows"][0]["value"] == 4);
    assert(app_wire["event"]["provider_id"] == "catalog");

    delivery.event.tool_result_json = "not-json";
    assert(!make_agent_daemon_jsonl_event_message("browser", delivery)["event"].contains("tool_result"));
    delivery.event.app_context_available = false;
    assert(!make_agent_daemon_jsonl_event_message("browser", delivery)["event"].contains("tool_input"));
    return 0;
}
