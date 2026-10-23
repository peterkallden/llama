#include "tools/agent/daemon/agent-daemon-event-json.h"
#include "tools/agent/daemon/agent-daemon-jsonl-protocol.h"
#include "tools/agent/daemon/agent-daemon-adapter.h"
#include "base64.hpp"
#include "tools/agent/web/agent-web-chat-title.h"
#include "agent/protocol/agent-sse.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>

int main() {
    nlohmann::ordered_json named_chat = {{"latest_turn_id", "new-turn"}, {"title", "Earlier name"}, {"attachments", {"resource://kept"}}};
    const auto plan_goals = nlohmann::ordered_json::array({
        {{"plan_id", "plan"}, {"goal", "Old goal"}, {"version", 1}},
        {{"plan_id", "plan"}, {"goal", "Investigate authentication failures"}, {"version", 3}},
        {{"plan_id", "other"}, {"goal", "Wrong conversation"}, {"version", 9}},
    });
    assert(agent_web_apply_goal_title(named_chat, "new-turn", "plan", plan_goals));
    assert(named_chat["title"] == "Investigate authentication failures");
    assert(named_chat["title_plan_version"] == 3);
    assert(named_chat["attachments"][0] == "resource://kept");
    assert(!agent_web_apply_goal_title(named_chat, "old-turn", "plan", plan_goals));
    assert(named_chat["title"] == "Investigate authentication failures");
    assert(!agent_web_apply_goal_title(named_chat, "new-turn", "missing", plan_goals));
    assert(!agent_web_apply_goal_title(named_chat, "new-turn", "plan", nlohmann::ordered_json::array({
        {{"plan_id", "plan"}, {"goal", "   "}, {"version", 4}}})));
    assert(agent_web_chat_title("  First question\nextra context") == "First question");
    const std::string utf8_title = std::string(76, 'a') + "ååå";
    assert(!nlohmann::ordered_json::parse(nlohmann::ordered_json(agent_web_chat_title(utf8_title)).dump()).is_discarded());

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
    delivery.event.visualization_json = R"({"schema_version":1,"kind":"relationship","title":"Call relationships","nodes":[],"edges":[],"complete":true,"truncated":false,"limitations":[],"view":{"layout":"cose","selected_nodes":[],"selected_edges":[]}})";
    const auto app_wire = make_agent_daemon_jsonl_event_message("browser", delivery);
    assert(app_wire["event"] == common_agent_daemon_event_to_json(delivery.event));
    assert(app_wire["event"]["app_context_available"] == true);
    assert(app_wire["event"]["tool_input"]["dataset"] == "dataset://sample");
    assert(app_wire["event"]["tool_result"]["structuredContent"]["rows"][0]["value"] == 4);
    assert(app_wire["event"]["provider_id"] == "catalog");
    assert(app_wire["event"]["visualization"]["kind"] == "relationship");

    delivery.event.visualization_json = "not-json";
    assert(!common_agent_daemon_event_to_json(delivery.event).contains("visualization"));

    delivery.event.tool_result_json = "not-json";
    assert(!make_agent_daemon_jsonl_event_message("browser", delivery)["event"].contains("tool_result"));
    delivery.event.app_context_available = false;
    assert(!make_agent_daemon_jsonl_event_message("browser", delivery)["event"].contains("tool_input"));
    agent_daemon_jsonl_read_resource_request byte_request;
    byte_request.uri = "agent-resource://snapshot";
    byte_request.bytes_base64 = true;
    const auto byte_command = make_agent_daemon_jsonl_read_resource_request(byte_request);
    assert(byte_command.at("encoding") == "base64");
    common_agent_daemon_command parsed_bytes;
    std::string byte_error;
    assert(parse_agent_daemon_command(byte_command, daemon_options{}, common_agent_runtime_host_mode::agent,
        parsed_bytes, byte_error) && parsed_bytes.resource->bytes_base64);
    auto invalid_encoding = byte_command;
    invalid_encoding["encoding"] = "executable";
    assert(!parse_agent_daemon_command(invalid_encoding, daemon_options{}, common_agent_runtime_host_mode::agent,
        parsed_bytes, byte_error));

    const std::string binary_png("\x89PNG\r\n\x1a\n\0\xff", 10);
    common_agent_daemon_command_result binary_result;
    binary_result.ok = true;
    binary_result.event = "resource_read";
    binary_result.response_kind = common_agent_daemon_response_kind::resource;
    binary_result.resource_result.resource.uri = byte_request.uri;
    binary_result.resource_result.resource.mime_type = "image/png";
    binary_result.resource_result.resource.size_bytes = binary_png.size();
    binary_result.resource_result.content = base64::encode(binary_png);
    binary_result.resource_result.bytes_base64 = true;
    const auto byte_response = make_agent_daemon_jsonl_command_response(binary_result);
    assert(byte_response.at("content_encoding") == "base64");
    assert(base64::decode(byte_response.at("content").get<std::string>()) == binary_png);
    // The response must be valid JSON even with non-UTF8 source bytes and NULs.
    assert(!byte_response.dump().empty());
    return 0;
}
