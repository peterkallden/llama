#include "agent-http-inference.h"

#include "agent/tooling/contracts/schema-contract.h"
#include "agent/adaptation/flydelta/flydelta-activation.h"
#include "agent/adaptation/flydelta/flydelta-hidden-state-hook.h"
#include "agent/adaptation/flydelta/flydelta-model-adapter.h"
#include "agent/agent-residual-patch.h"

#include <cpp-httplib/httplib.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <limits>
#include <unordered_map>

namespace {
using json = nlohmann::ordered_json;

struct endpoint_parts { std::string scheme, host, path; int port = 0; };

bool split_endpoint(const std::string & endpoint, endpoint_parts & out, std::string & error) {
    const size_t scheme_end = endpoint.find("://");
    if (scheme_end == std::string::npos) { error = "HTTP inference endpoint must include http:// or https://"; return false; }
    out.scheme = endpoint.substr(0, scheme_end);
    const size_t authority_start = scheme_end + 3;
    const size_t path_start = endpoint.find('/', authority_start);
    const std::string authority = endpoint.substr(authority_start,
        path_start == std::string::npos ? std::string::npos : path_start - authority_start);
    if (authority.empty() || authority.find('@') != std::string::npos) {
        error = "HTTP inference endpoint has an invalid authority"; return false;
    }
    if (!authority.empty() && authority.front() == '[') {
        const size_t close = authority.find(']');
        if (close == std::string::npos) { error = "HTTP inference endpoint has an invalid IPv6 host"; return false; }
        out.host = authority.substr(1, close - 1);
        if (close + 1 < authority.size()) {
            if (authority[close + 1] != ':') { error = "HTTP inference endpoint has an invalid port"; return false; }
            try { out.port = std::stoi(authority.substr(close + 2)); } catch (...) { out.port = 0; }
        } else {
            out.port = out.scheme == "https" ? 443 : 80;
        }
    } else {
        const size_t colon = authority.rfind(':');
        if (colon != std::string::npos) {
            out.host = authority.substr(0, colon);
            try { out.port = std::stoi(authority.substr(colon + 1)); } catch (...) { out.port = 0; }
        } else {
            out.host = authority;
            out.port = out.scheme == "https" ? 443 : 80;
        }
    }
    if (out.host.empty() || out.port < 1 || out.port > 65535 ||
            (out.scheme != "http" && out.scheme != "https")) {
        error = "HTTP inference endpoint must be a valid http or https URL"; return false;
    }
    out.path = path_start == std::string::npos ? "" : endpoint.substr(path_start);
    while (!out.path.empty() && out.path.back() == '/') out.path.pop_back();
    error.clear();
    return true;
}

template <typename Client>
void set_timeouts(Client & client, uint32_t connect_ms, uint32_t request_ms) {
    client.set_connection_timeout(connect_ms / 1000, static_cast<time_t>((connect_ms % 1000) * 1000));
    client.set_read_timeout(request_ms / 1000, static_cast<time_t>((request_ms % 1000) * 1000));
    client.set_write_timeout(request_ms / 1000, static_cast<time_t>((request_ms % 1000) * 1000));
    client.set_max_timeout(static_cast<time_t>(request_ms));
    client.set_payload_max_length(16U * 1024U * 1024U);
    client.set_follow_location(false);
}

bool locally_validated_dsl_purpose(common_agent_generation_purpose purpose) {
    switch (purpose) {
        case common_agent_generation_purpose::planner:
        case common_agent_generation_purpose::tool_family_selection:
        case common_agent_generation_purpose::route_selection:
        case common_agent_generation_purpose::operation_selection:
        case common_agent_generation_purpose::plan_selection:
        case common_agent_generation_purpose::blueprint_selection:
        case common_agent_generation_purpose::blueprint_binding:
        case common_agent_generation_purpose::reflection:
            return true;
        default:
            return false;
    }
}

bool is_media_mime_type(const std::string & mime_type) {
    std::string normalized = mime_type;
    std::transform(normalized.begin(), normalized.end(), normalized.begin(),
        [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return normalized.rfind("image/", 0) == 0 || normalized.rfind("audio/", 0) == 0;
}

// Escape every non-alphanumeric byte, including underscores, so that
// distinct internal names cannot collapse into the same external name.
bool external_tool_name(const std::string & name, std::string & external) {
    static constexpr char hex[] = "0123456789abcdef";
    external.clear();
    for (const unsigned char ch : name) {
        if ((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
                (ch >= '0' && ch <= '9') || ch == '-') external += static_cast<char>(ch);
        else { external += '_'; external += hex[ch >> 4]; external += hex[ch & 15]; }
        if (external.size() > 64) return false;
    }
    return !external.empty();
}

// OpenAI-compatible Chat Completions requires a JSON Schema object at the
// response_format root. Workflow action selection currently expresses each
// authorized tool as a root-level oneOf branch. Keep that internal contract,
// but project it into a supported flat object for HTTP providers. The host
// still validates the returned tool/args pair against the selected tool's
// original schema after generation.
bool normalize_response_schema(const json & input, json & output, bool & wrapped_value) {
    wrapped_value = false;
    if (input.is_object() && input.contains("type") && input["type"].is_string() &&
            input["type"] != "object") {
        output = {{"type", "object"}, {"additionalProperties", false},
            {"required", json::array({"value"})}, {"properties", {{"value", input}}}};
        wrapped_value = true;
        return true;
    }
    if (!input.is_object() || !input.contains("oneOf")) {
        output = input;
        return input.is_object();
    }
    const auto & branches = input["oneOf"];
    if (!branches.is_array() || branches.empty()) return false;

    json tool_names = json::array();
    json argument_schemas = json::array();
    for (const auto & branch : branches) {
        if (!branch.is_object() || !branch.contains("properties") || !branch["properties"].is_object()) return false;
        const auto & properties = branch["properties"];
        if (!properties.contains("tool") || !properties["tool"].is_object() ||
                !properties["tool"].contains("enum") || !properties["tool"]["enum"].is_array() ||
                properties["tool"]["enum"].size() != 1 || !properties["tool"]["enum"][0].is_string() ||
                !properties.contains("args") || !properties["args"].is_object()) return false;
        tool_names.push_back(properties["tool"]["enum"][0]);
        argument_schemas.push_back(properties["args"]);
    }
    output = {
        {"type", "object"},
        {"additionalProperties", false},
        {"required", json::array({"tool", "args"})},
        {"properties", {
            {"tool", {{"type", "string"}, {"enum", std::move(tool_names)}}},
            {"args", {{"oneOf", std::move(argument_schemas)}}},
        }},
    };
    return true;
}

class http_inference final : public common_agent_inference {
public:
    explicit http_inference(common_agent_model_selection selection)
        : selection_(std::move(selection)) {
        if (!selection_.token_env.empty()) {
            const char * token = std::getenv(selection_.token_env.c_str());
            if (token != nullptr) token_ = token;
        }
    }

    size_t context_capacity_tokens() const override { return selection_.context_size_tokens; }

    bool generate(const common_agent_generation_request & request,
            common_agent_generation_result & result) override {
        result = {};
        result.status = common_agent_generation_status::errored;
        result.stop_reason = common_agent_generation_stop_reason::error;
        const auto fail = [&](const std::string & message) { result.error_message = message; return true; };
        if (request.options.should_stop && request.options.should_stop()) {
            result.status = common_agent_generation_status::cancelled;
            result.stop_reason = common_agent_generation_stop_reason::cancelled;
            result.error_message = "HTTP inference request cancelled before submission";
            return true;
        }
        if (!selection_.token_env.empty() && token_.empty()) return fail("HTTP inference bearer-token environment variable is unset");
        if (!request.grammar.empty()) {
            if ((selection_.tool_output_format_configured &&
                    selection_.tool_output_format != common_agent_tool_output_format::compact_dsl) ||
                    !locally_validated_dsl_purpose(request.purpose) || !request.json_schema.empty())
                return fail("HTTP inference backend does not support GBNF grammar constraints");
            std::fprintf(stderr, "agent: HTTP inference using unconstrained text for a locally validated DSL request; GBNF is unavailable\n");
        }
        if ((request.flydelta_activation && request.flydelta_activation->gate.apply) ||
                (request.flydelta_capture && request.flydelta_capture->enabled) || request.residual_patch ||
                (request.flydelta_device_diagnostics && request.flydelta_device_diagnostics->enabled))
            return fail("HTTP inference backend does not support FlyDelta interventions, capture, or device diagnostics");
        if (std::any_of(request.input_resources.begin(), request.input_resources.end(),
                [](const common_agent_generation_resource & resource) {
                    return is_media_mime_type(resource.resource.mime_type);
                }) || (request.resource_chunk && is_media_mime_type(request.resource_chunk->source.mime_type)))
            return fail("HTTP inference backend does not support image or audio input resources");
        if (std::any_of(request.messages.begin(), request.messages.end(),
                [](const common_chat_msg & message) { return message.contains_media(); }))
            return fail("HTTP inference backend does not support image or audio message content");
        if (!request.json_schema.empty() && !selection_.supports_json_schema)
            return fail("HTTP model profile does not declare JSON-schema support");
        if (request.tools.empty() && request.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE)
            return fail("HTTP inference tool_choice requires at least one declared tool");
        if ((!request.tools.empty() || request.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE) &&
                !selection_.supports_native_tool_calls)
            return fail("HTTP model profile does not declare native tool-call support");
        if (selection_.token_limit_parameter != "max_tokens" && selection_.token_limit_parameter != "max_completion_tokens")
            return fail("HTTP inference has an invalid token limit parameter");

        endpoint_parts endpoint;
        std::string error;
        if (!split_endpoint(selection_.endpoint, endpoint, error)) return fail(error);
        auto messages = request.messages;
        auto tools = request.tools;
        std::unordered_map<std::string, std::string> internal_names;
        if (selection_.map_tool_names) {
            for (auto & tool : tools) {
                std::string mapped;
                if (!external_tool_name(tool.name, mapped) || !internal_names.emplace(mapped, tool.name).second)
                    return fail("HTTP inference cannot map tool names uniquely within the 64-character limit");
                tool.name = std::move(mapped);
            }
            for (auto & msg : messages) {
                for (auto & call : msg.tool_calls) {
                    std::string mapped;
                    if (!external_tool_name(call.name, mapped)) return fail("HTTP inference cannot map a historical tool name");
                    call.name = std::move(mapped);
                }
                if (!msg.tool_name.empty()) {
                    std::string mapped;
                    if (!external_tool_name(msg.tool_name, mapped)) return fail("HTTP inference cannot map a historical tool name");
                    msg.tool_name = std::move(mapped);
                }
            }
        }
        json body;
        json response_schema;
        bool wrapped_response_value = false;
        try {
            body["model"] = selection_.model_id;
            body["messages"] = json::parse(common_chat_msgs_to_json_oaicompat(messages).dump());
            body["stream"] = false;
            if (request.options.n_predict > 0) body[selection_.token_limit_parameter] = request.options.n_predict;
            if (!selection_.reasoning_effort.empty()) body["reasoning_effort"] = selection_.reasoning_effort;
            if (!tools.empty()) body["tools"] = json::parse(common_chat_tools_to_json_oaicompat(tools).dump());
            if (!request.tools.empty()) {
                if (request.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED) body["tool_choice"] = "required";
                else if (request.tool_choice == COMMON_CHAT_TOOL_CHOICE_NONE) body["tool_choice"] = "none";
                else body["tool_choice"] = "auto";
            }
            if (!request.json_schema.empty()) {
                const auto schema = json::parse(request.json_schema);
                if (!normalize_response_schema(schema, response_schema, wrapped_response_value))
                    return fail("HTTP inference cannot map the requested JSON schema to the provider's object-root contract");
                if (wrapped_response_value) {
                    // The internal contract can request a scalar. The provider
                    // receives an object, so clarify how to represent that value.
                    auto & wire_messages = body["messages"];
                    const auto position = wire_messages.empty() ? wire_messages.end() : wire_messages.end() - 1;
                    wire_messages.insert(position,
                        json{{"role", "system"}, {"content", "Return the requested value as the value field of the JSON object required by the response schema."}});
                }
                body["response_format"] = {{"type", "json_schema"}, {"json_schema", {
                    {"name", "agent_response"}, {"strict", selection_.strict_json_schema}, {"schema", response_schema}}}};
            }
        } catch (const std::exception &) {
            return fail("HTTP inference request contains invalid message or JSON-schema data");
        }

        // Generation tracing is opt-in and may include prompt and model content.
        // Never log authentication headers or a literal bearer token.
        const auto trace_payload = [&](const char * label, const std::string & payload) {
            if (!request.options.generation_trace) return;
            std::string safe = payload;
            if (!token_.empty()) {
                size_t pos = 0;
                while ((pos = safe.find(token_, pos)) != std::string::npos) {
                    safe.replace(pos, token_.size(), "[REDACTED]");
                    pos += sizeof("[REDACTED]") - 1;
                }
            }
            constexpr size_t max_trace_bytes = 65536;
            const bool truncated = safe.size() > max_trace_bytes;
            if (truncated) safe.resize(max_trace_bytes);
            std::fprintf(stderr, "agent HTTP generation trace: purpose=%s trace_id=%s %s bytes=%zu%s begin\n",
                common_agent_generation_purpose_name(request.purpose), request.trace_id.value_or("").c_str(),
                label, safe.size(), truncated ? " truncated" : "");
            if (!safe.empty()) std::fwrite(safe.data(), 1, safe.size(), stderr);
            std::fprintf(stderr, "\nagent HTTP generation trace: %s end\n", label);
        };
        trace_payload("request", body.dump());

        httplib::Headers headers;
        if (!token_.empty()) headers.emplace("Authorization", "Bearer " + token_);
        const std::string route = endpoint.path + "/v1/chat/completions";
        uint32_t request_timeout_ms = selection_.request_timeout_ms;
        for (const auto limit : {request.options.t_max_prompt_ms, request.options.t_max_predict_ms}) {
            if (limit && *limit > 0) {
                request_timeout_ms = std::min<uint32_t>(request_timeout_ms,
                    static_cast<uint32_t>(std::min<int64_t>(*limit, std::numeric_limits<uint32_t>::max())));
            }
        }
        httplib::Result response;
        const auto request_started = std::chrono::steady_clock::now();
        if (endpoint.scheme == "http") {
            httplib::Client client(endpoint.host, endpoint.port);
            set_timeouts(client, selection_.connect_timeout_ms, request_timeout_ms);
            response = client.Post(route, headers, body.dump(), "application/json");
        } else {
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
            httplib::SSLClient client(endpoint.host, endpoint.port);
            set_timeouts(client, selection_.connect_timeout_ms, request_timeout_ms);
            response = client.Post(route, headers, body.dump(), "application/json");
#else
            return fail("HTTPS inference requires an OpenSSL-enabled HTTP transport");
#endif
        }
        if (!response) {
            if (request.options.should_stop && request.options.should_stop()) {
                result.status = common_agent_generation_status::cancelled;
                result.stop_reason = common_agent_generation_stop_reason::cancelled;
                result.error_message = "HTTP inference request cancelled while awaiting the remote server";
                return true;
            }
            const auto transport_error = response.error();
            const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - request_started).count();
            const bool elapsed_timeout = elapsed_ms + 2 >= request_timeout_ms &&
                (transport_error == httplib::Error::Read || transport_error == httplib::Error::Write);
            const bool timeout = elapsed_timeout || transport_error == httplib::Error::ConnectionTimeout ||
                transport_error == httplib::Error::Timeout;
            return fail(std::string(timeout ? "HTTP inference timeout: " : "HTTP inference transport failure: ") +
                httplib::to_string(transport_error));
        }
        if (response->status >= 200 && response->status < 300) trace_payload("response", response->body);
        if (request.options.should_stop && request.options.should_stop()) {
            result.status = common_agent_generation_status::cancelled;
            result.stop_reason = common_agent_generation_stop_reason::cancelled;
            result.error_message = "HTTP inference request cancelled while awaiting the remote server";
            return true;
        }
        if (response->status < 200 || response->status >= 300) {
            trace_payload("error response", response->body);
            std::string detail = "HTTP inference server returned status " + std::to_string(response->status);
            try {
                const auto error_body = json::parse(response->body);
                if (error_body.contains("error") && error_body["error"].is_object() &&
                        error_body["error"].contains("message") && error_body["error"]["message"].is_string()) {
                    const std::string server_message = error_body["error"]["message"].get<std::string>();
                    std::string normalized_message = server_message;
                    std::transform(normalized_message.begin(), normalized_message.end(), normalized_message.begin(),
                        [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
                    detail += normalized_message.find("context length") != std::string::npos ||
                        normalized_message.find("context window") != std::string::npos ||
                        normalized_message.find("context limit") != std::string::npos ||
                        normalized_message.find("token limit") != std::string::npos ||
                        normalized_message.find("too many tokens") != std::string::npos
                        ? " (server reported a context limit)" : " (server rejected the request)";
                }
            } catch (...) {}
            return fail(detail);
        }
        json payload;
        try { payload = json::parse(response->body); }
        catch (...) { return fail("HTTP inference server returned invalid JSON"); }
        if (!payload.contains("choices") || !payload["choices"].is_array() || payload["choices"].empty() ||
                !payload["choices"][0].is_object() || !payload["choices"][0].contains("message") ||
                !payload["choices"][0]["message"].is_object()) return fail("HTTP inference response is missing choices[0].message");
        const auto & choice = payload["choices"][0];
        const auto & message = choice["message"];
        if (!message.contains("role") || !message["role"].is_string() || message["role"] != "assistant")
            return fail("HTTP inference response message is missing assistant role");
        if (message.contains("refusal") && message["refusal"].is_string() && !message["refusal"].get<std::string>().empty())
            return fail("remote model refused the generation request");
        if (message.contains("content") && !message["content"].is_null()) {
            if (!message["content"].is_string()) return fail("HTTP inference response message.content has an unsupported type");
            result.content = message["content"].get<std::string>();
        }
        common_chat_msg assistant;
        assistant.role = "assistant";
        assistant.content = result.content;
        if (message.contains("tool_calls") && !message["tool_calls"].is_null()) {
            if (!message["tool_calls"].is_array()) return fail("HTTP inference response tool_calls must be an array");
            for (const auto & call : message["tool_calls"]) {
                if (!call.is_object() || !call.contains("id") || !call["id"].is_string() ||
                        !call.contains("function") || !call["function"].is_object() ||
                        !call["function"].contains("name") || !call["function"]["name"].is_string() ||
                        !call["function"].contains("arguments")) return fail("HTTP inference response contains a malformed tool call");
                if ((call.contains("type") && call["type"] != "function") ||
                        call["id"].get<std::string>().empty() ||
                        call["function"]["name"].get<std::string>().empty())
                    return fail("HTTP inference response contains an invalid tool-call identity or type");
                std::string arguments;
                const auto & args = call["function"]["arguments"];
                if (args.is_string()) arguments = args.get<std::string>();
                else if (args.is_object() || args.is_array()) arguments = args.dump();
                else return fail("HTTP inference tool call arguments are not a JSON object or string");
                const auto parsed = json::parse(arguments, nullptr, false);
                if (!parsed.is_object()) return fail("HTTP inference tool-call arguments are invalid JSON object data");
                auto name = call["function"]["name"].get<std::string>();
                if (selection_.map_tool_names) {
                    const auto found = internal_names.find(name);
                    if (found == internal_names.end()) return fail("HTTP inference returned an undeclared tool name");
                    name = found->second;
                }
                assistant.tool_calls.push_back({std::move(name), arguments, call["id"].get<std::string>()});
            }
        }
        if (!assistant.tool_calls.empty() &&
                (request.tools.empty() || request.tool_choice == COMMON_CHAT_TOOL_CHOICE_NONE))
            return fail("HTTP inference returned tool calls for a request that did not allow them");
        if ((!message.contains("content") || message["content"].is_null()) && assistant.tool_calls.empty())
            return fail("HTTP inference response message is missing content and tool calls");
        if (payload.contains("usage") && payload["usage"].is_object() &&
                payload["usage"].contains("completion_tokens") && payload["usage"]["completion_tokens"].is_number_integer())
            result.decoded_tokens = std::max(0, payload["usage"]["completion_tokens"].get<int>());
        if (!choice.contains("finish_reason") || !choice["finish_reason"].is_string())
            return fail("HTTP inference response is missing a valid finish_reason");
        const std::string finish = choice["finish_reason"].get<std::string>();
        if (finish == "content_filter") return fail("HTTP inference response was blocked by the remote content filter");
        if (finish != "stop" && finish != "length" && finish != "tool_calls")
            return fail("HTTP inference response has an unsupported finish_reason");
        if (!request.json_schema.empty() && (!assistant.tool_calls.empty() || result.content.empty()))
            return fail("HTTP inference response rejected by requested JSON schema: expected JSON content without tool calls");
        if (!request.json_schema.empty()) {
            std::string normalized;
            if (!common_schema_normalize_and_validate_object(result.content, response_schema.dump(), normalized, error)) {
                result.output_schema_rejected = true;
                return fail("HTTP inference response rejected by requested JSON schema: " + error);
            }
            if (wrapped_response_value) {
                const auto wrapped = json::parse(normalized, nullptr, false);
                if (!wrapped.is_object() || !wrapped.contains("value"))
                    return fail("HTTP inference response is missing the wrapped schema value");
                result.content = wrapped["value"].dump();
            } else {
                result.content = std::move(normalized);
            }
            assistant.content = result.content;
        }
        result.assistant_message = std::move(assistant);
        result.status = common_agent_generation_status::completed;
        if (finish == "length") result.stop_reason = common_agent_generation_stop_reason::limit;
        else if (finish == "tool_calls") result.stop_reason = common_agent_generation_stop_reason::eos;
        else if (finish == "stop") result.stop_reason = request.json_schema.empty()
            ? common_agent_generation_stop_reason::eos : common_agent_generation_stop_reason::json_schema;
        else result.stop_reason = common_agent_generation_stop_reason::none;
        return true;
    }

    bool score_teacher_forced_choice(
            const common_agent_teacher_forced_choice_request & request,
            common_agent_teacher_forced_choice_result & result) override {
        result = {};
        result.sequence_id = request.sequence_id;
        result.error_message = "HTTP inference backend does not support teacher-forced scoring";
        return false;
    }

private:
    common_agent_model_selection selection_;
    std::string token_;
};
}

std::unique_ptr<common_agent_inference> make_agent_http_inference(
        const common_agent_model_selection & selection) {
    return std::make_unique<http_inference>(selection);
}
