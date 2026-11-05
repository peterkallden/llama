#include "tools/agent/runtime/agent-http-inference.h"
#include "tools/agent/runtime/agent-model-loaders.h"
#include "tools/agent/runtime/agent-runtime-session.h"
#include "tools/agent/runtime/agent-runtime-chat-driver.h"
#include "tools/agent/tooling/agent-tool-provider.h"

#include <cpp-httplib/httplib.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <thread>
#include <unordered_map>

#undef assert
#define assert(condition) do { \
    if (!(condition)) { \
        std::cerr << "HTTP inference smoke assertion failed at " << __FILE__ << ':' << __LINE__ \
                  << ": " #condition << '\n'; \
        std::abort(); \
    } \
} while (false)

using json = nlohmann::ordered_json;

class fixture_tool_view final : public agent_tool_view {
public:
    explicit fixture_tool_view(common_chat_tool tool) : tools{std::move(tool)} {}
    const std::vector<common_chat_tool> & chat_tools() const override { return tools; }
    bool exposes_tool(const std::string & name) const override { return name == tools.front().name; }
    bool is_read_only(const std::string &) const override { return true; }
    bool is_policy_gated(const std::string &) const override { return false; }
    bool validate(const agent_tool_call & call, std::string & error) const override {
        if (!exposes_tool(call.name)) { error = "tool is not authorized"; return false; }
        error.clear(); return true;
    }
    agent_tool_result call(const agent_tool_call & call, std::string & error) override {
        if (!validate(call, error)) return {};
        assert(call.name == "memory.search");
        assert(call.arguments_json.find("project status") != std::string::npos);
        ++calls;
        agent_tool_result result;
        result.ok = true;
        result.tool_call_id = call.id;
        result.tool_name = call.name;
        result.content_json = R"({"ok":true,"result":{"items":[{"id":"note-1"}]}})";
        error.clear();
        return result;
    }
    bool supports_async_call(const std::string &) const override { return false; }
    bool begin_call_async(const agent_tool_call &, agent_tool_pending_call &, std::string & error) override { error = "unsupported"; return false; }
    bool poll_call_async(const agent_tool_pending_call &, bool &, agent_tool_result &, std::string & error) override { error = "unsupported"; return false; }
    bool cancel_call_async(const agent_tool_pending_call &, std::string & error) override { error = "unsupported"; return false; }
    std::vector<common_chat_tool> tools;
    size_t calls = 0;
};

int main() {
    agent_inference_backend parsed_backend = agent_inference_backend::cli;
    assert(parse_agent_inference_backend("http", parsed_backend));
    assert(parsed_backend == agent_inference_backend::http);
    httplib::Server server;
    json seen;
    std::string received_auth;
    std::mutex seen_mutex;
    server.Post("/v1/chat/completions", [&](const httplib::Request & request, httplib::Response & response) {
        const json body = json::parse(request.body);
        { std::lock_guard<std::mutex> lock(seen_mutex); seen = body; received_auth = request.get_header_value("Authorization"); }
        const std::string prompt = body["messages"].back().value("content", std::string{});
        if (prompt == "dsl-fixture") {
            response.set_content(R"({"choices":[{"finish_reason":"stop","message":{"role":"assistant","content":"open! memory.search\nquery: project status"}}]})", "application/json");
            return;
        }
        if (prompt == "missing-choice") { response.set_content("{}", "application/json"); return; }
        if (prompt == "malformed-json") { response.set_content("{broken", "application/json"); return; }
        if (prompt == "slow-response") std::this_thread::sleep_for(std::chrono::milliseconds(100));
        if (prompt == "context-error") {
            response.status = 400;
            response.set_content(R"({"error":{"message":"maximum context length exceeded"}})", "application/json");
            return;
        }
        if (prompt == "schema-context-error") {
            response.status = 400;
            response.set_content(R"({"error":{"message":"Invalid schema: In context=('properties','args'), additionalProperties must be false"}})", "application/json");
            return;
        }
        if (body.contains("response_format")) {
            if (prompt == "scalar-schema" || prompt == "scalar-schema-invalid") {
                const auto value = prompt == "scalar-schema" ? "openalex.listWorks" : "not.allowed";
                response.set_content(json{{"choices", json::array({{{"finish_reason", "stop"},
                    {"message", {{"role", "assistant"}, {"content", json{{"value", value}}.dump()}}}}})}}.dump(), "application/json");
                return;
            }
            if (prompt == "invalid-schema") {
                response.set_content(R"({"choices":[{"finish_reason":"stop","message":{"role":"assistant","content":"{\"wrong\":\"fixture\"}"}}]})", "application/json");
                return;
            }
            response.set_content(R"({"choices":[{"finish_reason":"stop","message":{"role":"assistant","content":"{\"family\":\"memory\",\"tool_name\":\"memory.search\"}"}}],"usage":{"completion_tokens":3}})", "application/json");
            return;
        }
        if (prompt == "mapped-tools") {
            response.set_content(R"({"choices":[{"finish_reason":"tool_calls","message":{"role":"assistant","tool_calls":[{"id":"mapped-1","type":"function","function":{"name":"memory_2esearch","arguments":"{\"query\":\"test\"}"}}]}}]})", "application/json");
            return;
        }
        if (std::any_of(body["messages"].begin(), body["messages"].end(), [](const json & message) {
                return message.value("role", std::string{}) == "tool";
            })) {
            response.set_content(R"({"choices":[{"finish_reason":"stop","message":{"role":"assistant","content":"Local memory result was returned."}}],"usage":{"completion_tokens":5}})", "application/json");
            return;
        }
        if (prompt == "unauthorized-tool") {
            response.set_content(R"({"choices":[{"finish_reason":"tool_calls","message":{"role":"assistant","tool_calls":[{"id":"call-bad","type":"function","function":{"name":"secret.delete","arguments":"{}"}}]}}]})", "application/json");
            return;
        }
        if (prompt == "multiple-tools") {
            response.set_content(R"({"choices":[{"finish_reason":"tool_calls","message":{"role":"assistant","tool_calls":[{"id":"call-1","type":"function","function":{"name":"memory.search","arguments":"{\"query\":\"one\"}"}},{"id":"call-2","type":"function","function":{"name":"memory.search","arguments":"{\"query\":\"two\"}"}}]}}]})", "application/json");
            return;
        }
        response.set_content(R"({"choices":[{"finish_reason":"tool_calls","message":{"role":"assistant","content":null,"tool_calls":[{"id":"call-1","type":"function","function":{"name":"memory.search","arguments":"{\"query\":\"project status\"}"}}]}}],"usage":{"completion_tokens":7}})", "application/json");
    });
    const int port = server.bind_to_any_port("127.0.0.1");
    assert(port > 0);
    std::thread server_thread([&] { server.listen_after_bind(); });
    struct server_guard { httplib::Server & server; std::thread & thread; ~server_guard() { server.stop(); if (thread.joinable()) thread.join(); } } guard{server, server_thread};
    for (int i = 0; i < 100 && !server.is_running(); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(5));

    common_agent_model_selection selection;
    selection.backend = "http";
    selection.endpoint = "http://127.0.0.1:" + std::to_string(port);
    selection.model_id = "fixture-model";
    selection.token_env = "LLAMA_AGENT_HTTP_SMOKE_TOKEN";
#ifdef _WIN32
    _putenv_s("LLAMA_AGENT_HTTP_SMOKE_TOKEN", "smoke-secret");
#else
    setenv("LLAMA_AGENT_HTTP_SMOKE_TOKEN", "smoke-secret", 1);
#endif
    selection.context_size_tokens = 8192;
    selection.supports_native_tool_calls = true;
    selection.connect_timeout_ms = 500;
    selection.request_timeout_ms = 1000;
    auto inference = make_agent_http_inference(selection);
    assert(inference->context_capacity_tokens() == 8192);
    assert(!inference->count_context_text_tokens("text").has_value());
    common_agent_teacher_forced_choice_request scoring_request;
    common_agent_teacher_forced_choice_result scoring_result;
    assert(!inference->score_teacher_forced_choice(scoring_request, scoring_result));
    assert(scoring_result.error_message.find("does not support teacher-forced scoring") != std::string::npos);

    common_agent_generation_request request;
    request.messages = {{"user", "Search the project memory"}};
    request.tools.emplace_back("memory.search", "Search saved project notes", R"({"type":"object","required":["query"],"properties":{"query":{"type":"string"}}})");
    request.tool_choice = COMMON_CHAT_TOOL_CHOICE_REQUIRED;
    request.options.n_predict = 72;
    common_agent_generation_result result;
    assert(inference->generate(request, result));
    assert(result.status == common_agent_generation_status::completed);
    assert(result.stop_reason == common_agent_generation_stop_reason::eos);
    assert(result.decoded_tokens == 7);
    assert(result.assistant_message && result.assistant_message->tool_calls.size() == 1);
    assert(result.assistant_message->tool_calls.front().name == "memory.search");
    {
        std::lock_guard<std::mutex> lock(seen_mutex);
        assert(seen["model"] == "fixture-model");
        assert(seen["max_tokens"] == 72);
        assert(!seen.contains("max_completion_tokens") && !seen.contains("reasoning_effort"));
        assert(seen["stream"] == false);
        assert(seen["tool_choice"] == "required");
        assert(seen["messages"].size() == 1 && seen["tools"].size() == 1);
        assert(received_auth == "Bearer smoke-secret");
    }

    {
        auto mapped_selection = selection;
        mapped_selection.map_tool_names = true;
        mapped_selection.token_limit_parameter = "max_completion_tokens";
        mapped_selection.reasoning_effort = "none";
        auto mapped_inference = make_agent_http_inference(mapped_selection);
        common_agent_generation_request mapped;
        common_chat_msg prior;
        prior.role = "assistant";
        prior.tool_calls.push_back({"memory.search", R"({"query":"old"})", "old-call"});
        common_chat_msg prior_result;
        prior_result.role = "tool";
        prior_result.tool_name = "memory.search";
        prior_result.tool_call_id = "old-call";
        prior_result.content = "result";
        mapped.messages = {prior, prior_result, {"user", "mapped-tools"}};
        mapped.tools.emplace_back("memory.search", "Search", R"({"type":"object"})");
        mapped.tools.emplace_back("memory_search", "Other search", R"({"type":"object"})");
        mapped.tool_choice = COMMON_CHAT_TOOL_CHOICE_REQUIRED;
        mapped.options.n_predict = 72;
        assert(mapped_inference->generate(mapped, result));
        assert(result.status == common_agent_generation_status::completed);
        assert(result.assistant_message && result.assistant_message->tool_calls.front().name == "memory.search");
        {
            std::lock_guard<std::mutex> lock(seen_mutex);
            assert(seen["tools"][0]["function"]["name"] == "memory_2esearch");
            assert(seen["tools"][1]["function"]["name"] == "memory_5fsearch");
            assert(seen["messages"][0]["tool_calls"][0]["function"]["name"] == "memory_2esearch");
            assert(seen["max_completion_tokens"] == 72 && !seen.contains("max_tokens"));
            assert(seen["reasoning_effort"] == "none");
        }
        mapped.messages = {{"user", "unauthorized-tool"}};
        assert(mapped_inference->generate(mapped, result));
        assert(result.status == common_agent_generation_status::errored);
        assert(result.error_message.find("undeclared tool name") != std::string::npos);
        mapped.tools[0].name = std::string(65, 'a');
        assert(mapped_inference->generate(mapped, result));
        assert(result.status == common_agent_generation_status::errored);
    }

    common_agent_model_catalog runtime_catalog;
    common_agent_model_base_spec runtime_base;
    runtime_base.backend = "http";
    runtime_base.endpoint = selection.endpoint;
    runtime_base.model_id = selection.model_id;
    runtime_base.token_env = selection.token_env;
    runtime_base.supports_json_schema = true;
    runtime_base.supports_native_tool_calls = true;
    runtime_catalog.bases.emplace("fixture-base", runtime_base);
    common_agent_model_profile_spec runtime_profile;
    runtime_profile.base_model_id = "fixture-base";
    runtime_profile.context_size_tokens = selection.context_size_tokens;
    runtime_catalog.profiles.emplace("fixture-profile", runtime_profile);
    runtime_catalog.default_profile = "fixture-profile";
    std::unordered_map<std::string, std::shared_ptr<common_agent_runtime_model_loader>> loaders;
    loaders.emplace("http", std::make_shared<common_agent_runtime_http_model_loader>());
    common_agent_runtime_model_residency residency(runtime_catalog, std::move(loaders));
    common_agent_runtime_model_resident_handle resident_handle;
    std::string runtime_error;
    assert(residency.acquire("fixture-profile", resident_handle, runtime_error));
    agent_inference_backend runtime_backend = agent_inference_backend::cli;
    assert(parse_agent_inference_backend("http", runtime_backend));
    common_agent_inference_options runtime_options;
    runtime_options.context_size_tokens = selection.context_size_tokens;
    common_agent_runtime_session runtime_session;
    assert(initialize_agent_runtime_session_from_resident_model(
        runtime_options, runtime_backend, resident_handle.model, runtime_session, runtime_error));
    assert(runtime_session.active_inference_session() != nullptr);
    assert(runtime_session.active_inference_session()->backend == agent_inference_backend::http);
    auto & runtime_inference = *runtime_session.active_inference_session()->inference;

    fixture_tool_view tool_view(request.tools.front());
    common_agent_runtime_tooling tooling;
    tooling.tools = request.tools;
    tooling.tool_view = &tool_view;
    common_agent_request chat_request;
    chat_request.prompt = "Check project status";
    chat_request.tool_output_format = common_agent_tool_output_format::native;
    chat_request.messages = {{"user", chat_request.prompt}};
    chat_request.session_id = "http-smoke-session";
    chat_request.turn_id = "http-smoke-turn";
    common_agent_generation_options generation_options;
    generation_options.n_predict = 64;
    common_agent_chat_runtime_execution execution{
        runtime_inference,
        chat_request,
        generation_options,
        {4, 1, false},
        tooling,
    };
    common_agent_result runtime_result;
    const bool runtime_ok = run_agent_chat_runtime(execution, runtime_result, runtime_error);
    if (!runtime_ok) std::cerr << "HTTP runtime fixture failed: " << runtime_error << '\n';
    assert(runtime_ok);
    assert(runtime_error.empty());
    assert(tool_view.calls == 1);
    assert(runtime_result.response == "Local memory result was returned.");
    chat_request.messages.front().content = "unauthorized-tool";
    chat_request.session_id = "http-unauthorized-session";
    common_agent_chat_runtime_execution unauthorized_execution{
        runtime_inference, chat_request, generation_options, {4, 1, false}, tooling};
    const bool unauthorized_ok = run_agent_chat_runtime(unauthorized_execution, runtime_result, runtime_error);
    assert(!unauthorized_ok);
    assert(tool_view.calls == 1);
    assert(runtime_error.find("not exposed") != std::string::npos || runtime_error.find("not authorized") != std::string::npos);
    chat_request.messages.front().content = "multiple-tools";
    chat_request.session_id = "http-multiple-session";
    common_agent_chat_runtime_execution multiple_execution{
        runtime_inference, chat_request, generation_options, {4, 1, false}, tooling};
    assert(!run_agent_chat_runtime(multiple_execution, runtime_result, runtime_error));
    assert(tool_view.calls == 1);
    assert(runtime_error.find("exceeds configured limit") != std::string::npos);

    request.grammar = "root ::= \"x\"";
    assert(inference->generate(request, result));
    assert(result.status == common_agent_generation_status::errored);
    assert(result.error_message.find("does not support GBNF") != std::string::npos);
    request.grammar.clear();

    // DSL requests for locally validated purposes use text when HTTP cannot
    // enforce their grammar. Other grammar requests remain unsupported.
    auto dsl_selection = selection;
    dsl_selection.tool_output_format_configured = true;
    dsl_selection.tool_output_format = common_agent_tool_output_format::compact_dsl;
    auto dsl_inference = make_agent_http_inference(dsl_selection);
    common_agent_generation_request dsl_request;
    dsl_request.messages = {{"user", "dsl-fixture"}};
    dsl_request.grammar = "root ::= \"x\"";
    assert(dsl_inference->generate(dsl_request, result));
    assert(result.status == common_agent_generation_status::errored);
    dsl_request.purpose = common_agent_generation_purpose::planner;
    assert(dsl_inference->generate(dsl_request, result));
    assert(result.status == common_agent_generation_status::completed);
    assert(result.content == "open! memory.search\nquery: project status");
    // The inherited DSL default works without a profile override.
    assert(inference->generate(dsl_request, result));
    assert(result.status == common_agent_generation_status::completed);
    auto native_selection = selection;
    native_selection.tool_output_format_configured = true;
    native_selection.tool_output_format = common_agent_tool_output_format::native;
    auto native_inference = make_agent_http_inference(native_selection);
    assert(native_inference->generate(dsl_request, result));
    assert(result.status == common_agent_generation_status::errored);
    dsl_request.json_schema = R"({"type":"object"})";
    assert(dsl_inference->generate(dsl_request, result));
    assert(result.status == common_agent_generation_status::errored);
    { std::lock_guard<std::mutex> lock(seen_mutex);
      assert(!seen.contains("response_format") && !seen.contains("tools")); }

    common_agent_generation_resource image_resource;
    image_resource.resource.mime_type = "image/png";
    request.input_resources.push_back(std::move(image_resource));
    assert(inference->generate(request, result));
    assert(result.status == common_agent_generation_status::errored);
    assert(result.error_message.find("does not support image or audio") != std::string::npos);
    request.input_resources.clear();

    selection.supports_json_schema = true;
    auto schema_inference = make_agent_http_inference(selection);
    request = {};
    request.messages = {{"user", "Choose the memory family and search operation for this request"}};
    request.json_schema = R"({"type":"object","required":["family","tool_name"],"properties":{"family":{"type":"string"},"tool_name":{"type":"string"}},"additionalProperties":false})";
    assert(schema_inference->generate(request, result));
    // Structured JSON output is sent in the server's response_format and then
    // checked locally against the requested schema.
    assert(result.status == common_agent_generation_status::completed);
    assert(result.stop_reason == common_agent_generation_stop_reason::json_schema);
    assert(result.content == R"({"family":"memory","tool_name":"memory.search"})");
    { std::lock_guard<std::mutex> lock(seen_mutex); assert(seen["response_format"]["json_schema"]["strict"] == true); }
    const auto planning = json::parse(result.content);
    assert(planning["family"] == "memory" && planning["tool_name"] == "memory.search");
    auto nonstrict_selection = selection;
    nonstrict_selection.strict_json_schema = false;
    auto nonstrict_inference = make_agent_http_inference(nonstrict_selection);
    assert(nonstrict_inference->generate(request, result));
    assert(result.status == common_agent_generation_status::completed);
    { std::lock_guard<std::mutex> lock(seen_mutex); assert(seen["response_format"]["json_schema"]["strict"] == false); }

    auto no_tool_support_selection = selection;
    no_tool_support_selection.supports_native_tool_calls = false;
    auto no_tool_support = make_agent_http_inference(no_tool_support_selection);
    request = {};
    request.messages = {{"user", "search"}};
    request.tools.emplace_back("memory.search", "Search", R"({"type":"object"})");
    request.tool_choice = COMMON_CHAT_TOOL_CHOICE_REQUIRED;
    assert(no_tool_support->generate(request, result));
    assert(result.status == common_agent_generation_status::errored);
    assert(result.error_message.find("does not declare native tool-call support") != std::string::npos);

    request = {};
    request.messages = {{"user", "scalar-schema"}};
    request.json_schema = R"({"type":"string","enum":["openalex.listWorks","openalex.getWork"]})";
    assert(schema_inference->generate(request, result));
    assert(result.status == common_agent_generation_status::completed);
    assert(result.content == R"("openalex.listWorks")");
    {
        std::lock_guard<std::mutex> lock(seen_mutex);
        const auto & schema = seen["response_format"]["json_schema"]["schema"];
        assert(schema["type"] == "object" && schema["additionalProperties"] == false);
        assert(schema["required"] == json::array({"value"}));
        assert(schema["properties"]["value"]["type"] == "string");
    }
    request.messages.front().content = "scalar-schema-invalid";
    assert(schema_inference->generate(request, result));
    assert(result.status == common_agent_generation_status::errored);
    assert(result.error_message.find("rejected by requested JSON schema") != std::string::npos);
    assert(result.output_schema_rejected);

    selection.supports_json_schema = false;
    auto text_only = make_agent_http_inference(selection);
    request.json_schema = R"({"type":"object"})";
    assert(text_only->generate(request, result));
    assert(result.status == common_agent_generation_status::errored);
    assert(result.error_message.find("does not declare JSON-schema support") != std::string::npos);

    request = {};
    request.messages = {{"user", "invalid-schema"}};
    request.json_schema = R"({"type":"object","required":["answer"],"properties":{"answer":{"type":"string"}},"additionalProperties":false})";
    assert(schema_inference->generate(request, result));
    assert(result.status == common_agent_generation_status::errored);
    assert(result.error_message.find("rejected by requested JSON schema") != std::string::npos);
    assert(result.output_schema_rejected);
    assert(nonstrict_inference->generate(request, result));
    assert(result.status == common_agent_generation_status::errored);
    assert(result.error_message.find("rejected by requested JSON schema") != std::string::npos);
    assert(result.output_schema_rejected);
    request.json_schema.clear();
    request.messages.front().content = "malformed-json";
    assert(inference->generate(request, result));
    assert(result.status == common_agent_generation_status::errored);
    assert(result.error_message.find("invalid JSON") != std::string::npos);
    assert(!result.output_schema_rejected);
    request.messages.front().content = "missing-choice";
    assert(inference->generate(request, result));
    assert(result.status == common_agent_generation_status::errored);
    assert(result.error_message.find("missing choices") != std::string::npos);
    request.messages.front().content = "context-error";
    assert(inference->generate(request, result));
    assert(result.status == common_agent_generation_status::errored);
    assert(result.error_message.find("context limit") != std::string::npos);
    assert(!result.output_schema_rejected);
    request.messages.front().content = "schema-context-error";
    assert(inference->generate(request, result));
    assert(result.status == common_agent_generation_status::errored);
    assert(result.error_message.find("context limit") == std::string::npos);
    assert(result.error_message.find("server rejected") != std::string::npos);
    request.options.should_stop = [] { return true; };
    assert(inference->generate(request, result));
    assert(result.status == common_agent_generation_status::cancelled);
    request.options.should_stop = {};
    request.messages.front().content = "slow-response";
    auto short_timeout_selection = selection;
    short_timeout_selection.request_timeout_ms = 20;
    auto short_timeout_inference = make_agent_http_inference(short_timeout_selection);
    assert(short_timeout_inference->generate(request, result));
    assert(result.status == common_agent_generation_status::errored);
    assert(result.error_message.find("timeout") != std::string::npos);
    auto cancellation_selection = selection;
    cancellation_selection.request_timeout_ms = 1000;
    auto cancellation_inference = make_agent_http_inference(cancellation_selection);
    std::atomic<bool> stop_requested{false};
    request.options.should_stop = [&] { return stop_requested.load(); };
    std::thread cancel_during_request([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        stop_requested.store(true);
    });
    request.messages.front().content = "slow-response";
    assert(cancellation_inference->generate(request, result));
    cancel_during_request.join();
    assert(result.status == common_agent_generation_status::cancelled);
    assert(result.stop_reason == common_agent_generation_stop_reason::cancelled);
    request.options.should_stop = {};

    common_agent_model_catalog catalog;
    catalog.directory.clear();
    common_agent_model_base_spec base;
    base.backend = "http";
    base.endpoint = "http://127.0.0.1:8080";
    base.model_id = "llama-server-model";
    base.token_env = "AGENT_TEST_TOKEN";
    base.supports_native_tool_calls = true;
    base.map_tool_names = true;
    base.token_limit_parameter = "max_completion_tokens";
    base.reasoning_effort = "none";
    base.strict_json_schema = false;
    catalog.bases.emplace("remote", base);
    common_agent_model_profile_spec profile;
    profile.base_model_id = "remote";
    profile.context_size_tokens = 4096;
    catalog.profiles.emplace("remote-default", profile);
    catalog.default_profile = "remote-default";
    const auto serialized = common_agent_model_catalog_to_json(catalog);
    assert(serialized.find("secret-value") == std::string::npos);
    assert(serialized.find("smoke-secret") == std::string::npos);
    assert(!json::parse(serialized)["bases"]["remote"].contains("path"));
    common_agent_model_catalog parsed;
    std::string error;
    assert(common_agent_model_catalog_from_json(serialized, parsed, error));
    common_agent_model_selection resolved;
    assert(common_agent_model_catalog_resolve_profile(parsed, "remote-default", resolved, error));
    assert(resolved.path.empty() && resolved.endpoint == "http://127.0.0.1:8080");
    assert(resolved.context_size_tokens == 4096 && resolved.model_id == "llama-server-model");
    assert(resolved.map_tool_names);
    assert(resolved.token_limit_parameter == "max_completion_tokens" && resolved.reasoning_effort == "none");
    assert(!resolved.strict_json_schema);
    auto different_selection = resolved;
    different_selection.reasoning_effort = "low";
    assert(common_agent_model_selection_cache_key(resolved) != common_agent_model_selection_cache_key(different_selection));
    different_selection = resolved;
    different_selection.strict_json_schema = true;
    assert(common_agent_model_selection_cache_key(resolved) != common_agent_model_selection_cache_key(different_selection));
    assert(common_agent_model_selection_cache_key(resolved).find("secret-value") == std::string::npos);
#ifdef _WIN32
    _putenv_s("LLAMA_AGENT_HTTP_MISSING_TOKEN_FOR_TEST", "");
#else
    unsetenv("LLAMA_AGENT_HTTP_MISSING_TOKEN_FOR_TEST");
#endif
    resolved.token_env = "LLAMA_AGENT_HTTP_MISSING_TOKEN_FOR_TEST";
    common_agent_runtime_http_model_loader http_loader;
    std::shared_ptr<common_agent_runtime_resident_model> loaded_model;
    assert(!http_loader.load(resolved, loaded_model, error));
    assert(error.find("environment variable is unset") != std::string::npos);
    auto invalid_catalog = json::parse(serialized);
    invalid_catalog["bases"]["remote"]["generation"]["strict_json_schema"] = "false";
    assert(!common_agent_model_catalog_from_json(invalid_catalog.dump(), parsed, error));
    invalid_catalog = json::parse(serialized);
    invalid_catalog["bases"]["remote"]["generation"]["token_limit_parameter"] = "arbitrary";
    assert(!common_agent_model_catalog_from_json(invalid_catalog.dump(), parsed, error));
    invalid_catalog = json::parse(serialized);
    invalid_catalog["bases"]["remote"]["generation"]["reasoning_effort"] = "invalid";
    assert(!common_agent_model_catalog_from_json(invalid_catalog.dump(), parsed, error));
    invalid_catalog = json::parse(serialized);
    invalid_catalog["bases"]["remote"]["generation"]["unknown"] = true;
    assert(!common_agent_model_catalog_from_json(invalid_catalog.dump(), parsed, error));
    invalid_catalog = json::parse(serialized);
    invalid_catalog["bases"]["remote"]["endpoint"] = "";
    assert(!common_agent_model_catalog_from_json(invalid_catalog.dump(), parsed, error));
    invalid_catalog = json::parse(serialized);
    invalid_catalog["bases"]["remote"].erase("model_id");
    assert(!common_agent_model_catalog_from_json(invalid_catalog.dump(), parsed, error));
    invalid_catalog = json::parse(serialized);
    invalid_catalog["profiles"]["remote-default"]["context_size"] = 0;
    assert(!common_agent_model_catalog_from_json(invalid_catalog.dump(), parsed, error));

#ifdef _WIN32
    _putenv_s("LLAMA_AGENT_HTTP_SMOKE_TOKEN", "");
#else
    unsetenv("LLAMA_AGENT_HTTP_SMOKE_TOKEN");
#endif
    const char * remote_endpoint = std::getenv("LLAMA_AGENT_HTTP_INFERENCE_URL");
    const char * remote_model = std::getenv("LLAMA_AGENT_HTTP_INFERENCE_MODEL");
    if (remote_endpoint == nullptr || remote_endpoint[0] == '\0' || remote_model == nullptr || remote_model[0] == '\0') {
        std::cout << "HTTP model smoke skipped: set LLAMA_AGENT_HTTP_INFERENCE_URL and LLAMA_AGENT_HTTP_INFERENCE_MODEL.\n";
    } else {
        common_agent_model_selection remote = resolved;
        remote.endpoint = remote_endpoint;
        remote.model_id = remote_model;
        remote.token_env = std::getenv("LLAMA_AGENT_HTTP_INFERENCE_TOKEN_ENV")
            ? std::getenv("LLAMA_AGENT_HTTP_INFERENCE_TOKEN_ENV") : "";
        remote.supports_native_tool_calls = true;
        remote.request_timeout_ms = 180000;
        auto remote_inference = make_agent_http_inference(remote);
        common_agent_generation_request remote_request;
        remote_request.messages = {{"user", "Reply with exactly: remote inference is ready"}};
        remote_request.options.n_predict = 48;
        common_agent_generation_result remote_result;
        if (!remote_inference->generate(remote_request, remote_result) ||
                remote_result.status != common_agent_generation_status::completed) {
            std::cerr << "HTTP model chat failed for endpoint " << remote_endpoint << " model " << remote_model
                      << ": " << remote_result.error_message << '\n';
            return 1;
        }
        remote_request = {};
        remote_request.messages = {{"user", "Call the required tool with query set to remote smoke"}};
        remote_request.tools.emplace_back("memory.search", "Search memory", R"({"type":"object","required":["query"],"properties":{"query":{"type":"string"}},"additionalProperties":false})");
        remote_request.tool_choice = COMMON_CHAT_TOOL_CHOICE_REQUIRED;
        remote_request.options.n_predict = 96;
        if (!remote_inference->generate(remote_request, remote_result) ||
                remote_result.status != common_agent_generation_status::completed ||
                !remote_result.assistant_message || remote_result.assistant_message->tool_calls.size() != 1 ||
                remote_result.assistant_message->tool_calls.front().name != "memory.search") {
            std::cerr << "HTTP model tool-call smoke failed for endpoint " << remote_endpoint << " model " << remote_model
                      << ": " << remote_result.error_message << '\n';
            return 1;
        }
        std::cout << "HTTP model smoke passed: endpoint=" << remote_endpoint << " model=" << remote_model
                  << " (chat and required tool call).\n";
    }

    std::cout << "HTTP inference contract smoke passed (text/tool normalization, schema rejection, capability gates, catalog identity).\n";
    return 0;
}
