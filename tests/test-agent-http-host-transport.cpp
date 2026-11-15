#include "tools/agent/runtime/agent-http-inference.h"
#include <nlohmann/json.hpp>
#include <iostream>

#define REQUIRE(expr) do { if (!(expr)) { std::cerr << "contract failure at line " << __LINE__ << '\n'; return 1; } } while (0)

int main() {
    common_agent_model_selection profile;
    profile.endpoint = "https://example.invalid/base/"; profile.model_id = "model";
    profile.context_size_tokens = 8192; profile.supports_json_schema = true;
    int calls = 0; bool reject = false;
    auto inference = make_agent_http_inference(profile, [&](const std::string & url, const std::string & body,
            uint32_t connect_timeout, uint32_t request_timeout, int & status, std::string & response, std::string &) {
        ++calls;
        const auto request = nlohmann::json::parse(body);
        if (url != "https://example.invalid/base/v1/chat/completions" || request.value("model", "") != "model" ||
                connect_timeout == 0 || request_timeout == 0) return false;
        status = reject ? 403 : 200;
        response = reject ? R"({"error":{"message":"denied"}})" :
            R"({"choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]})";
        return true;
    });
    common_agent_generation_request request;
    request.purpose = common_agent_generation_purpose::conversation;
    request.messages.push_back({"user", "hello"});
    request.options.n_predict = 16;
    common_agent_generation_result result;
    REQUIRE(inference->generate(request, result));
    REQUIRE(result.status == common_agent_generation_status::completed && result.content == "ok" && calls == 1);
    reject = true;
    REQUIRE(inference->generate(request, result));
    REQUIRE(result.status == common_agent_generation_status::errored && result.error_message.find("403") != std::string::npos);
    request.options.should_stop = [] { return true; };
    REQUIRE(inference->generate(request, result));
    REQUIRE(result.status == common_agent_generation_status::cancelled && calls == 2);
    return 0;
}
