#include <nlohmann/json.hpp>

#include <cstdio>
#include <cstring>
#include <string>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

using json = nlohmann::ordered_json;

namespace {

bool read_header_line(FILE * stream, std::string & line) {
    line.clear();
    for (;;) {
        const int ch = std::fgetc(stream);
        if (ch == EOF) {
            return !line.empty();
        }
        if (ch == '\n') {
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
            return true;
        }
        line.push_back(static_cast<char>(ch));
    }
}

bool read_json_rpc_message(json & message) {
    message = json();

    size_t content_length = 0;
    bool have_content_length = false;
    std::string line;
    while (read_header_line(stdin, line)) {
        if (line.empty()) {
            break;
        }

        constexpr const char * prefix = "Content-Length:";
        if (line.rfind(prefix, 0) == 0) {
            std::string value = line.substr(std::strlen(prefix));
            while (!value.empty() && value.front() == ' ') {
                value.erase(value.begin());
            }
            content_length = static_cast<size_t>(std::strtoull(value.c_str(), nullptr, 10));
            have_content_length = true;
        }
    }

    if (!have_content_length) {
        return false;
    }

    std::string body(content_length, '\0');
    if (std::fread(body.data(), 1, content_length, stdin) != content_length) {
        return false;
    }

    const auto parsed = json::parse(body, nullptr, false);
    if (parsed.is_discarded() || !parsed.is_object()) {
        return false;
    }

    message = parsed;
    return true;
}

bool write_json_rpc_message(const json & message) {
    const std::string body = message.dump();
    const std::string framed =
        "Content-Length: " + std::to_string(body.size()) + "\r\n\r\n" + body;
    if (std::fwrite(framed.data(), 1, framed.size(), stdout) != framed.size()) {
        return false;
    }
    return std::fflush(stdout) == 0;
}

json make_tools_list_result() {
    return {
        {"tools", json::array({
            {
                {"name", "search_issues"},
                {"description", "Search GitHub issues."},
                {"inputSchema", {
                    {"type", "object"},
                    {"required", json::array({"query"})},
                }},
                {"annotations", {
                    {"readOnlyHint", true},
                }},
                {"hostPolicy", {
                    {"usesNetwork", true},
                }},
            },
            {
                {"name", "search_recent_failures"},
                {"description", "Simulate a retryable upstream failure."},
                {"inputSchema", {
                    {"type", "object"},
                    {"required", json::array({"query"})},
                }},
                {"annotations", {
                    {"readOnlyHint", true},
                }},
                {"hostPolicy", {
                    {"usesNetwork", true},
                }},
            },
            {
                {"name", "create_issue"},
                {"description", "Create a GitHub issue."},
                {"inputSchema", {
                    {"type", "object"},
                    {"required", json::array({"title"})},
                }},
                {"annotations", {
                    {"readOnlyHint", false},
                }},
                {"hostPolicy", {
                    {"usesNetwork", true},
                    {"requiresConfirmation", true},
                }},
            },
        })},
    };
}

json make_tools_call_result(const json & params) {
    const std::string name = params.value("name", "");
    const auto arguments = params.value("arguments", json::object());

    if (name == "search_issues") {
        return {
            {"content", json::array({
                {{"type", "text"}, {"text", "stub issue"}},
            })},
            {"structuredContent", {
                {"items", json::array({
                    {{"title", "stub issue"}},
                })},
            }},
        };
    }

    if (name == "create_issue") {
        return {
            {"content", json::array({
                {{"type", "text"}, {"text", "created issue #321: " + arguments.value("title", std::string())}},
            })},
        };
    }

    if (name == "search_recent_failures") {
        return {
            {"isError", true},
            {"content", json::array({
                {{"type", "text"}, {"text", "upstream search provider is rate limited"}},
            })},
            {"errorInfo", {
                {"code", "github.rate_limited"},
                {"class", "network"},
                {"retryable", true},
                {"safeSummary", "The upstream GitHub search provider is temporarily rate limited."},
            }},
        };
    }

    return {
        {"isError", true},
        {"content", json::array({
            {{"type", "text"}, {"text", "unknown tool"}},
        })},
    };
}

} // namespace

int main() {
#ifdef _WIN32
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
#endif
    bool shutdown_requested = false;
    for (;;) {
        json message;
        if (!read_json_rpc_message(message)) {
            return 0;
        }

        const std::string method = message.value("method", "");
        if (method == "exit") {
            return shutdown_requested ? 0 : 1;
        }
        if (!message.contains("id")) {
            continue;
        }

        json response = {
            {"jsonrpc", "2.0"},
            {"id", message["id"]},
        };

        if (method == "initialize") {
            response["result"] = {
                {"protocolVersion", "2024-11-05"},
                {"capabilities", {
                    {"tools", json::object()},
                }},
                {"serverInfo", {
                    {"name", "fake-mcp"},
                    {"version", "0.1"},
                }},
            };
        } else if (method == "shutdown") {
            shutdown_requested = true;
            response["result"] = json::object();
        } else if (method == "tools/list") {
            response["result"] = make_tools_list_result();
        } else if (method == "tools/call") {
            response["result"] = make_tools_call_result(message.value("params", json::object()));
        } else {
            response["error"] = {
                {"code", -32601},
                {"message", "method not found"},
            };
        }

        if (!write_json_rpc_message(response)) {
            return 1;
        }
    }
}
