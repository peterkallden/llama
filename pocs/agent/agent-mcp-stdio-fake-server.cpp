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
                {
                    {"type", "resource_link"},
                    {"uri", "mcp-resource://github/search_issues/stub-1"},
                    {"name", "search-results.json"},
                    {"description", "Full GitHub issue search result set"},
                    {"mimeType", "application/json"},
                    {"sizeBytes", 128},
                },
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

int main(int argc, char ** argv) {
#ifdef _WIN32
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
#endif
    std::string mode;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
            mode = argv[++i];
        }
    }

    if (mode == "crash-before-initialize") {
        std::fprintf(stderr, "fake-mcp: crash-before-initialize\n");
        return 7;
    }

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
            if (mode == "exit-after-initialize") {
                std::fprintf(stderr, "fake-mcp: exiting after initialize\n");
                if (!write_json_rpc_message(response)) {
                    return 1;
                }
                return 9;
            }
        } else if (method == "shutdown") {
            shutdown_requested = true;
            response["result"] = json::object();
        } else if (method == "tools/list") {
            if (mode == "bad-tools-list") {
                std::fprintf(stderr, "fake-mcp: emitting malformed tools/list payload\n");
                const std::string body = "{\"jsonrpc\":\"2.0\",\"id\":1,\"result\":";
                const std::string framed =
                    "Content-Length: " + std::to_string(body.size()) + "\r\n\r\n" + body;
                std::fwrite(framed.data(), 1, framed.size(), stdout);
                std::fflush(stdout);
                return 11;
            }
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
