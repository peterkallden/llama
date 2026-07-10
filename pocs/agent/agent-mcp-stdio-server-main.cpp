#include "agent-mcp-stdio-server.h"

#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <string>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

namespace {

class calculator_parser {
public:
    explicit calculator_parser(const std::string & text) : text_(text) {}

    bool parse(double & value, std::string & error) {
        value = expression(error);
        skip_space();
        if (!error.empty()) return false;
        if (pos_ != text_.size() || !std::isfinite(value)) {
            error = "invalid bounded arithmetic expression";
            return false;
        }
        return true;
    }

private:
    double expression(std::string & error) {
        double value = term(error);
        while (error.empty()) {
            skip_space();
            if (take('+')) value += term(error);
            else if (take('-')) value -= term(error);
            else break;
        }
        return value;
    }

    double term(std::string & error) {
        double value = factor(error);
        while (error.empty()) {
            skip_space();
            if (take('*')) value *= factor(error);
            else if (take('/')) {
                const double divisor = factor(error);
                if (error.empty() && divisor == 0.0) error = "division by zero";
                else value /= divisor;
            } else break;
        }
        return value;
    }

    double factor(std::string & error) {
        skip_space();
        if (take('+')) return factor(error);
        if (take('-')) return -factor(error);
        if (take('(')) {
            const double value = expression(error);
            skip_space();
            if (error.empty() && !take(')')) error = "missing closing parenthesis";
            return value;
        }
        const size_t begin = pos_;
        bool digit = false;
        while (pos_ < text_.size() && (std::isdigit((unsigned char) text_[pos_]) || text_[pos_] == '.')) {
            digit = digit || std::isdigit((unsigned char) text_[pos_]);
            ++pos_;
        }
        if (!digit || pos_ - begin > 64) {
            error = "expected a number";
            return 0.0;
        }
        try {
            return std::stod(text_.substr(begin, pos_ - begin));
        } catch (...) {
            error = "invalid number";
            return 0.0;
        }
    }

    void skip_space() {
        while (pos_ < text_.size() && std::isspace((unsigned char) text_[pos_])) ++pos_;
    }

    bool take(char c) {
        if (pos_ < text_.size() && text_[pos_] == c) {
            ++pos_;
            return true;
        }
        return false;
    }

    const std::string & text_;
    size_t pos_ = 0;
};

std::string utc_now() {
    const auto now = std::chrono::system_clock::now();
    const auto time = std::chrono::system_clock::to_time_t(now);
    std::tm utc{};
#ifdef _WIN32
    gmtime_s(&utc, &time);
#else
    gmtime_r(&time, &utc);
#endif
    std::ostringstream out;
    out << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
    return out.str();
}

bool add_text_content(
        agent_mcp_server_tool_result & result,
        std::string text) {
    result.content = agent_mcp_json::array({
        {
            {"type", "text"},
            {"text", std::move(text)},
        },
    });
    return true;
}

bool register_builtin_tools(
        agent_mcp_server_tool_registry & registry,
        std::string & error) {
    if (!registry.register_tool({
            "echo",
            "Echo bounded input text.",
            R"({"type":"object","additionalProperties":false,"required":["text"],"properties":{"text":{"type":"string","minLength":1,"maxLength":1024}}})",
            true,
            false,
            false,
            false,
            false,
            [](const agent_mcp_json & arguments, agent_mcp_server_tool_result & result, std::string & call_error) {
                if (!arguments.contains("text") || !arguments["text"].is_string()) {
                    call_error = "echo requires a bounded text string";
                    return false;
                }
                const auto text = arguments["text"].get<std::string>();
                if (text.empty() || text.size() > 1024) {
                    call_error = "echo text is out of bounds";
                    return false;
                }
                result.ok = true;
                result.structured_content = {
                    {"ok", true},
                    {"result", {
                        {"text", text},
                    }},
                };
                result.safe_summary = "Echoed bounded input text.";
                return add_text_content(result, text);
            },
        }, error)) {
        return false;
    }

    if (!registry.register_tool({
            "time_now",
            "Return the current UTC time.",
            R"({"type":"object","additionalProperties":false})",
            true,
            false,
            false,
            false,
            false,
            [](const agent_mcp_json & arguments, agent_mcp_server_tool_result & result, std::string & call_error) {
                if (!arguments.empty()) {
                    call_error = "time_now does not take arguments";
                    return false;
                }
                const auto now = utc_now();
                result.ok = true;
                result.structured_content = {
                    {"ok", true},
                    {"result", {
                        {"utc", now},
                    }},
                };
                result.safe_summary = "Returned the current UTC time.";
                return add_text_content(result, now);
            },
        }, error)) {
        return false;
    }

    if (!registry.register_tool({
            "calculator",
            "Evaluate a bounded arithmetic expression.",
            R"({"type":"object","additionalProperties":false,"required":["expression"],"properties":{"expression":{"type":"string","minLength":1,"maxLength":128}}})",
            true,
            false,
            false,
            false,
            false,
            [](const agent_mcp_json & arguments, agent_mcp_server_tool_result & result, std::string & call_error) {
                if (!arguments.contains("expression") || !arguments["expression"].is_string()) {
                    call_error = "calculator requires a bounded expression string";
                    return false;
                }
                const auto expression = arguments["expression"].get<std::string>();
                if (expression.empty() || expression.size() > 128) {
                    call_error = "calculator expression is out of bounds";
                    return false;
                }
                double value = 0.0;
                calculator_parser parser(expression);
                if (!parser.parse(value, call_error)) {
                    return false;
                }
                result.ok = true;
                result.structured_content = {
                    {"ok", true},
                    {"result", {
                        {"value", value},
                    }},
                };
                result.safe_summary = "Evaluated a bounded arithmetic expression.";
                return add_text_content(result, std::to_string(value));
            },
        }, error)) {
        return false;
    }

    error.clear();
    return true;
}

} // namespace

int main() {
#ifdef _WIN32
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
#endif

    std::string error;
    agent_mcp_server_tool_registry registry;
    if (!register_builtin_tools(registry, error)) {
        std::fprintf(stderr, "failed to register MCP stdio server tools: %s\n", error.c_str());
        return 1;
    }

    agent_mcp_stdio_server server(
        std::move(registry),
        {
            "llama-agent-mcp-stdio-server",
            "0.1",
            "2024-11-05",
            false,
            false,
        });
    return server.run(stdin, stdout, stderr);
}
