#include "agent/tool-adapters.h"

#include "http.h"
#include "memory/memory-retrieval.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <nlohmann/json.hpp>
#include <regex>
#include <set>
#include <sstream>

#ifdef LLAMA_AGENT_WEB_USE_WINHTTP
#include <windows.h>
#include <winhttp.h>
#endif

using json = nlohmann::ordered_json;

namespace {

class calculator_parser {
public:
    explicit calculator_parser(const std::string & text) : text(text) {}

    bool parse(double & value, std::string & error) {
        value = expression(error);
        skip_space();
        if (!error.empty()) return false;
        if (pos != text.size() || !std::isfinite(value)) { error = "invalid bounded arithmetic expression"; return false; }
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
        const size_t begin = pos;
        bool digit = false;
        while (pos < text.size() && (std::isdigit((unsigned char) text[pos]) || text[pos] == '.')) { digit = digit || std::isdigit((unsigned char) text[pos]); ++pos; }
        if (!digit || pos - begin > 64) { error = "expected a number"; return 0.0; }
        try { return std::stod(text.substr(begin, pos - begin)); }
        catch (...) { error = "invalid number"; return 0.0; }
    }
    void skip_space() { while (pos < text.size() && std::isspace((unsigned char) text[pos])) ++pos; }
    bool take(char c) { if (pos < text.size() && text[pos] == c) { ++pos; return true; } return false; }
    const std::string & text;
    size_t pos = 0;
};

bool parse_object(const std::string & arguments_json, json & arguments, std::string & error) {
    arguments = json::parse(arguments_json, nullptr, false);
    if (!arguments.is_object()) { error = "tool arguments must be a JSON object"; return false; }
    return true;
}

json memory_value(const common_memory_record & memory) {
    return {
        {"id", memory.id}, {"kind", common_memory_kind_name(memory.kind)},
        {"content", memory.content}, {"summary", memory.summary},
        {"scope", common_memory_scope_name(memory.scope)}, {"importance", memory.importance},
        {"confidence", memory.confidence}, {"created_at", memory.created_at},
    };
}

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

bool register_definition(const common_tool_definition & definition, common_tool_registry & registry,
        std::function<bool(const std::string &, std::string &, std::string &)> handler, std::string & error, bool read_only = true, bool policy_gated = false) {
    common_registered_tool tool;
    tool.name = definition.name;
    tool.version = definition.version;
    tool.executor_id = definition.executor_id;
    tool.arguments_schema = definition.input_schema_json;
    tool.read_only = read_only;
    tool.policy_gated = policy_gated;
    tool.handler = std::move(handler);
    return registry.register_tool(std::move(tool), error);
}

bool repository_path(const std::string & root, const std::string & relative, std::filesystem::path & out, std::string & error) {
    if (root.empty()) { error = "repository tools require a runtime repository root"; return false; }
    const auto base = std::filesystem::weakly_canonical(root);
    const auto requested = relative.empty() ? base : std::filesystem::weakly_canonical(base / relative);
    const auto base_text = base.generic_string();
    const auto requested_text = requested.generic_string();
    if (requested_text != base_text && requested_text.rfind(base_text + "/", 0) != 0) { error = "repository path escapes the runtime root"; return false; }
    out = requested; return true;
}

bool text_file(const std::filesystem::path & path) {
    std::ifstream input(path, std::ios::binary);
    char buffer[1024]; input.read(buffer, sizeof(buffer));
    return input.good() || input.eof() ? std::find(buffer, buffer + input.gcount(), '\0') == buffer + input.gcount() : false;
}

bool git_read(const std::string & root, const std::string & arguments, std::string & output, std::string & error) {
    if (root.find_first_of("\"&|;<>`") != std::string::npos) { error = "repository root cannot be represented safely for Git"; return false; }
    const std::string command = "git -C \"" + root + "\" " + arguments + " 2>&1";
#ifdef _WIN32
    FILE * process = _popen(command.c_str(), "r");
#else
    FILE * process = popen(command.c_str(), "r");
#endif
    if (!process) { error = "unable to launch Git"; return false; }
    char buffer[512]; output.clear(); while (fgets(buffer, sizeof(buffer), process) && output.size() < 16384) output += buffer;
#ifdef _WIN32
    const int status = _pclose(process);
#else
    const int status = pclose(process);
#endif
    if (status != 0) { error = output.empty() ? "Git command failed" : output; return false; }
    return true;
}

std::string lower_copy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return (char) std::tolower(c); });
    return value;
}

std::string trim_copy(const std::string & value) {
    size_t begin = 0, end = value.size();
    while (begin < end && std::isspace((unsigned char) value[begin])) ++begin;
    while (end > begin && std::isspace((unsigned char) value[end - 1])) --end;
    return value.substr(begin, end - begin);
}

std::string collapse_ws(const std::string & value) {
    std::string out;
    bool spaced = false;
    for (unsigned char c : value) {
        if (std::isspace(c)) {
            if (!spaced && !out.empty()) out.push_back(' ');
            spaced = true;
        } else {
            out.push_back((char) c);
            spaced = false;
        }
    }
    return trim_copy(out);
}

void replace_all(std::string & value, const std::string & from, const std::string & to) {
    if (from.empty()) return;
    size_t pos = 0;
    while ((pos = value.find(from, pos)) != std::string::npos) {
        value.replace(pos, from.size(), to);
        pos += to.size();
    }
}

std::string html_decode(std::string value) {
    replace_all(value, "&amp;", "&");
    replace_all(value, "&lt;", "<");
    replace_all(value, "&gt;", ">");
    replace_all(value, "&quot;", "\"");
    replace_all(value, "&#39;", "'");
    replace_all(value, "&nbsp;", " ");
    return value;
}

std::string html_to_text(std::string html) {
    html = std::regex_replace(html, std::regex(R"(<script[\s\S]*?</script>)", std::regex::icase), " ");
    html = std::regex_replace(html, std::regex(R"(<style[\s\S]*?</style>)", std::regex::icase), " ");
    html = std::regex_replace(html, std::regex(R"(<[^>]+>)"), " ");
    return collapse_ws(html_decode(html));
}

std::string html_title(const std::string & html) {
    std::smatch match;
    if (!std::regex_search(html, match, std::regex(R"(<title[^>]*>([\s\S]*?)</title>)", std::regex::icase))) return {};
    return collapse_ws(html_to_text(match[1].str()));
}

std::string url_encode(const std::string & value) {
    std::ostringstream out;
    out << std::hex << std::uppercase;
    for (unsigned char c : value) {
        if (std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') out << c;
        else if (c == ' ') out << '+';
        else out << '%' << std::setw(2) << std::setfill('0') << (int) c;
    }
    return out.str();
}

std::string url_decode(const std::string & value) {
    std::string out;
    out.reserve(value.size());
    for (size_t i = 0; i < value.size(); ++i) {
        if (value[i] == '+' ) out.push_back(' ');
        else if (value[i] == '%' && i + 2 < value.size()) {
            const auto hex = value.substr(i + 1, 2);
            char * end = nullptr;
            const long code = std::strtol(hex.c_str(), &end, 16);
            if (end == hex.c_str() + 2) {
                out.push_back((char) code);
                i += 2;
            } else out.push_back(value[i]);
        } else out.push_back(value[i]);
    }
    return out;
}

bool private_ipv4(const std::string & host) {
    int a = -1, b = -1, c = -1, d = -1;
    char tail = 0;
    if (std::sscanf(host.c_str(), "%d.%d.%d.%d%c", &a, &b, &c, &d, &tail) != 4) return false;
    if (a == 10 || a == 127) return true;
    if (a == 192 && b == 168) return true;
    if (a == 172 && b >= 16 && b <= 31) return true;
    if (a == 169 && b == 254) return true;
    if (a == 0) return true;
    return false;
}

bool public_https_url(const common_http_url & parts, std::string & error) {
    if (parts.scheme != "https") { error = "only HTTPS URLs are allowed"; return false; }
    if (!parts.user.empty() || !parts.password.empty()) { error = "credentialed URLs are not allowed"; return false; }
    const auto host = lower_copy(parts.host);
    if (host.empty()) { error = "URL host is empty"; return false; }
    if (host == "localhost" || host == "::1" || host.find(".local") != std::string::npos) { error = "private or loopback hosts are not allowed"; return false; }
    if (private_ipv4(host)) { error = "private or loopback hosts are not allowed"; return false; }
    if (host.find(':') != std::string::npos) {
        if (host.rfind("fc", 0) == 0 || host.rfind("fd", 0) == 0 || host.rfind("fe80", 0) == 0 || host == "::1") {
            error = "private or loopback hosts are not allowed"; return false;
        }
    }
    return true;
}

#ifdef LLAMA_AGENT_WEB_USE_WINHTTP

std::wstring utf8_to_wide(const std::string & value) {
    if (value.empty()) return {};
    const int count = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value.data(), (int) value.size(), nullptr, 0);
    if (count <= 0) return {};
    std::wstring wide((size_t) count, L'\0');
    if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value.data(), (int) value.size(), wide.data(), count) != count) return {};
    return wide;
}

std::string winhttp_error(DWORD code) {
    LPSTR text = nullptr;
    const DWORD length = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        nullptr, code, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPSTR) &text, 0, nullptr);
    std::string result = length && text ? trim_copy(text) : "WinHTTP error " + std::to_string(code);
    if (text) LocalFree(text);
    return result;
}

bool winhttp_query_string(HINTERNET request, DWORD query, std::string & value, std::string & error) {
    DWORD bytes = 0;
    WinHttpQueryHeaders(request, query, WINHTTP_HEADER_NAME_BY_INDEX, nullptr, &bytes, WINHTTP_NO_HEADER_INDEX);
    if (GetLastError() != ERROR_INSUFFICIENT_BUFFER || bytes < sizeof(wchar_t)) {
        error = "WinHTTP header query failed: " + winhttp_error(GetLastError());
        return false;
    }
    std::wstring wide(bytes / sizeof(wchar_t), L'\0');
    if (!WinHttpQueryHeaders(request, query, WINHTTP_HEADER_NAME_BY_INDEX, wide.data(), &bytes, WINHTTP_NO_HEADER_INDEX)) {
        error = "WinHTTP header query failed: " + winhttp_error(GetLastError());
        return false;
    }
    const int output_size = WideCharToMultiByte(CP_UTF8, 0, wide.c_str(), -1, nullptr, 0, nullptr, nullptr);
    if (output_size <= 0) { error = "WinHTTP header conversion failed"; return false; }
    value.resize((size_t) output_size);
    WideCharToMultiByte(CP_UTF8, 0, wide.c_str(), -1, value.data(), output_size, nullptr, nullptr);
    value.pop_back();
    return true;
}

bool winhttp_fetch_text(const std::string & url, const common_http_url & parts, size_t max_bytes,
        std::string & body, int & status, std::string & content_type, std::string & error, bool & truncated) {
    const auto host = utf8_to_wide(parts.host);
    const auto path = utf8_to_wide(parts.path);
    if (host.empty() || path.empty()) { error = "URL is not valid UTF-8"; return false; }

    HINTERNET session = WinHttpOpen(L"llama-agent-poc/1", WINHTTP_ACCESS_TYPE_DEFAULT_PROXY,
        WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0);
    if (!session) { error = "WinHTTP session failed: " + winhttp_error(GetLastError()); return false; }
    const auto close_session = [&] { WinHttpCloseHandle(session); };
    WinHttpSetTimeouts(session, 10000, 10000, 10000, 10000);

    HINTERNET connection = WinHttpConnect(session, host.c_str(), (INTERNET_PORT) parts.port, 0);
    if (!connection) { error = "WinHTTP connection failed: " + winhttp_error(GetLastError()); close_session(); return false; }
    const auto close_connection = [&] { WinHttpCloseHandle(connection); close_session(); };

    HINTERNET request = WinHttpOpenRequest(connection, L"GET", path.c_str(), nullptr, WINHTTP_NO_REFERER,
        WINHTTP_DEFAULT_ACCEPT_TYPES, WINHTTP_FLAG_SECURE);
    if (!request) { error = "WinHTTP request creation failed: " + winhttp_error(GetLastError()); close_connection(); return false; }
    const auto close_request = [&] { WinHttpCloseHandle(request); close_connection(); };
    const wchar_t headers[] = L"Accept: text/html, text/plain;q=0.9, application/xhtml+xml;q=0.8\r\n";
    if (!WinHttpSendRequest(request, headers, (DWORD) -1L, WINHTTP_NO_REQUEST_DATA, 0, 0, 0) || !WinHttpReceiveResponse(request, nullptr)) {
        error = "WinHTTP request failed: " + winhttp_error(GetLastError()); close_request(); return false;
    }

    DWORD status_code = 0;
    DWORD status_size = sizeof(status_code);
    if (!WinHttpQueryHeaders(request, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER, WINHTTP_HEADER_NAME_BY_INDEX,
            &status_code, &status_size, WINHTTP_NO_HEADER_INDEX)) {
        error = "WinHTTP status query failed: " + winhttp_error(GetLastError()); close_request(); return false;
    }
    status = (int) status_code;
    if (status < 200 || status >= 300) { error = "HTTP status " + std::to_string(status); close_request(); return false; }
    std::string ignored;
    if (!winhttp_query_string(request, WINHTTP_QUERY_CONTENT_TYPE, content_type, ignored)) content_type.clear();

    body.clear();
    body.reserve(std::min<size_t>(max_bytes, 65536));
    for (;;) {
        DWORD available = 0;
        if (!WinHttpQueryDataAvailable(request, &available)) { error = "WinHTTP read failed: " + winhttp_error(GetLastError()); close_request(); return false; }
        if (available == 0) break;
        const size_t remaining = max_bytes > body.size() ? max_bytes - body.size() : 0;
        if (remaining == 0) { truncated = true; break; }
        const DWORD to_read = (DWORD) std::min<size_t>(available, std::min<size_t>(remaining, 8192));
        std::string chunk(to_read, '\0');
        DWORD received = 0;
        if (!WinHttpReadData(request, chunk.data(), to_read, &received)) { error = "WinHTTP read failed: " + winhttp_error(GetLastError()); close_request(); return false; }
        body.append(chunk.data(), received);
        if (received < available || body.size() >= max_bytes) { truncated = true; break; }
    }
    close_request();
    return true;
}

#endif

bool http_fetch_text(const std::string & url, size_t max_bytes, json & response, std::string & error, std::string * raw_body = nullptr) {
    try {
#ifdef LLAMA_AGENT_WEB_USE_WINHTTP
        const auto parts = common_http_parse_url(url);
        if (!public_https_url(parts, error)) return false;
        std::string body;
        bool truncated = false;
        int status = 0;
        std::string content_type;
        if (!winhttp_fetch_text(url, parts, max_bytes, body, status, content_type, error, truncated)) return false;
#else
        auto [cli, parts] = common_http_client(url);
        if (!public_https_url(parts, error)) return false;
        cli.set_connection_timeout(10);
        cli.set_read_timeout(10);
        cli.set_write_timeout(10);
        cli.set_follow_location(true);
        cli.set_default_headers({{"User-Agent", "llama-agent-poc/1"}, {"Accept", "text/html, text/plain;q=0.9, application/xhtml+xml;q=0.8"}});

        std::string body;
        body.reserve(std::min<size_t>(max_bytes, 65536));
        bool truncated = false;
        int status = 0;
        std::string content_type;
        auto res = cli.Get(parts.path,
            [&](const httplib::Response & r) {
                status = r.status;
                if (r.has_header("Content-Type")) content_type = r.get_header_value("Content-Type");
                return r.status >= 200 && r.status < 300;
            },
            [&](const char * data, size_t len) {
                const size_t remaining = max_bytes > body.size() ? max_bytes - body.size() : 0;
                const size_t take = std::min(len, remaining);
                body.append(data, take);
                if (take < len || body.size() >= max_bytes) {
                    truncated = true;
                    return false;
                }
                return true;
            },
            nullptr);
        if (!res) { error = "HTTP request failed: " + std::string(httplib::to_string(res.error())); return false; }
#endif
        if (raw_body) *raw_body = body;
        response = {
            {"url", url},
            {"final_url", url},
            {"status", status},
            {"content_type", content_type},
            {"title", html_title(body)},
            {"text", html_to_text(body)},
            {"truncated", truncated},
        };
        return true;
    } catch (const std::exception & e) {
        error = e.what();
        return false;
    }
}

bool parse_search_results(const std::string & html, int limit, json & results) {
    results = json::array();
    std::regex anchor(R"ddg(<a[^>]*href="([^"]+)"[^>]*>([\s\S]*?)</a>)ddg", std::regex::icase);
    std::set<std::string> seen;
    for (std::sregex_iterator it(html.begin(), html.end(), anchor), end; it != end && results.size() < (size_t) limit; ++it) {
        std::string href = html_decode((*it)[1].str());
        std::string title = collapse_ws(html_to_text((*it)[2].str()));
        if (title.empty()) continue;
        const auto lower_href = lower_copy(href);
        if (lower_href.find("duckduckgo.com/l/?") != std::string::npos) {
            const auto pos = href.find("uddg=");
            if (pos != std::string::npos) {
                auto encoded = href.substr(pos + 5);
                const auto amp = encoded.find('&');
                if (amp != std::string::npos) encoded = encoded.substr(0, amp);
                href = url_decode(encoded);
            }
        }
        if (lower_copy(href).rfind("https://", 0) != 0) continue;
        if (!seen.insert(href).second) continue;
        results.push_back({{"title", title}, {"url", href}, {"snippet", ""}, {"source", "duckduckgo-lite"}});
    }
    return !results.empty();
}

} // namespace

bool common_register_native_tool_adapters(const common_tool_catalog & catalog, const std::string & profile_id,
        const common_native_tool_bindings & bindings, common_tool_registry & registry,
        common_tool_adapter_result & result, std::string & error) {
    result = {};
    const auto definitions = catalog.load_profile(profile_id, error);
    if (!error.empty()) return false;
    for (const auto & definition : definitions) {
        bool installed = false;
        if (definition.executor_id == "builtin.calculator") {
            installed = register_definition(definition, registry, [](const std::string & input, std::string & output, std::string & err) {
                json arguments; if (!parse_object(input, arguments, err) || !arguments.contains("expression") || !arguments["expression"].is_string()) { if (err.empty()) err = "calculator requires an expression"; return false; }
                const auto expression = arguments["expression"].get<std::string>();
                if (expression.size() > 256) { err = "calculator expression exceeds limit"; return false; }
                double value = 0.0; calculator_parser parser(expression); if (!parser.parse(value, err)) return false;
                output = json({{"value", value}}).dump(); return true;
            }, error);
        } else if (definition.executor_id == "builtin.time_now") {
            installed = register_definition(definition, registry, [](const std::string & input, std::string & output, std::string & err) {
                json arguments; if (!parse_object(input, arguments, err)) return false;
                const auto timezone = arguments.value("timezone", std::string("UTC"));
                if (timezone != "UTC") { err = "time_now currently supports only UTC"; return false; }
                output = json({{"timezone", "UTC"}, {"time", utc_now()}}).dump(); return true;
            }, error);
        } else if (definition.executor_id == "builtin.repository_list" && !bindings.repository_root.empty()) {
            installed = register_definition(definition, registry, [bindings](const std::string & input, std::string & output, std::string & err) {
                json arguments; if (!parse_object(input, arguments, err)) return false;
                std::filesystem::path path; if (!repository_path(bindings.repository_root, arguments.value("path", std::string{}), path, err)) return false;
                const int depth = arguments.value("depth", 1); if (depth < 0 || depth > 3 || !std::filesystem::is_directory(path)) { err = "repository_list path or depth is invalid"; return false; }
                json entries = json::array();
                for (auto it = std::filesystem::recursive_directory_iterator(path, std::filesystem::directory_options::skip_permission_denied); it != std::filesystem::recursive_directory_iterator() && entries.size() < 128; ++it) {
                    if (it.depth() >= depth && it->is_directory()) it.disable_recursion_pending();
                    entries.push_back({{"path", std::filesystem::relative(it->path(), bindings.repository_root).generic_string()}, {"directory", it->is_directory()}});
                }
                output = json({{"entries", entries}}).dump(); return true;
            }, error);
        } else if (definition.executor_id == "builtin.repository_read" && !bindings.repository_root.empty()) {
            installed = register_definition(definition, registry, [bindings](const std::string & input, std::string & output, std::string & err) {
                json arguments; if (!parse_object(input, arguments, err) || !arguments.contains("path") || !arguments["path"].is_string()) { if (err.empty()) err = "repository_read requires a path"; return false; }
                std::filesystem::path path; if (!repository_path(bindings.repository_root, arguments["path"].get<std::string>(), path, err)) return false;
                const int start = arguments.value("start_line", 1), end = arguments.value("end_line", start + 199); if (start < 1 || end < start || end - start > 399 || !std::filesystem::is_regular_file(path) || !text_file(path)) { err = "repository_read range or file is invalid"; return false; }
                std::ifstream file(path); std::string line; json lines = json::array(); for (int number = 1; std::getline(file, line); ++number) if (number >= start && number <= end) lines.push_back({{"line", number}, {"text", line}});
                output = json({{"path", std::filesystem::relative(path, bindings.repository_root).generic_string()}, {"lines", lines}}).dump(); return true;
            }, error);
        } else if (definition.executor_id == "builtin.repository_search" && !bindings.repository_root.empty()) {
            installed = register_definition(definition, registry, [bindings](const std::string & input, std::string & output, std::string & err) {
                json arguments; if (!parse_object(input, arguments, err) || !arguments.contains("query") || !arguments["query"].is_string()) { if (err.empty()) err = "repository_search requires a query"; return false; }
                const auto query = arguments["query"].get<std::string>(); const int limit = arguments.value("max_results", 16); if (query.empty() || query.size() > 256 || limit < 1 || limit > 32) { err = "repository_search arguments are out of bounds"; return false; }
                std::filesystem::path root; if (!repository_path(bindings.repository_root, arguments.value("path", std::string{}), root, err)) return false;
                json matches = json::array(); for (auto it = std::filesystem::recursive_directory_iterator(root, std::filesystem::directory_options::skip_permission_denied); it != std::filesystem::recursive_directory_iterator() && matches.size() < (size_t) limit; ++it) { if (!it->is_regular_file() || it->file_size() > 512 * 1024 || !text_file(it->path())) continue; std::ifstream file(it->path()); std::string line; for (int number = 1; std::getline(file, line) && matches.size() < (size_t) limit; ++number) if (line.find(query) != std::string::npos) matches.push_back({{"path", std::filesystem::relative(it->path(), bindings.repository_root).generic_string()}, {"line", number}, {"preview", line.substr(0, 512)}}); }
                output = json({{"matches", matches}}).dump(); return true;
            }, error);
        } else if (definition.executor_id == "builtin.repository_diff" && !bindings.repository_root.empty()) {
            installed = register_definition(definition, registry, [bindings](const std::string & input, std::string & output, std::string & err) {
                json arguments; if (!parse_object(input, arguments, err) || !arguments.empty()) { if (err.empty()) err = "repository_diff takes no arguments"; return false; }
                std::string diff; if (!git_read(bindings.repository_root, "diff --no-ext-diff --stat", diff, err)) return false;
                output = json({{"summary", diff}}).dump(); return true;
            }, error);
        } else if (definition.executor_id == "builtin.repository_log" && !bindings.repository_root.empty()) {
            installed = register_definition(definition, registry, [bindings](const std::string & input, std::string & output, std::string & err) {
                json arguments; if (!parse_object(input, arguments, err)) return false;
                const int limit = arguments.value("limit", 8); if (limit < 1 || limit > 20) { err = "repository_log limit is out of bounds"; return false; }
                std::string log; if (!git_read(bindings.repository_root, "log --no-ext-diff --max-count=" + std::to_string(limit) + " --pretty=format:%h%x09%s", log, err)) return false;
                output = json({{"commits", log}}).dump(); return true;
            }, error);
        } else if (definition.executor_id == "builtin.web_search") {
            if (bindings.web_search) {
                installed = register_definition(definition, registry, bindings.web_search, error);
            } else {
                installed = register_definition(definition, registry, [](const std::string & input, std::string & output, std::string & err) {
                    json arguments; if (!parse_object(input, arguments, err) || !arguments.contains("query") || !arguments["query"].is_string()) { if (err.empty()) err = "web_search requires a query"; return false; }
                    const auto query = trim_copy(arguments["query"].get<std::string>());
                    const int limit = arguments.value("limit", 5);
                    const auto site = trim_copy(arguments.value("site", std::string{}));
                    if (query.empty() || query.size() > 512 || limit < 1 || limit > 8 || site.size() > 256) { err = "web_search arguments are out of bounds"; return false; }
                    const auto search_query = site.empty() ? query : query + " site:" + site;
                    json fetched;
                    std::string raw_html;
                    if (!http_fetch_text("https://lite.duckduckgo.com/lite/?q=" + url_encode(search_query), 128 * 1024, fetched, err, &raw_html)) return false;
                    json results;
                    if (!parse_search_results(raw_html, limit, results)) {
                        output = json({{"results", json::array()}, {"provider", "duckduckgo-lite"}}).dump();
                        return true;
                    }
                    output = json({{"results", results}, {"provider", "duckduckgo-lite"}}).dump();
                    return true;
                }, error);
            }
        } else if (definition.executor_id == "builtin.web_fetch") {
            if (bindings.web_fetch) {
                installed = register_definition(definition, registry, bindings.web_fetch, error);
            } else {
                installed = register_definition(definition, registry, [](const std::string & input, std::string & output, std::string & err) {
                    json arguments; if (!parse_object(input, arguments, err) || !arguments.contains("url") || !arguments["url"].is_string()) { if (err.empty()) err = "web_fetch requires a url"; return false; }
                    const auto url = trim_copy(arguments["url"].get<std::string>());
                    const auto extract = arguments.value("extract", std::string("text"));
                    const int max_bytes = arguments.value("max_bytes", 64000);
                    if (url.size() < 9 || url.size() > 2048 || extract != "text" || max_bytes < 1 || max_bytes > 500000) { err = "web_fetch arguments are out of bounds"; return false; }
                    json fetched;
                    if (!http_fetch_text(url, (size_t) max_bytes, fetched, err)) return false;
                    output = fetched.dump();
                    return true;
                }, error);
            }
        } else if (definition.executor_id == "builtin.memory_search" && bindings.memory_store) {
            installed = register_definition(definition, registry, [bindings](const std::string & input, std::string & output, std::string & err) {
                json arguments; if (!parse_object(input, arguments, err) || !arguments.contains("query") || !arguments["query"].is_string()) { if (err.empty()) err = "memory_search requires a query"; return false; }
                common_memory_query query = bindings.memory_query;
                query.text = arguments["query"].get<std::string>();
                query.limit = (size_t) arguments.value("limit", 5);
                if (query.text.empty() || query.text.size() > 1024 || query.limit < 1 || query.limit > 8) { err = "memory_search arguments are out of bounds"; return false; }
                if (bindings.embed_memory_query) { query.embedding.clear(); if (!bindings.embed_memory_query(query.text, query.embedding, err)) return false; }
                common_memory_retrieval retrieval(*bindings.memory_store);
                const auto hits = retrieval.retrieve(query, err); if (!err.empty()) return false;
                json values = json::array();
                for (const auto & hit : hits) values.push_back({{"memory", memory_value(hit.memory)}, {"score", hit.final_score}, {"provenance", hit.provenance}});
                output = json({{"results", values}}).dump(); return true;
            }, error);
        } else if (definition.executor_id == "builtin.memory_get" && bindings.memory_store) {
            installed = register_definition(definition, registry, [bindings](const std::string & input, std::string & output, std::string & err) {
                json arguments; if (!parse_object(input, arguments, err) || !arguments.contains("id") || !arguments["id"].is_string()) { if (err.empty()) err = "memory_get requires an id"; return false; }
                const auto id = arguments["id"].get<std::string>();
                if (id.empty() || id.size() > 256) { err = "memory id is out of bounds"; return false; }
                const auto memory = bindings.memory_store->get(id, err); if (!err.empty()) return false;
                if (!memory || !common_memory_scope_matches(*memory, bindings.memory_query)) { err = "memory is unavailable in the current scope"; return false; }
                output = json({{"memory", memory_value(*memory)}}).dump(); return true;
            }, error);
        } else if (definition.executor_id == "builtin.memory_remember" && bindings.memory_remember_proposal) {
            installed = register_definition(definition, registry, bindings.memory_remember_proposal, error, false, true);
        } else if (definition.executor_id == "builtin.plan_get" && bindings.plan_store && !bindings.plan_id.empty()) {
            installed = register_definition(definition, registry, [bindings](const std::string & input, std::string & output, std::string & err) {
                json arguments; if (!parse_object(input, arguments, err)) return false;
                const auto plan = bindings.plan_store->get(bindings.plan_id, err); if (!err.empty()) return false;
                if (!plan) { err = "bound plan is unavailable"; return false; }
                json steps = json::array();
                for (const auto & step : plan->steps) steps.push_back({{"id", step.id}, {"title", step.title}, {"objective", step.objective}, {"status", (int) step.status}, {"selected_tool", step.selected_tool}});
                json response = {{"plan_id", plan->id}, {"version", plan->version}, {"goal", plan->goal}, {"active_step", plan->active_step_id}, {"next_action", plan->next_action}, {"steps", steps}};
                if (arguments.value("include_history", false)) {
                    const auto history = bindings.plan_store->history(bindings.plan_id, err); if (!err.empty()) return false;
                    response["history_count"] = history.size();
                }
                output = response.dump(); return true;
            }, error);
        }
        if (!error.empty()) return false;
        if (installed) result.registered.push_back(definition.name);
        else result.unavailable.push_back(definition.name);
    }
    error.clear();
    return true;
}
