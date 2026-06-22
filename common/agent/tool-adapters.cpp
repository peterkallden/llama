#include "agent/tool-adapters.h"

#include "memory/memory-retrieval.h"

#include <chrono>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <nlohmann/json.hpp>
#include <sstream>

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
