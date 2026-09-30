#include "agent/tooling/schema/tool-schema-compact.h"
#include "tools/agent/openapi/agent-openapi-catalog.h"
#include "tools/agent/runtime/agent-server-context-host.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

using json = nlohmann::ordered_json;

struct options {
    std::string model;
    std::string spec;
    int threads = 4;
    int gpu_layers = 99;
    int n_predict = 48;
    bool strict = false;
};

bool parse_options(int argc, char ** argv, options & value) {
    if (const char * model = std::getenv("LLAMA_AGENT_MODEL")) value.model = model;
    if (const char * spec = std::getenv("LLAMA_AGENT_OPENALEX_SPEC")) value.spec = spec;
    if (const char * threads = std::getenv("LLAMA_AGENT_THREADS")) value.threads = std::stoi(threads);
    if (const char * gpu_layers = std::getenv("LLAMA_AGENT_GPU_LAYERS")) value.gpu_layers = std::stoi(gpu_layers);
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        const auto next = [&]() -> const char * {
            return index + 1 < argc ? argv[++index] : nullptr;
        };
        if (argument == "--model") {
            const char * value_text = next(); if (!value_text) return false; value.model = value_text;
        } else if (argument == "--openalex-spec") {
            const char * value_text = next(); if (!value_text) return false; value.spec = value_text;
        } else if (argument == "--threads") {
            const char * value_text = next(); if (!value_text) return false; value.threads = std::stoi(value_text);
        } else if (argument == "--n-gpu-layers") {
            const char * value_text = next(); if (!value_text) return false; value.gpu_layers = std::stoi(value_text);
        } else if (argument == "--n-predict") {
            const char * value_text = next(); if (!value_text) return false; value.n_predict = std::stoi(value_text);
        } else if (argument == "--strict") {
            value.strict = true;
        } else if (argument == "--help" || argument == "-h") {
            return false;
        } else {
            return false;
        }
    }
    return value.threads > 0 && value.threads <= 4 && value.n_predict > 0 &&
        !value.model.empty() && !value.spec.empty();
}

std::string trim(const std::string & input) {
    size_t first = 0;
    while (first < input.size() && std::isspace(static_cast<unsigned char>(input[first]))) ++first;
    size_t last = input.size();
    while (last > first && std::isspace(static_cast<unsigned char>(input[last - 1]))) --last;
    return input.substr(first, last - first);
}

bool identifier_char(char value) {
    return std::isalnum(static_cast<unsigned char>(value)) || value == '_' || value == '.' || value == '-';
}

bool parse_compact_dsl(
        const std::string & text,
        std::string & tool,
        json & arguments,
        std::string & error) {
    const std::string line = trim(text);
    if (line.empty() || line.find('\n') != std::string::npos || line.find('\r') != std::string::npos) {
        error = "compact DSL must be exactly one non-empty line";
        return false;
    }
    if (line.rfind("open!", 0) != 0) {
        error = "compact DSL must start with open!";
        return false;
    }
    size_t position = 5;
    const size_t tool_start = position;
    while (position < line.size() && !std::isspace(static_cast<unsigned char>(line[position]))) {
        if (!identifier_char(line[position])) {
            error = "compact DSL contains an invalid tool name";
            return false;
        }
        ++position;
    }
    if (position == tool_start) {
        error = "compact DSL has no tool name";
        return false;
    }
    tool = line.substr(tool_start, position - tool_start);
    arguments = json::object();
    while (position < line.size()) {
        while (position < line.size() && std::isspace(static_cast<unsigned char>(line[position]))) ++position;
        if (position == line.size()) break;
        const size_t key_start = position;
        while (position < line.size() && identifier_char(line[position])) ++position;
        if (position == key_start || position >= line.size() || line[position] != '=') {
            error = "compact DSL argument must be key=value";
            return false;
        }
        const std::string key = line.substr(key_start, position - key_start);
        if (arguments.contains(key)) {
            error = "compact DSL repeats argument: " + key;
            return false;
        }
        ++position;
        if (position >= line.size()) {
            error = "compact DSL argument has no value: " + key;
            return false;
        }
        if (line[position] == '"') {
            ++position;
            std::string value;
            bool closed = false;
            while (position < line.size()) {
                const char character = line[position++];
                if (character == '"') { closed = true; break; }
                if (character == '\\') {
                    if (position >= line.size()) break;
                    const char escaped = line[position++];
                    if (escaped == '"' || escaped == '\\') value.push_back(escaped);
                    else { error = "compact DSL has unsupported string escape"; return false; }
                } else {
                    value.push_back(character);
                }
            }
            if (!closed) { error = "compact DSL has an unterminated string"; return false; }
            arguments[key] = value;
            continue;
        }
        const size_t value_start = position;
        while (position < line.size() && !std::isspace(static_cast<unsigned char>(line[position]))) ++position;
        const std::string value = line.substr(value_start, position - value_start);
        if (value == "true" || value == "false") {
            arguments[key] = value == "true";
        } else {
            const auto parsed = json::parse(value, nullptr, false);
            if (parsed.is_discarded() || !parsed.is_number()) {
                error = "compact DSL scalar is not a JSON number or boolean: " + key;
                return false;
            }
            arguments[key] = parsed;
        }
    }
    error.clear();
    return true;
}

std::string one_line(const std::string & text) {
    std::string result = text;
    for (char & character : result) {
        if (character == '\n') character = ' ';
        if (character == '\r') character = ' ';
        if (character == '\t') character = ' ';
    }
    return result;
}

bool load_tool(const options & value, common_chat_tool & tool, std::string & error) {
    std::ifstream input(value.spec);
    if (!input) { error = "could not open OpenAlex spec: " + value.spec; return false; }
    json document;
    input >> document;
    agent_host_openapi_provider_config config;
    config.id = "openalex";
    config.prefix = "openalex";
    config.access = "read_only";
    config.exposure = "auto";
    agent_openapi_catalog catalog;
    if (!build_agent_openapi_catalog(document, config, catalog, error)) return false;
    const auto operation = std::find_if(catalog.operations.begin(), catalog.operations.end(),
        [](const agent_openapi_operation & candidate) { return candidate.operation_id == "listWorks"; });
    if (operation == catalog.operations.end()) {
        error = "OpenAlex spec does not contain listWorks";
        return false;
    }
    tool = {
        "openalex.listWorks",
        operation->description.empty() ? operation->summary : operation->description,
        operation->input_schema_json,
        operation->result_schema_json,
    };
    return true;
}

} // namespace

int main(int argc, char ** argv) {
    options value;
    if (!parse_options(argc, argv, value)) {
        std::cerr << "usage: " << argv[0]
                  << " --model MODEL --openalex-spec PATH [--threads N]"
                  << " [--n-gpu-layers N] [--n-predict N] [--strict]\n";
        return 2;
    }
    if (!std::filesystem::is_regular_file(value.model) ||
            !std::filesystem::is_regular_file(value.spec)) return 77;

    common_chat_tool tool;
    std::string error;
    if (!load_tool(value, tool, error)) {
        std::cerr << error << '\n';
        return 1;
    }
    const std::string compact_contract = common_render_compact_tool_description(
        tool.name, tool.description, tool.parameters, tool.result_schema, error);
    if (!error.empty()) {
        std::cerr << "could not render model-facing contract: " << error << '\n';
        return 1;
    }

    auto host = std::make_shared<common_agent_server_context_host>();
    common_agent_server_context_host_config host_config;
    host_config.context_key.load_key.model = value.model;
    host_config.context_key.load_key.n_gpu_layers = value.gpu_layers;
    host_config.context_key.load_key.fit_params = true;
    host_config.context_key.n_parallel = 1;
    host_config.context_key.n_sequences = 1;
    host_config.context_key.n_ctx = 2048;
    host_config.context_key.n_threads = value.threads;
    host_config.verbosity = LOG_LEVEL_WARN;
    if (!host->start(host_config, error)) {
        std::cerr << "server host failed: " << error << '\n';
        return 1;
    }
    common_agent_inference_session session;
    if (!host->build_inference_session(session, error) || !session.inference) {
        std::cerr << "server inference session failed: " << error << '\n';
        return 1;
    }

    common_agent_generation_request request;
    request.purpose = common_agent_generation_purpose::operation_selection;
    request.trace_id = "openalex-compact-dsl-smoke";
    request.options.n_predict = value.n_predict;
    request.options.n_threads = value.threads;
    request.options.generation_trace = true;
    // This is deliberately an empty native-tools vector. The smoke is testing
    // the model's response to an explicitly requested compact DSL, not asking
    // the production native tool grammar to intercept the output.
    request.messages = {
        {"system",
            "Tool execution is required. Use the registered model-facing tool contract below, "
            "but return exactly one compact DSL call on one line. The format is "
            "open!TOOL_NAME key=value ... . Quote string values with double quotes. "
            "Do not return JSON, native tool-call wrappers, explanations, or a final answer.\n\n"
            "Registered tool contract:\n" + compact_contract},
        {"user",
            "Search OpenAlex for machine learning and request one result with fields id and display_name. "
            "Emit the compact DSL call now."},
    };

    const auto result = session.inference->generate_result(request);
    std::cout << "compact_dsl_raw=" << one_line(result.content) << '\n';
    if (!common_agent_generation_succeeded(result)) {
        std::cout << "compact_dsl_result=not_generated error=" << result.error_message << '\n';
        return 1;
    }
    std::string selected_tool;
    json arguments;
    std::string parse_error;
    const bool parsed = parse_compact_dsl(result.content, selected_tool, arguments, parse_error);
    const json expected_arguments = {
        {"search", "machine learning"},
        {"per_page", 1},
        {"select", "id,display_name"},
    };
    const bool contract_match = parsed && selected_tool == tool.name && arguments == expected_arguments;
    std::cout << "compact_dsl_parsed=" << (parsed ? "yes" : "no")
              << " tool=" << (selected_tool.empty() ? "<none>" : selected_tool)
              << " arguments=" << (parsed ? arguments.dump() : "<none>")
              << " parse_error=" << (parse_error.empty() ? "<none>" : parse_error) << '\n';
    std::cout << "compact_dsl_result=" << (contract_match ? "passed" : "invalid")
              << " production_changed=no server_context=yes native_parser_used=no\n";
    return contract_match || !value.strict ? 0 : 1;
}
