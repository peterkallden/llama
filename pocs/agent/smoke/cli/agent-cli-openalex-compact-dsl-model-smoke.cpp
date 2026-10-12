#include "agent/tooling/schema/tool-output-codec.h"
#include "tools/agent/openapi/agent-openapi-catalog.h"
#include "tools/agent/runtime/agent-server-context-host.h"

#include <nlohmann/json.hpp>

#include <algorithm>
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
    common_agent_tool_output_format tool_output_format = common_agent_tool_output_format::compact_dsl;
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
        } else if (argument == "--tool-output-format") {
            const char * value_text = next(); if (!value_text) return false;
            std::string format_error;
            if (!common_parse_agent_tool_output_format(value_text, value.tool_output_format, format_error) ||
                    value.tool_output_format == common_agent_tool_output_format::native) return false;
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

std::string one_line(const std::string & text) {
    std::string result = text;
    for (char & character : result) {
        if (character == '\n') character = ' ';
        if (character == '\r') character = ' ';
        if (character == '\t') character = ' ';
    }
    return result;
}

bool model_arguments_match_fixture(
        const json & arguments,
        std::string & mismatch) {
    mismatch.clear();
    if (arguments.value("search", std::string{}) != "machine learning") {
        mismatch = "search does not match";
        return false;
    }
    if (!arguments.contains("per_page") || !arguments["per_page"].is_number_integer() ||
            arguments["per_page"].get<int>() != 1) {
        mismatch = "fixture requests per_page=1";
        return false;
    }
    if (!arguments.contains("select") || !arguments["select"].is_string()) {
        mismatch = "select is missing or not a string";
        return false;
    }
    std::string normalized_select;
    std::string select_error;
    if (!common_normalize_comma_separated_string(
            arguments["select"].get<std::string>(), normalized_select, select_error)) {
        mismatch = select_error;
        return false;
    }
    if (normalized_select != "id,display_name") {
        mismatch = "select does not match id,display_name";
        return false;
    }
    return true;
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
                  << " [--tool-output-format {json,dsl}]"
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
    std::vector<std::string> unsupported;
    const std::string model_contract = common_render_model_tool_output_instructions(
        value.tool_output_format, {tool}, &unsupported, error);
    if (!error.empty()) {
        std::cerr << "could not render model-facing contract: " << error << '\n';
        return 1;
    }
    if (!unsupported.empty()) {
        std::cerr << "OpenAlex operation is not compact-DSL representable\n";
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
            model_contract + "\nDo not return JSON, native tool-call wrappers, explanations, or a final answer."},
        {"user",
            "Search OpenAlex for machine learning and request one result with fields id and display_name. "
            "Emit the compact DSL call now."},
    };

    const auto result = session.inference->generate_result(request);
    std::cout << "model_output_format="
              << common_agent_tool_output_format_name(value.tool_output_format) << '\n';
    std::cout << "model_output_raw=" << one_line(result.content) << '\n';
    if (!common_agent_generation_succeeded(result)) {
        std::cout << "model_output_result=not_generated error=" << result.error_message << '\n';
        return 1;
    }
    common_agent_tool_call parsed_call;
    std::string parse_error;
    const bool parsed = common_parse_model_tool_call(
        value.tool_output_format, result.content, parsed_call, parse_error);
    const auto arguments = parsed ? json::parse(parsed_call.arguments_json) : json::object();
    std::string fixture_mismatch;
    const bool contract_match = parsed && parsed_call.name == tool.name &&
        model_arguments_match_fixture(arguments, fixture_mismatch);
    std::cout << "model_output_parsed=" << (parsed ? "yes" : "no")
              << " tool=" << (parsed_call.name.empty() ? "<none>" : parsed_call.name)
              << " arguments=" << (parsed ? arguments.dump() : "<none>")
              << " parse_error=" << (parse_error.empty() ? "<none>" : parse_error)
              << " fixture_mismatch=" << (fixture_mismatch.empty() ? "<none>" : fixture_mismatch)
              << '\n';
    std::cout << "model_output_result=" << (contract_match ? "passed" : "invalid")
              << " production_changed=no server_context=yes native_parser_used=no\n";
    return contract_match || !value.strict ? 0 : 1;
}
