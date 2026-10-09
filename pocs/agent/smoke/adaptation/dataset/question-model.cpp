#include "agent/adaptation/flydelta/flydelta-capture.h"
#include "agent/tooling/catalog/tool-catalog.h"
#include "agent/tooling/schema/tool-schema-compact.h"
#include "agent/tool-family-index.h"
#include "tools/agent/runtime/agent-server-context-host.h"
#include "tools/server/server-context.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

namespace {

using suite_json = nlohmann::json;

struct options {
    std::string model;
    std::string suite;
    int n_predict = 96;
    int n_threads = 3;
    int n_gpu_layers = 0;
    bool strict = false;
};

bool parse_args(int argc, char ** argv, options & value) {
    if (const char * model = std::getenv("LLAMA_AGENT_MODEL")) value.model = model;
    if (const char * suite = std::getenv("LLAMA_AGENT_DATASET_QUESTION_SUITE")) value.suite = suite;
    if (const char * threads = std::getenv("LLAMA_AGENT_THREADS")) value.n_threads = std::stoi(threads);
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&](const char * name) -> const char * {
            if (i + 1 >= argc) { std::cerr << "missing value for " << name << '\n'; return nullptr; }
            return argv[++i];
        };
        if (arg == "--model") { const char * v = next("--model"); if (!v) return false; value.model = v; }
        else if (arg == "--suite") { const char * v = next("--suite"); if (!v) return false; value.suite = v; }
        else if (arg == "--n-predict") { const char * v = next("--n-predict"); if (!v) return false; value.n_predict = std::stoi(v); }
        else if (arg == "--threads") { const char * v = next("--threads"); if (!v) return false; value.n_threads = std::stoi(v); }
        else if (arg == "--n-gpu-layers") { const char * v = next("--n-gpu-layers"); if (!v) return false; value.n_gpu_layers = std::stoi(v); }
        else if (arg == "--strict") value.strict = true;
        else if (arg == "--help" || arg == "-h") return false;
        else { std::cerr << "unknown argument: " << arg << '\n'; return false; }
    }
    return true;
}

bool read_json(const std::string & path, suite_json & value, std::string & error) {
    std::ifstream input(path, std::ios::binary);
    if (!input) { error = "could not open suite: " + path; return false; }
    std::ostringstream text; text << input.rdbuf();
    value = suite_json::parse(text.str(), nullptr, false);
    if (value.is_discarded() || !value.is_object()) { error = "suite is not valid JSON object"; return false; }
    return true;
}

std::string preview(const common_agent_generation_result & result) {
    if (!common_agent_generation_succeeded(result)) return "<failed: " + result.error_message + ">";
    std::string value = result.content;
    for (char & c : value) if (c == '\n' || c == '\r' || c == '\t') c = ' ';
    if (value.size() > 300) value.resize(300);
    return value;
}

std::string selected_tool(const common_agent_generation_result & result) {
    if (!common_agent_generation_succeeded(result)) return {};
    if (result.chat_params) {
        common_chat_parser_params parser_params(*result.chat_params);
        parser_params.parse_tool_calls = true;
        if (!result.chat_params->parser.empty()) {
            parser_params.parser.load(result.chat_params->parser);
        }
        const auto assistant = common_chat_parse(result.content, false, parser_params);
        if (!assistant.tool_calls.empty()) return assistant.tool_calls.front().name;
    }
    const auto parsed = suite_json::parse(result.content, nullptr, false);
    if (parsed.is_object() && parsed.contains("name") && parsed["name"].is_string()) return parsed["name"].get<std::string>();
    return {};
}

} // namespace

int main(int argc, char ** argv) {
    options value;
    if (!parse_args(argc, argv, value)) {
        std::cerr << "usage: " << argv[0] << " --model MODEL --suite SUITE_JSON [--strict]\n";
        return 2;
    }
    if (value.model.empty() || !std::filesystem::is_regular_file(value.model)) {
        std::cerr << "dataset question model smoke skipped: provide --model or LLAMA_AGENT_MODEL\n";
        return 77;
    }
    if (value.suite.empty() || !std::filesystem::is_regular_file(value.suite)) {
        std::cerr << "dataset question model smoke requires --suite or LLAMA_AGENT_DATASET_QUESTION_SUITE\n";
        return 2;
    }
    if (value.n_threads <= 0 || value.n_threads > 3 || value.n_predict <= 0) return 2;

    suite_json suite;
    std::string error;
    if (!read_json(value.suite, suite, error)) { std::cerr << error << '\n'; return 1; }
    common_tool_catalog catalog;
    common_tool_bootstrap_result bootstrap;
    if (!catalog.bootstrap("analysis", bootstrap, error)) { std::cerr << error << '\n'; return 1; }
    auto server_host = std::make_shared<common_agent_server_context_host>();
    common_agent_server_context_host_config server_config;
    server_config.context_key.load_key.model = value.model;
    server_config.context_key.load_key.n_gpu_layers = value.n_gpu_layers;
    server_config.context_key.load_key.fit_params = true;
    server_config.context_key.n_parallel = 1;
    server_config.context_key.n_sequences = 1;
    // Dataset-question schemas and the generated model-facing contract can
    // exceed the small runtime smoke budget; keep the resident server
    // context large enough for the real request rather than failing before
    // inference.
    server_config.context_key.n_ctx = 4096;
    server_config.context_key.n_threads = value.n_threads;
    server_config.verbosity = LOG_LEVEL_WARN;
    if (!server_host->start(server_config, error)) {
        std::cerr << "server-context start failed: " << error << '\n';
        return 1;
    }
    common_agent_inference_session server_session;
    if (!server_host->build_inference_session(server_session, error) ||
            !server_session.inference) {
        std::cerr << "server-context session failed: " << error << '\n';
        return 1;
    }
    auto inference = std::move(server_session.inference);
    auto * model_context = server_host->server().get_llama_context();
    auto * model = model_context == nullptr ? nullptr : llama_get_model(model_context);
    if (model == nullptr) return 1;

    const auto definitions = catalog.load_profile("analysis", error);
    if (!error.empty()) {
        std::cerr << "could not load analysis tool profile: " << error << '\n';
        return 1;
    }
    std::vector<common_chat_tool> chat_tools;
    chat_tools.reserve(definitions.size());
    for (const auto & definition : definitions) {
        std::string compact_error;
        const auto description = common_render_compact_tool_description(
            definition.name, definition.description,
            common_tool_model_input_schema(definition),
            common_tool_model_result_schema(definition), compact_error);
        if (!compact_error.empty()) {
            std::cerr << compact_error << '\n';
            return 1;
        }
        chat_tools.push_back({definition.name, description,
            common_tool_model_input_schema(definition),
            common_tool_model_result_schema(definition)});
    }
    const auto families = common_generate_tool_family_index(chat_tools);
    const auto family_schema = common_tool_family_required_selection_schema(families);

    auto capture = std::make_shared<common_flydelta_hidden_state_capture_request>();
    capture->enabled = true;
    const auto layers = static_cast<uint32_t>(llama_model_n_layer(model));
    for (uint32_t layer = 1; layer < layers && layer <= 4; ++layer) capture->layer_indices.push_back(layer);
    capture->token_index = -1;
    capture->max_bytes = 4U * 1024U * 1024U;
    capture->model_profile_fingerprint = "sha256:flydelta-dataset-question-suite";
    capture->capture_layout_revision = "layer-input:v1";

    size_t matched = 0;
    size_t mismatched = 0;
    for (const auto & scenario : suite["scenarios"]) {
        const auto expected = scenario.value("expected_tool", "");
        const auto family_prompt = common_tool_family_required_selection_prompt(families);
        common_agent_generation_options family_options;
        family_options.n_predict = 64;
        family_options.n_threads = value.n_threads;
        family_options.generation_trace = true;
        auto family_request = common_agent_make_generation_request(
            common_agent_generation_purpose::tool_family_selection,
            "flydelta-dataset-question-family-" + scenario.value("id", ""),
            std::nullopt,
            {{"system", family_prompt}, {"user", scenario.value("question", "")}},
            family_options,
            family_schema);
        auto family_result = inference->generate_result(family_request);
        const std::string initial_family_output = preview(family_result);
        common_tool_family_selection family_selection;
        std::string family_error;
        bool family_selected = common_agent_generation_succeeded(family_result) &&
            common_parse_tool_family_required_selection(
                family_result.content, families, family_selection, family_error);
        bool family_repaired = false;
        if (!family_selected) {
            common_agent_generation_options repair_options = family_options;
            auto repair_request = common_agent_make_generation_request(
                common_agent_generation_purpose::tool_family_selection,
                "flydelta-dataset-question-family-repair-" + scenario.value("id", ""),
                std::nullopt,
                {{"system", family_prompt},
                 {"system", common_tool_family_required_selection_repair_prompt()},
                 {"user", scenario.value("question", "")}},
                repair_options,
                family_schema);
            auto repaired_result = inference->generate_result(repair_request);
            std::string repair_error;
            family_repaired = common_agent_generation_succeeded(repaired_result) &&
                common_parse_tool_family_required_selection(
                    repaired_result.content, families, family_selection, repair_error);
            if (family_repaired) {
                family_result = std::move(repaired_result);
                family_selected = true;
                family_error.clear();
            } else {
                if (!repair_error.empty()) family_error = std::move(repair_error);
                family_result = std::move(repaired_result);
            }
        }
        if (!family_selected) {
            ++mismatched;
            std::cout << "scenario=" << scenario.value("id", "")
                      << " expected=" << expected
                      << " family_selection=failed"
                      << " family_repaired=no"
                      << " error=" << (family_error.empty() ? family_result.error_message : family_error)
                      << " initial_output=" << initial_family_output
                      << " output=" << preview(family_result) << '\n';
            continue;
        }
        const auto family_tools = common_filter_tools_by_families(
            chat_tools, family_selection.family_ids);
        const bool expected_tool_available = std::any_of(
            family_tools.begin(), family_tools.end(), [&](const auto & tool) {
                return tool.name == expected;
            });
        if (!expected_tool_available) {
            ++mismatched;
            std::cout << "scenario=" << scenario.value("id", "")
                      << " expected=" << expected
                      << " selected_families=";
            for (size_t i = 0; i < family_selection.family_ids.size(); ++i) {
                if (i != 0) std::cout << ',';
                std::cout << family_selection.family_ids[i];
            }
            std::cout << " family_mismatch=expected_tool_not_exposed"
                      << " initial_family_output=" << initial_family_output
                      << " family_output=" << preview(family_result) << '\n';
            continue;
        }

        common_agent_generation_request request;
        request.purpose = common_agent_generation_purpose::tool_followup;
        request.options.n_predict = value.n_predict;
        request.options.n_threads = value.n_threads;
        request.options.generation_trace = true;
        request.tools = family_tools;
        request.tool_choice = COMMON_CHAT_TOOL_CHOICE_REQUIRED;
        request.messages = {
            {"system", "You are a host-controlled dataset tool selector. Select exactly one most-specific read-only tool. "
                        "Use the canonical dataset reference " + suite.value("dataset", "dataset://local/sales") + ". "
                        "Return exactly one native tool call with its arguments, with no markdown or explanation."},
            {"user", scenario.value("question", "")},
        };
        request.flydelta_capture = capture;
        common_agent_generation_result result;
        const bool executed = inference->generate(request, result);
        const auto actual = selected_tool(result);
        const bool correct = executed && actual == expected;
        if (correct) ++matched; else ++mismatched;
        std::cout << "scenario=" << scenario.value("id", "")
                  << " expected=" << expected << " selected=" << (actual.empty() ? "<none>" : actual)
                  << " selected_families=";
        for (size_t i = 0; i < family_selection.family_ids.size(); ++i) {
            if (i != 0) std::cout << ',';
            std::cout << family_selection.family_ids[i];
        }
        std::cout << " exposed_tools=" << family_tools.size()
                  << " family_repaired=" << (family_repaired ? "yes" : "no")
                  << " initial_family_output=" << initial_family_output
                  << " family_output=" << preview(family_result)
                  << " outcome=" << (correct ? "matched" : "repair_observation")
                  << " capture=" << (result.flydelta_capture && result.flydelta_capture->captured ? "yes" : "no")
                  << " output=" << preview(result) << '\n';
    }
    std::cout << "flydelta_dataset_question_model_smoke scenarios=" << suite["scenarios"].size()
              << " matched=" << matched << " mismatched=" << mismatched
              << " learning_evidence=none_without_host_verification\n";
    return value.strict && mismatched != 0 ? 1 : 0;
}
