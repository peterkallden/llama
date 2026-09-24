#include "agent/adaptation/flydelta/flydelta-capture.h"
#include "agent/tooling/catalog/tool-catalog.h"
#include "agent/tooling/schema/tool-schema-compact.h"
#include "tools/agent/cli/agent-cli-inference.h"
#include "tools/agent/runtime/agent-model-loaders.h"

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
    common_agent_model_selection selection;
    selection.profile_id = "flydelta-dataset-question-suite";
    selection.base_model_id = "generation-base";
    selection.backend = "cli";
    selection.path = value.model;
    selection.context_size_tokens = 2048;
    selection.load_policy = "resident";
    common_agent_runtime_cli_model_loader loader({value.n_gpu_layers, value.n_threads, true});
    std::shared_ptr<common_agent_runtime_resident_model> resident;
    if (!loader.load(selection, resident, error)) { std::cerr << "model load failed: " << error << '\n'; return 1; }
    const auto loaded = common_agent_runtime_loaded_model_cast(resident);
    if (!loaded || !loaded->model || !loaded->chat_templates) return 1;
    auto inference = make_llama_cli_agent_inference(loaded->model, loaded->chat_templates.get());

    std::vector<common_chat_tool> chat_tools;
    chat_tools.reserve(suite["scenarios"].size());
    for (const auto & scenario : suite["scenarios"]) {
        const auto name = scenario.value("expected_tool", "");
        const auto * definition = catalog.find_definition(name);
        if (!definition) {
            std::cerr << "unknown expected tool: " << name << '\n';
            return 1;
        }
        std::string compact_error;
        const auto description = common_render_compact_tool_description(
            definition->name, definition->description,
            common_tool_model_input_schema(*definition),
            common_tool_model_result_schema(*definition), compact_error);
        if (!compact_error.empty()) {
            std::cerr << compact_error << '\n';
            return 1;
        }
        if (std::none_of(chat_tools.begin(), chat_tools.end(), [&](const auto & tool) {
                return tool.name == definition->name;
            })) {
            chat_tools.push_back({definition->name, description,
                common_tool_model_input_schema(*definition),
                common_tool_model_result_schema(*definition)});
        }
    }

    auto capture = std::make_shared<common_flydelta_hidden_state_capture_request>();
    capture->enabled = true;
    const auto layers = static_cast<uint32_t>(llama_model_n_layer(loaded->model));
    for (uint32_t layer = 1; layer < layers && layer <= 4; ++layer) capture->layer_indices.push_back(layer);
    capture->token_index = -1;
    capture->max_bytes = 4U * 1024U * 1024U;
    capture->model_profile_fingerprint = "sha256:flydelta-dataset-question-suite";
    capture->capture_layout_revision = "layer-input:v1";

    size_t matched = 0;
    size_t mismatched = 0;
    for (const auto & scenario : suite["scenarios"]) {
        common_agent_generation_request request;
        request.purpose = common_agent_generation_purpose::tool_followup;
        request.options.n_predict = value.n_predict;
        request.options.n_threads = value.n_threads;
        request.options.generation_trace = true;
        request.tools = chat_tools;
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
        const auto expected = scenario.value("expected_tool", "");
        const bool correct = executed && actual == expected;
        if (correct) ++matched; else ++mismatched;
        std::cout << "scenario=" << scenario.value("id", "")
                  << " expected=" << expected << " selected=" << (actual.empty() ? "<none>" : actual)
                  << " outcome=" << (correct ? "matched" : "repair_observation")
                  << " capture=" << (result.flydelta_capture && result.flydelta_capture->captured ? "yes" : "no")
                  << " output=" << preview(result) << '\n';
    }
    std::cout << "flydelta_dataset_question_model_smoke scenarios=" << suite["scenarios"].size()
              << " matched=" << matched << " mismatched=" << mismatched
              << " learning_evidence=none_without_host_verification\n";
    return value.strict && mismatched != 0 ? 1 : 0;
}
