#include "agent-flydelta-dataset-repair-host.h"

#include "agent/adaptation/flydelta/flydelta-capture.h"
#include "agent/adaptation/flydelta/flydelta-direction-search.h"
#include "agent/adaptation/flydelta/flydelta-evidence.h"
#include "agent/adaptation/flydelta/flydelta-aggregation.h"
#include "agent/adaptation/flydelta/flydelta-evidence-depth.h"
#include "agent/adaptation/flydelta/flydelta-activation.h"
#include "agent/adaptation/flydelta/flydelta-alpha-response-search.h"
#include "agent/adaptation/flydelta/flydelta-evaluator.h"
#include "agent/adaptation/flydelta/flydelta-deep-search.h"
#include "agent/adaptation/flydelta/flydelta-model-adapter.h"
#include "agent/adaptation/flydelta/flydelta-search-pipeline.h"
#include "agent/adaptation/flydelta/flydelta-experiment.h"
#include "agent/adaptation/flydelta/flydelta-worker.h"
#include "agent/adaptation/learning-transaction.h"
#include "agent/tooling/schema/tool-schema-compact.h"
#include "tools/agent/cli/agent-cli-inference.h"
#include "tools/agent/host/agent-host-config.h"
#include "tools/agent/runtime/agent-model-loaders.h"
#include "tools/agent/runtime/agent-server-context-host.h"
#include "tools/server/server-context.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace {
using json = nlohmann::ordered_json;

struct options {
    std::string model;
    std::string suite;
    std::string config;
    std::string learning_ledger;
    bool force_deep = false;
    bool force_tfo_lite = false;
    bool adaptive_alpha_search = false;
    int n_predict = 128;
    int n_threads = 3;
    int n_gpu_layers = 0;
    std::string backend = "cli";
    std::string scenario_id;
    size_t max_scenarios = 0;
};

struct host_verdict {
    enum class kind { passed, certified_failure, unresolved_alternative } value = kind::certified_failure;
    std::string selected_tool;
    std::string diagnostic;
    bool schema_valid = false;
    bool executable = false;
    bool tool_matches = false;
    bool normalized_call_matches = false;
};

bool parse_args(int argc, char ** argv, options & value) {
    if (const char * model = std::getenv("LLAMA_AGENT_MODEL")) value.model = model;
    if (const char * suite = std::getenv("LLAMA_AGENT_DATASET_QUESTION_SUITE")) value.suite = suite;
    if (const char * config = std::getenv("LLAMA_AGENT_CONFIG")) value.config = config;
    if (const char * ledger = std::getenv("LLAMA_AGENT_LEARNING_LEDGER")) value.learning_ledger = ledger;
    if (const char * force_deep = std::getenv("LLAMA_AGENT_FORCE_DEEP")) {
        value.force_deep = std::string(force_deep) == "1" || std::string(force_deep) == "true";
    }
    if (const char * force_tfo = std::getenv("LLAMA_AGENT_FORCE_TFO_LITE")) {
        value.force_tfo_lite = std::string(force_tfo) == "1" || std::string(force_tfo) == "true";
        value.force_deep = value.force_deep || value.force_tfo_lite;
    }
    if (const char * adaptive_alpha = std::getenv("LLAMA_AGENT_ADAPTIVE_ALPHA_SEARCH")) {
        value.adaptive_alpha_search = std::string(adaptive_alpha) == "1" ||
            std::string(adaptive_alpha) == "true";
    }
    if (const char * threads = std::getenv("LLAMA_AGENT_THREADS")) value.n_threads = std::stoi(threads);
    if (const char * backend = std::getenv("LLAMA_AGENT_BACKEND")) value.backend = backend;
    if (const char * scenario_id = std::getenv("LLAMA_AGENT_SCENARIO_ID")) value.scenario_id = scenario_id;
    if (const char * max_scenarios = std::getenv("LLAMA_AGENT_MAX_SCENARIOS")) value.max_scenarios = std::stoul(max_scenarios);
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        const auto next = [&](const char * name) -> const char * {
            if (index + 1 >= argc) { std::cerr << "missing value for " << name << '\n'; return nullptr; }
            return argv[++index];
        };
        if (argument == "--model") { const auto v = next("--model"); if (!v) return false; value.model = v; }
        else if (argument == "--suite") { const auto v = next("--suite"); if (!v) return false; value.suite = v; }
        else if (argument == "--config") { const auto v = next("--config"); if (!v) return false; value.config = v; }
        else if (argument == "--learning-ledger") { const auto v = next("--learning-ledger"); if (!v) return false; value.learning_ledger = v; }
        else if (argument == "--force-deep") value.force_deep = true;
        else if (argument == "--force-tfo-lite") { value.force_tfo_lite = true; value.force_deep = true; }
        else if (argument == "--adaptive-alpha-search") value.adaptive_alpha_search = true;
        else if (argument == "--n-predict") { const auto v = next("--n-predict"); if (!v) return false; value.n_predict = std::stoi(v); }
        else if (argument == "--threads") { const auto v = next("--threads"); if (!v) return false; value.n_threads = std::stoi(v); }
        else if (argument == "--n-gpu-layers") { const auto v = next("--n-gpu-layers"); if (!v) return false; value.n_gpu_layers = std::stoi(v); }
        else if (argument == "--backend") { const auto v = next("--backend"); if (!v) return false; value.backend = v; }
        else if (argument == "--scenario-id") { const auto v = next("--scenario-id"); if (!v) return false; value.scenario_id = v; }
        else if (argument == "--max-scenarios") { const auto v = next("--max-scenarios"); if (!v) return false; value.max_scenarios = std::stoul(v); }
        else if (argument == "--help" || argument == "-h") return false;
        else { std::cerr << "unknown argument: " << argument << '\n'; return false; }
    }
    return true;
}

struct bootstrap_case {
    std::string id;
    std::string behavior_key;
    std::string expected_tool;
    std::string question;
    std::string capture_manifest_id;
    std::string delta_id;
    json canonical_arguments = json::object();
    std::string verification_mode = "normalized_call";
    common_tool_execution_result canonical_execution;
    // The actual failed tool is the negative side of this fixture's
    // behavior-specific teacher-forced choice pair. It is deliberately not
    // a global data.inspect/data.describe assumption.
    std::string failed_tool;
    std::string failed_continuation;
};

std::string tool_family(const std::string & tool_name) {
    const auto separator = tool_name.find_first_of("._");
    return separator == std::string::npos ? tool_name : tool_name.substr(0, separator);
}

bool policy_allows_tool_family(const agent_host_config * config, const std::string & tool_name) {
    if (config == nullptr) return true;
    if (!config->adaptation_collection_allowed) return false;
    const auto & policy = config->adaptation_domains;
    if (!policy.configured) return true;
    const auto family = tool_family(tool_name);
    const auto override = policy.tool_use_families.find(family);
    return override == policy.tool_use_families.end() ? policy.tool_use : override->second;
}

bool read_json(const std::string & path, json & value, std::string & error) {
    std::ifstream input(path, std::ios::binary);
    if (!input) { error = "could not open suite: " + path; return false; }
    std::ostringstream contents;
    contents << input.rdbuf();
    const std::string text = contents.str();
    value = json::parse(text, nullptr, false);
    if (!value.is_discarded() && value.is_object() && value.contains("scenarios") &&
            value["scenarios"].is_array()) {
        return true;
    }

    // The replay fixture is JSONL because it is also a durable, append-only
    // learning input. Normalize it to the model smoke's existing suite shape
    // so the host repair, capture and worker pipeline remain shared.
    json scenarios = json::array();
    std::istringstream lines(text);
    std::string line;
    size_t line_number = 0;
    while (std::getline(lines, line)) {
        ++line_number;
        if (line.empty()) continue;
        const auto entry = json::parse(line, nullptr, false);
        if (entry.is_discarded() || !entry.is_object() ||
                !entry.contains("id") || !entry.contains("question") ||
                !entry.contains("expected_tool") || !entry.contains("canonical_repair") ||
                !entry["canonical_repair"].is_object() ||
                !entry["canonical_repair"].contains("arguments")) {
            error = "dataset repair replay is invalid at line " + std::to_string(line_number);
            return false;
        }
        json scenario = entry;
        json steps = json::array();
        steps.push_back({
            {"tool", entry["expected_tool"]},
            {"args", entry["canonical_repair"]["arguments"]},
        });
        scenario["plan"] = {
            {"goal", "Replay a host-constructed canonical repair from JSONL"},
            {"steps", std::move(steps)},
        };
        scenarios.push_back(std::move(scenario));
    }
    if (scenarios.empty()) {
        error = "dataset repair suite or replay is invalid";
        return false;
    }
    value = {
        {"schema_version", 1},
        {"id", "flydelta://fixture/dataset-repair-replay"},
        {"dataset", "dataset://local/sales"},
        {"replay_mode", true},
        {"scenarios", std::move(scenarios)},
    };
    return true;
}

std::string preview(const common_agent_generation_result & result) {
    if (!common_agent_generation_succeeded(result)) return "<generation failed: " + result.error_message + ">";
    std::string output = result.content;
    for (char & character : output) if (character == '\n' || character == '\r' || character == '\t') character = ' ';
    if (output.size() > 300) output.resize(300);
    return output;
}

bool parse_model_tool_call(const common_agent_generation_result & result, json & parsed) {
    if (!common_agent_generation_succeeded(result)) return false;
    parsed = json::parse(result.content, nullptr, false);
    if (parsed.is_discarded()) {
        const size_t first_object = result.content.find('{');
        const size_t last_object = result.content.rfind('}');
        if (first_object != std::string::npos && last_object > first_object) {
            parsed = json::parse(result.content.substr(first_object,
                last_object - first_object + 1), nullptr, false);
        }
    }
    if (!parsed.is_object() || !parsed.contains("name") || !parsed["name"].is_string()) {
        return false;
    }
    // The host's canonical repair uses "arguments", while several model
    // tool-call serializers use the equivalent short form "args". Keep the
    // host verifier strict and normalize this smoke-boundary representation
    // before the verifier sees it.
    if (!parsed.contains("arguments") && parsed.contains("args") &&
            parsed["args"].is_object()) {
        parsed["arguments"] = parsed["args"];
    }
    return parsed.contains("arguments") && parsed["arguments"].is_object();
}

std::string tool_call_continuation(const std::string & tool, const json & arguments) {
    if (tool.empty() || !arguments.is_object()) return {};
    return tool + "\",\"arguments\":" + arguments.dump() + "}";
}

std::string parsed_tool_call_continuation(const common_agent_generation_result & result) {
    json parsed;
    if (!parse_model_tool_call(result, parsed)) return {};
    return tool_call_continuation(parsed["name"].get<std::string>(), parsed["arguments"]);
}

bool result_oracle_matches(
        const common_tool_execution_result & actual,
        const common_tool_execution_result & expected) {
    if (!actual.ok || !expected.ok) return false;
    const json actual_json = json::parse(actual.output, nullptr, false);
    const json expected_json = json::parse(expected.output, nullptr, false);
    if (!actual_json.is_discarded() && !expected_json.is_discarded()) {
        return actual_json == expected_json;
    }
    return actual.output == expected.output;
}

host_verdict verify_model_call(const common_agent_generation_result & result,
        const std::string & expected_tool, const json & expected_arguments,
        const std::string & verification_mode,
        const common_tool_execution_result * expected_execution,
        agent_flydelta_dataset_repair_host & host) {
    host_verdict verdict;
    if (!common_agent_generation_succeeded(result)) {
        verdict.diagnostic = "model generation failed";
        return verdict;
    }
    json parsed;
    if (!parse_model_tool_call(result, parsed)) {
        verdict.diagnostic = "host rejected a non-canonical tool-call object";
        return verdict;
    }
    verdict.selected_tool = parsed["name"].get<std::string>();
    json normalized_arguments;
    std::string normalize_error;
    if (!host.normalize_call(verdict.selected_tool, parsed["arguments"], normalized_arguments, normalize_error)) {
        verdict.diagnostic = "host rejected tool arguments: " + normalize_error;
        return verdict;
    }
    verdict.schema_valid = true;
    common_tool_execution_result execution;
    std::string error;
    if (!host.execute_call(verdict.selected_tool, parsed["arguments"], execution, error)) {
        verdict.diagnostic = "host rejected tool call: " + error;
        return verdict;
    }
    verdict.executable = true;
    verdict.tool_matches = verdict.selected_tool == expected_tool;
    if (verdict.selected_tool != expected_tool) {
        // A different executable read-only tool can still be semantically
        // useful. This fixture has no equivalence oracle, so it remains
        // UNKNOWN rather than creating false tool-repair evidence.
        verdict.value = host_verdict::kind::unresolved_alternative;
        verdict.diagnostic = "different executable tool; semantic equivalence is unknown";
        return verdict;
    }
    json normalized_expected;
    if (!host.normalize_call(expected_tool, expected_arguments, normalized_expected, normalize_error)) {
        verdict.diagnostic = "host canonical repair is invalid: " + normalize_error;
        return verdict;
    }
    verdict.normalized_call_matches = normalized_arguments == normalized_expected;
    if (verification_mode == "normalized_call" && !verdict.normalized_call_matches) {
        verdict.diagnostic = "expected tool executed, but normalized arguments differ from canonical repair";
        return verdict;
    }
    if (verification_mode == "result_oracle") {
        if (expected_execution == nullptr || !result_oracle_matches(execution, *expected_execution)) {
            verdict.diagnostic = "expected tool executed, but result oracle did not match";
            return verdict;
        }
    }
    verdict.value = host_verdict::kind::passed;
    verdict.diagnostic = verification_mode == "tool_only"
        ? "host executed the expected tool"
        : verification_mode == "result_oracle"
            ? "host executed the expected tool and result matched the oracle"
            : "host executed the expected canonical normalized call";
    return verdict;
}

common_learning_transaction transaction(const std::string & id, const std::string & turn_id,
        common_learning_signal_type signal, const std::string & tool, const std::string & evidence) {
    common_learning_transaction value;
    value.id = id;
    value.created_at = "2026-09-15T00:00:00Z";
    value.observation.id = id;
    value.observation.scope.namespace_id = "local";
    value.observation.scope.session_id = "flydelta-dataset-repair";
    value.observation.scope.project_id = "dataset-question-suite";
    value.observation.scope.turn_id = turn_id;
    value.observation.source_turn_id = turn_id;
    value.observation.source_plan_id = "plan:" + turn_id;
    value.observation.signals.push_back({signal, value.observation.source_plan_id,
        signal == common_learning_signal_type::tool_failure ? "failed" : "repaired",
        tool, evidence, "host-verified dataset tool repair", "data", "native"});
    value.observation.evidence_ids = {evidence};
    value.observation.cause = common_learning_cause::model_behavior;
    value.observation.verification = common_learning_verification::host_verified;
    value.observation.collection_allowed = true;
    value.observation.idempotency_key = id + ":idempotency";
    value.observation.content_hash = common_learning_observation_hash(value.observation);
    return value;
}

common_agent_generation_request make_request(const options & value, const std::string & contract,
        const std::string & user, const std::shared_ptr<const common_flydelta_hidden_state_capture_request> & capture,
        std::shared_ptr<const common_flydelta_activation_result> activation = {}) {
    common_agent_generation_request request;
    request.purpose = common_agent_generation_purpose::tool_followup;
    request.options.n_predict = value.n_predict;
    request.options.n_threads = value.n_threads;
    request.messages = {{"system", "You are a host-controlled dataset tool selector. Return only one canonical JSON object with name and arguments, no markdown.\n" + contract}, {"user", user}};
    request.flydelta_capture = capture;
    request.flydelta_activation = std::move(activation);
    return request;
}
}

int main(int argc, char ** argv) {
    options value;
    if (!parse_args(argc, argv, value)) {
        std::cerr << "usage: " << argv[0]
                  << " --model MODEL --suite SUITE_JSON [--config AGENT_CONFIG]"
                  << " [--backend cli|server]"
                  << " [--scenario-id ID]"
                  << " [--max-scenarios N]"
                  << " [--learning-ledger JSONL]"
                  << " [--force-deep|--force-tfo-lite] [--adaptive-alpha-search]"
                  << " [--threads N] [--n-gpu-layers N]\n";
        return 2;
    }
    if (value.model.empty() || !std::filesystem::is_regular_file(value.model)) {
        std::cerr << "dataset repair model smoke skipped: provide --model or LLAMA_AGENT_MODEL\n";
        return 77;
    }
    if (value.suite.empty() || !std::filesystem::is_regular_file(value.suite) ||
            value.n_threads <= 0 || value.n_threads > 3 || value.n_predict <= 0) return 2;

    json suite;
    std::string error;
    if (!read_json(value.suite, suite, error)) { std::cerr << error << '\n'; return 1; }
    std::optional<agent_host_config> host_config;
    if (!value.config.empty()) {
        host_config.emplace();
        if (!load_agent_host_config(value.config, *host_config, error)) {
            std::cerr << "could not load agent config: " << error << '\n';
            return 1;
        }
    }
    if (value.backend != "cli" && value.backend != "server") {
        std::cerr << "backend must be cli or server\n";
        return 2;
    }

    agent_flydelta_dataset_repair_host host;
    if (!host.open("model", error)) { std::cerr << error << '\n'; return 1; }

    common_tool_catalog catalog;
    common_tool_bootstrap_result bootstrap;
    if (!catalog.bootstrap("analysis", bootstrap, error)) { host.close(); std::cerr << error << '\n'; return 1; }
    std::string contract = "Available read-only tools:\n";
    size_t selected_scenarios = 0;
    for (const auto & scenario : suite["scenarios"]) {
        const auto expected = scenario.value("expected_tool", "");
        if (!policy_allows_tool_family(host_config ? &*host_config : nullptr, expected)) continue;
        ++selected_scenarios;
        const auto * definition = catalog.find_definition(expected);
        if (!definition) { host.close(); std::cerr << "unknown expected tool: " << expected << '\n'; return 1; }
        std::string compact_error;
        const auto description = common_render_compact_tool_description(definition->name, definition->description,
            common_tool_model_input_schema(*definition), common_tool_model_result_schema(*definition), compact_error);
        if (!compact_error.empty()) { host.close(); std::cerr << compact_error << '\n'; return 1; }
        if (contract.find("\n- " + expected + "\n") == std::string::npos) contract += "- " + description + "\n";
    }
    if (selected_scenarios == 0) {
        host.close();
        std::cerr << "no scenarios enabled by agent learning policy\n";
        return 2;
    }

    std::unique_ptr<common_agent_server_context_host> server_host;
    std::unique_ptr<common_agent_inference> inference;
    std::shared_ptr<common_agent_runtime_resident_model> resident;
    size_t n_embd = 0;
    size_t n_layers = 0;
    if (value.backend == "cli") {
        common_agent_model_selection selection;
        selection.profile_id = "flydelta-dataset-question-repair";
        selection.base_model_id = "generation-base";
        selection.backend = "cli";
        selection.path = value.model;
        selection.context_size_tokens = 2048;
        selection.load_policy = "resident";
        common_agent_runtime_cli_model_loader loader({value.n_gpu_layers, value.n_threads, true});
        if (!loader.load(selection, resident, error)) { host.close(); std::cerr << error << '\n'; return 1; }
        const auto loaded = common_agent_runtime_loaded_model_cast(resident);
        if (!loaded || !loaded->model || !loaded->chat_templates) { host.close(); return 1; }
        inference = make_llama_cli_agent_inference(loaded->model, loaded->chat_templates.get());
        n_embd = static_cast<size_t>(llama_model_n_embd(loaded->model));
        n_layers = static_cast<size_t>(llama_model_n_layer(loaded->model));
    } else {
        server_host = std::make_unique<common_agent_server_context_host>();
        common_agent_server_context_host_config server_config;
        server_config.context_key.load_key.model = value.model;
        server_config.context_key.load_key.n_gpu_layers = value.n_gpu_layers;
        server_config.context_key.load_key.fit_params = true;
        server_config.context_key.n_parallel = 1;
        server_config.context_key.n_sequences = 1;
        server_config.context_key.n_ctx = 2048;
        server_config.context_key.n_threads = value.n_threads;
        if (!server_host->start(server_config, error)) {
            host.close();
            std::cerr << "could not start server backend: " << error << '\n';
            return 1;
        }
        common_agent_inference_session session;
        if (!server_host->build_inference_session(session, error) || !session.inference) {
            host.close();
            std::cerr << "could not build server inference session: " << error << '\n';
            return 1;
        }
        inference = std::move(session.inference);
        auto * context = server_host->server().get_llama_context();
        if (context == nullptr || llama_get_model(context) == nullptr) {
            host.close();
            std::cerr << "server backend did not expose a loaded model\n";
            return 1;
        }
        n_embd = static_cast<size_t>(llama_model_n_embd(llama_get_model(context)));
        n_layers = static_cast<size_t>(llama_model_n_layer(llama_get_model(context)));
    }
    if (n_embd == 0 || n_layers <= 2) { host.close(); return 1; }

    auto capture = std::make_shared<common_flydelta_hidden_state_capture_request>();
    capture->enabled = true;
    // Use the full bounded layer profile. Layer discovery/Whirlpool selects
    // intervention anchors from these captures; it must not be constrained
    // to the early-layer smoke subset.
    for (uint32_t layer = 1; layer < n_layers; ++layer) capture->layer_indices.push_back(layer);
    capture->token_index = -1;
    capture->position = common_flydelta_capture_position::generation_boundary;
    capture->max_bytes = 4U * 1024U * 1024U;
    capture->model_profile_fingerprint = "sha256:flydelta-dataset-question-repair";
    capture->capture_layout_revision = "layer-input:generation-boundary:v1";

    std::filesystem::path learning_ledger_path;
    bool remove_learning_ledger = false;
    if (value.learning_ledger.empty()) {
        learning_ledger_path = std::filesystem::temp_directory_path() /
            "llama-agent-flydelta-repair-echo-learning.jsonl";
        remove_learning_ledger = true;
        std::error_code remove_error;
        std::filesystem::remove(learning_ledger_path, remove_error);
    } else {
        learning_ledger_path = value.learning_ledger;
    }
    common_learning_jsonl_transaction_store learning_ledger;
    if (!learning_ledger.open(learning_ledger_path, error)) {
        host.close();
        std::cerr << "could not open learning ledger: " << error << '\n';
        return 1;
    }

    std::map<std::string, std::vector<common_flydelta_contrast_sample>> samples_by_behavior;
    std::map<std::string, bootstrap_case> bootstrap_cases_by_delta;
    std::map<std::string, std::vector<common_flydelta_behavior_delta>> deltas_by_case;
    size_t passed = 0, unresolved = 0, failures = 0, repaired = 0, synthetic_repairs = 0,
        repair_echo_failures = 0;
    const bool replay_mode = suite.value("replay_mode", false);
    size_t processed_scenarios = 0;
    for (const auto & scenario : suite["scenarios"]) {
        const auto id = scenario.value("id", "unnamed");
        if (!value.scenario_id.empty() && id != value.scenario_id) continue;
        if (value.max_scenarios != 0 && processed_scenarios >= value.max_scenarios) break;
        ++processed_scenarios;
        const auto expected = scenario.value("expected_tool", "");
        const auto behavior_key = scenario.value(
            "behavior_key", "tool_use/dataset/structured_call_repair");
        if (!policy_allows_tool_family(host_config ? &*host_config : nullptr, expected)) continue;
        const auto steps = scenario.value("plan", json::object()).value("steps", json::array());
        if (steps.size() != 1) { host.close(); std::cerr << "scenario has invalid plan: " << id << '\n'; return 1; }
        common_tool_execution_result canonical_execution;
        if (!host.execute_step(steps.front(), canonical_execution, error)) {
            std::cout << "scenario=" << id << " host_verifier=unavailable error=" << error << '\n';
            continue;
        }
        const json canonical_repair = {{"name", expected}, {"arguments", steps.front()["args"]}};
        const std::string verification_mode = scenario.value("verification_mode", "normalized_call");
        if (verification_mode != "tool_only" && verification_mode != "normalized_call" &&
                verification_mode != "result_oracle") {
            host.close();
            std::cerr << "scenario has invalid verification_mode: " << id << '\n';
            return 1;
        }

        common_agent_generation_result failed_result;
        const bool generated = inference->generate(make_request(value, contract, scenario.value("question", ""), capture), failed_result);
        const auto failed_verdict = verify_model_call(
            failed_result, expected, canonical_repair["arguments"], verification_mode,
            &canonical_execution, host);
        std::string decision_failed_tool = failed_verdict.selected_tool;
        std::string decision_failed_continuation = parsed_tool_call_continuation(failed_result);
        // Replay fixtures deliberately carry a host-constructed negative call.
        // Use it only when Qwen's failed output cannot be parsed into a
        // model-facing choice pair; this keeps experimental margin scoring
        // available without turning the synthetic fixture into learning
        // credit.
        if (replay_mode && (decision_failed_tool.empty() || decision_failed_continuation.empty()) &&
                scenario.contains("failed_output") && scenario["failed_output"].is_object()) {
            const auto & failed_fixture = scenario["failed_output"];
            const auto fixture_tool = failed_fixture.value("name", "");
            const json * fixture_arguments = nullptr;
            if (failed_fixture.contains("arguments")) fixture_arguments = &failed_fixture["arguments"];
            else if (failed_fixture.contains("args")) fixture_arguments = &failed_fixture["args"];
            if (!fixture_tool.empty() && fixture_arguments != nullptr && fixture_arguments->is_object()) {
                decision_failed_tool = fixture_tool;
                decision_failed_continuation = tool_call_continuation(fixture_tool, *fixture_arguments);
            }
        }
        if (generated && failed_verdict.value == host_verdict::kind::passed) {
            ++passed;
            std::cout << "scenario=" << id << " host_outcome=passed selected=" << failed_verdict.selected_tool
                      << " output=" << preview(failed_result) << '\n';
            continue;
        }
        if (failed_verdict.value == host_verdict::kind::unresolved_alternative) {
            ++unresolved;
            std::cout << "scenario=" << id << " host_outcome=unknown selected=" << failed_verdict.selected_tool
                      << " reason=" << failed_verdict.diagnostic << '\n';
            continue;
        }
        ++failures;
        const std::string repair_request = "The host rejected the prior tool-call attempt (" +
            failed_verdict.diagnostic + "). The host constructed the canonical repair below. " +
            "Return this JSON object unchanged, including both name and arguments, with no markdown or explanation: " +
            canonical_repair.dump();
        common_agent_generation_result repaired_result;
        const bool repair_generated = inference->generate(make_request(value, contract, repair_request, capture), repaired_result);
        const auto repaired_verdict = verify_model_call(
            repaired_result, expected, canonical_repair["arguments"], verification_mode,
            &canonical_execution, host);
        const bool capture_pair = failed_result.flydelta_capture && repaired_result.flydelta_capture &&
            failed_result.flydelta_capture->captured && repaired_result.flydelta_capture->captured;
        // Replay fixtures are deliberately allowed to contribute experimental
        // model captures when the model produced a repair-shaped response and
        // both sides were captured. The replay ledger supplies the canonical
        // host repair, while the host verdict remains authoritative for
        // learning credit. This keeps half-synthetic algorithm evaluation
        // useful without turning it into HELPED evidence.
        const bool synthetic_repair = replay_mode && repair_generated && capture_pair;
        if ((!repair_generated || repaired_verdict.value != host_verdict::kind::passed || !capture_pair) &&
                !synthetic_repair) {
            const auto echo_evidence_id = "evidence:dataset-repair:" + id + ":repair-echo-failure";
            const json echo_context = {
                {"kind", "repair_echo_failure"},
                {"expected_tool", expected},
                {"canonical_repair", canonical_repair},
                {"failed_output", failed_result.content},
                {"repair_output", repaired_result.content},
                {"diagnostic", repaired_verdict.diagnostic},
                {"host_evaluated", true},
                {"verifier_known", repaired_verdict.value != host_verdict::kind::unresolved_alternative},
                {"host_outcome", "UNKNOWN"},
                {"learning_credit", false},
            };
            common_learning_transaction echo_transaction;
            echo_transaction.id = "learning://observation/repair-echo-failure/" + id;
            echo_transaction.created_at = "2026-09-17T00:00:00Z";
            echo_transaction.observation.id = echo_transaction.id;
            echo_transaction.observation.scope.namespace_id = "local";
            echo_transaction.observation.scope.session_id = "flydelta-dataset-repair";
            echo_transaction.observation.scope.project_id = "dataset-question-suite";
            echo_transaction.observation.scope.turn_id = "turn:dataset-repair:" + id;
            echo_transaction.observation.source_turn_id = echo_transaction.observation.scope.turn_id;
            echo_transaction.observation.source_plan_id = "plan:" + echo_transaction.observation.scope.turn_id;
            common_learning_signal echo_signal(
                common_learning_signal_type::repair_echo_failure,
                echo_transaction.observation.source_plan_id,
                "repair-echo:" + id,
                expected,
                echo_evidence_id,
                "model failed to replay the host-constructed canonical repair",
                tool_family(expected),
                "native");
            echo_signal.repair_context_json = echo_context.dump();
            echo_transaction.observation.signals.push_back(std::move(echo_signal));
            echo_transaction.observation.evidence_ids = {echo_evidence_id};
            echo_transaction.observation.cause = common_learning_cause::model_behavior;
            echo_transaction.observation.verification = common_learning_verification::unverified;
            echo_transaction.observation.idempotency_key = echo_transaction.id + ":idempotency";
            echo_transaction.observation.collection_allowed = true;
            echo_transaction.observation.content_hash = common_learning_observation_hash(
                echo_transaction.observation);
            if (!common_learning_transaction_validate(echo_transaction, 16, error) ||
                    !learning_ledger.append(echo_transaction, error)) {
                host.close();
                std::cerr << "could not persist repair-echo failure: " << error << '\n';
                return 1;
            }
            ++repair_echo_failures;
            std::cout << "scenario=" << id << " host_outcome=repair_failed selected=" << repaired_verdict.selected_tool
                      << " failed_output=" << preview(failed_result)
                      << " repaired_output=" << preview(repaired_result)
                      << " persisted=" << echo_transaction.id << '\n';
            continue;
        }

        if (synthetic_repair) {
            // The replay log supplies a host-constructed canonical repair, but
            // this branch deliberately does not claim host-certified learning
            // evidence. It only turns the real Qwen baseline/repair captures
            // into experimental material for layer discovery and search.
            common_flydelta_capture_manifest manifest;
            manifest.id = "flydelta://capture/dataset-repair-replay/" + id;
            manifest.observation_id = "learning://observation/repair-echo-failure/" + id;
            manifest.behavior_key = behavior_key;
            manifest.model_profile_fingerprint = capture->model_profile_fingerprint;
            manifest.template_fingerprint = "template:dataset-question-compact-v1";
            manifest.execution_context_fingerprint = "sha256:dataset-question-context-v1";
            manifest.positive_execution_ref = "execution:dataset-repair-replay:" + id + ":synthetic-repaired";
            manifest.negative_execution_ref = "execution:dataset-repair-replay:" + id + ":failed";
            manifest.capture_layout_revision = capture->capture_layout_revision;
            manifest.evidence_hash = "sha256:dataset-repair-replay:" + id;
            manifest.redaction_attested = true;
            manifest.captured_bytes = (failed_result.flydelta_capture->values.size() +
                repaired_result.flydelta_capture->values.size()) * sizeof(float);
            if (!common_flydelta_capture_manifest_validate(
                    manifest, 64U * 1024U * 1024U, error)) {
                host.close();
                std::cerr << "synthetic replay manifest failed: " << error << '\n';
                return 1;
            }
            std::vector<common_flydelta_behavior_delta> replay_deltas;
            if (!common_flydelta_behavior_deltas_from_captures(
                    manifest, *failed_result.flydelta_capture, *repaired_result.flydelta_capture,
                    "evidence:dataset-repair-replay:" + id,
                    64U * 1024U * 1024U, 64U * 1024U * 1024U, replay_deltas, error)) {
                host.close();
                std::cerr << "synthetic replay delta failed: " << error << '\n';
                return 1;
            }
            common_flydelta_intervention_credit credit;
            credit.experiment_id = "flydelta://replay-experiment/" + id;
            credit.candidate_id = "flydelta://replay-candidate/" + id;
            credit.fixture_id = "flydelta://dataset-question-replay/" + id;
            credit.outcome = common_flydelta_counterfactual_outcome::unknown;
            credit.quality_delta = 0.0f;
            credit.eligible_for_learning = false;
            for (const auto & delta : replay_deltas) {
                bootstrap_case case_record{
                    id, behavior_key, expected, scenario.value("question", ""), manifest.id, delta.id,
                    canonical_repair["arguments"], verification_mode, canonical_execution,
                    decision_failed_tool, decision_failed_continuation};
                bootstrap_cases_by_delta[delta.id] = std::move(case_record);
                deltas_by_case[id].push_back(delta);
                if (delta.layer_index == 2) samples_by_behavior[behavior_key].push_back({delta, credit});
            }
            ++synthetic_repairs;
            std::cout << "scenario=" << id << " host_outcome=synthetic_repair_observation"
                      << " selected=" << repaired_verdict.selected_tool
                      << " failed_output=" << preview(failed_result)
                      << " repaired_output=" << preview(repaired_result) << '\n';
            continue;
        }

        const auto turn = "turn:dataset-repair:" + id;
        const auto failed_transaction = transaction("learning://dataset-repair/" + id + "/failed", turn,
            common_learning_signal_type::tool_failure, failed_verdict.selected_tool,
            "evidence:dataset-repair:" + id + ":failed");
        const auto repaired_transaction = transaction("learning://dataset-repair/" + id + "/repaired", turn,
            common_learning_signal_type::successful_recovery, expected,
            "evidence:dataset-repair:" + id + ":repaired");
        common_flydelta_behavior_transition transition;
        if (!common_flydelta_tool_repair_transition_from_transactions(failed_transaction, repaired_transaction,
                "sha256:dataset-question:" + id, behavior_key,
                "execution:dataset-repair:" + id + ":failed", "execution:dataset-repair:" + id + ":repaired",
                "verifier:cozo-native-data-adapter-v1", transition, error)) {
            host.close(); std::cerr << "transition failed: " << error << '\n'; return 1;
        }
        common_flydelta_capture_manifest manifest;
        manifest.id = "flydelta://capture/dataset-repair/" + id;
        manifest.observation_id = repaired_transaction.id;
        manifest.behavior_key = transition.behavior_key;
        manifest.model_profile_fingerprint = capture->model_profile_fingerprint;
        manifest.template_fingerprint = "template:dataset-question-compact-v1";
        manifest.execution_context_fingerprint = "sha256:dataset-question-context-v1";
        manifest.positive_execution_ref = transition.candidate_execution_ref;
        manifest.negative_execution_ref = transition.baseline_execution_ref;
        manifest.capture_layout_revision = capture->capture_layout_revision;
        manifest.evidence_hash = "sha256:dataset-repair:" + id;
        manifest.redaction_attested = true;
        manifest.captured_bytes = (failed_result.flydelta_capture->values.size() +
            repaired_result.flydelta_capture->values.size()) * sizeof(float);
        std::vector<common_flydelta_behavior_delta> deltas;
        if (!common_flydelta_behavior_deltas_from_captures(manifest, *failed_result.flydelta_capture,
                *repaired_result.flydelta_capture, "evidence:dataset-repair:" + id,
                64U * 1024U * 1024U, 64U * 1024U * 1024U, deltas, error)) {
            host.close(); std::cerr << "capture delta failed: " << error << '\n'; return 1;
        }
        common_flydelta_intervention_credit credit;
        credit.experiment_id = "flydelta://repair-observation/" + id;
        credit.candidate_id = "flydelta://repair-candidate/" + id;
        credit.fixture_id = "flydelta://dataset-question/" + id;
        credit.outcome = common_flydelta_counterfactual_outcome::helped;
        credit.quality_delta = 1.0f;
        credit.eligible_for_learning = true;
        for (const auto & delta : deltas) {
            bootstrap_case case_record{
                id, behavior_key, expected, scenario.value("question", ""), manifest.id, delta.id,
                canonical_repair["arguments"], verification_mode, canonical_execution,
                decision_failed_tool, decision_failed_continuation};
            bootstrap_cases_by_delta[delta.id] = std::move(case_record);
            deltas_by_case[id].push_back(delta);
            if (delta.layer_index != 2) continue;
            samples_by_behavior[behavior_key].push_back({delta, credit});
        }
        ++repaired;
        std::cout << "scenario=" << id << " host_outcome=host_certified_repair"
                  << " transition=" << transition.id << " selected=" << repaired_verdict.selected_tool
                  << " failed_output=" << preview(failed_result)
                  << " repaired_output=" << preview(repaired_result) << '\n';
    }
    common_flydelta_evidence_depth_config depth_config;
    depth_config.min_shallow_samples = 2;
    depth_config.min_deep_samples = 6;
    depth_config.max_samples = 32;
    depth_config.rank_relative_tolerance = 0.10f;
    depth_config.min_median_alignment = 0.25f;
    depth_config.max_condition_number = 100.0f;
    size_t layer2_samples = 0;
    size_t deep_groups = 0;
    size_t deep_search_executed = 0;
    size_t deep_diagnostic_trials = 0;
    size_t deep_full_generation_trials = 0;
    size_t direction_candidates = 0;
    const auto worker_root = std::filesystem::temp_directory_path() /
        "llama-agent-flydelta-dataset-question-bootstrap-worker";
    std::error_code ignored;
    std::filesystem::remove_all(worker_root, ignored);
    std::cout << "flydelta_dataset_question_repair_model_smoke"
              << " baseline_passed=" << passed
              << " baseline_failures=" << failures
              << " unresolved=" << unresolved
              << " config=" << (value.config.empty() ? "none" : value.config)
              << " selected_scenarios=" << selected_scenarios
              << " host_certified_repairs=" << repaired
              << " synthetic_repair_observations=" << synthetic_repairs
              << " repair_echo_failures=" << repair_echo_failures << '\n';
    for (const auto & entry : samples_by_behavior) {
        common_flydelta_direction_search_config direction_config;
        direction_config.dimension = n_embd;
        direction_config.layer_index = 2;
        direction_config.min_samples = 2;
        direction_config.max_samples = depth_config.max_samples;
        direction_config.min_median_alignment = depth_config.min_median_alignment;
        direction_config.trim_fraction = 0.20f;
        direction_config.variance_ridge = 0.001f;
        // Replay captures are real model-facing observations, but their
        // canonical repair comes from the host fixture and therefore has no
        // positive learning credit. Keep them available for experimental
        // direction/search work without admitting them to learning mode.
        direction_config.mode = replay_mode
            ? common_flydelta_direction_search_mode::experimental
            : common_flydelta_direction_search_mode::learning;
        direction_config.source = common_adaptation_evidence_source::tool_repair;
        direction_config.behavior_key = entry.first;
        direction_config.model_profile_fingerprint = capture->model_profile_fingerprint;
        direction_config.execution_context_fingerprint =
            "sha256:dataset-question-context-v1";
        direction_config.capture_layout_revision = capture->capture_layout_revision;

        common_flydelta_aggregation_config aggregation_config;
        aggregation_config.identity = direction_config;
        aggregation_config.depth = depth_config;
        aggregation_config.max_retained_samples = depth_config.max_samples;
        common_flydelta_incremental_aggregation aggregation(aggregation_config);
        for (const auto & sample : entry.second) {
            if (!aggregation.ingest(sample, error)) {
                std::cerr << "aggregation failed for " << entry.first << ": " << error << '\n';
                return 1;
            }
        }
        common_flydelta_evidence_depth_result evidence_depth;
        if (!aggregation.assess_depth(evidence_depth, error)) {
            std::cerr << "evidence depth failed for " << entry.first << ": " << error << '\n';
            return 1;
        }
        const auto search_budget = common_flydelta_search_budget_for_depth(evidence_depth.depth);
        std::vector<common_flydelta_direction_candidate> directions;
        if (!common_flydelta_build_direction_candidates(
                direction_config, aggregation.snapshot().retained_samples, directions, error)) {
            std::cerr << "direction search failed for " << entry.first << ": " << error << '\n';
            return 1;
        }
        layer2_samples += evidence_depth.compatible_samples;
        if (evidence_depth.deep_ready) ++deep_groups;
        direction_candidates += directions.size();
        std::cout << "flydelta_partition group=" << entry.first
                  << " observations=" << entry.second.size()
                  << " compatible=" << evidence_depth.compatible_samples
                  << " rejected=" << evidence_depth.incompatible_samples
                  << " depth=" << common_flydelta_search_depth_name(evidence_depth.depth)
                  << " shallow_ready=" << (evidence_depth.shallow_ready ? "yes" : "no")
                  << " deep_ready=" << (evidence_depth.deep_ready ? "yes" : "no")
                  << " effective_rank=" << evidence_depth.effective_rank
                  << " stable_rank=" << evidence_depth.stable_rank
                  << " median_alignment=" << evidence_depth.median_alignment
                  << " condition_number=" << evidence_depth.condition_number
                  << " geometry_stable=" << (evidence_depth.geometry_stable ? "yes" : "no")
                  << " region_budget=" << search_budget.max_region_trials
                  << " coefficient_budget=" << search_budget.max_coefficient_trials
                  << " tfo_lite=" << (search_budget.allow_tfo_lite ? "yes" : "no")
                  << " direction_candidates=" << directions.size() << '\n';
        for (const auto & direction : directions) {
            std::cout << "flydelta_direction group=" << entry.first
                      << " kind=" << common_flydelta_direction_kind_name(direction.kind)
                      << " source_samples=" << direction.source_samples
                      << " retained_samples=" << direction.retained_samples
                      << " median_alignment=" << direction.median_alignment
                      << " experimental_only=" << (direction.experimental_only ? "yes" : "no")
                      << '\n';
        }

        // The normal evidence gate remains authoritative. This explicit
        // smoke-only override evaluates Deep on a smaller group so the
        // algorithm and model-facing arm diagnostics can be inspected before
        // enough production evidence has accumulated.
        // --force-deep/--force-tfo-lite is an explicit smoke override.  It
        // must still execute when the fixture naturally reaches Deep; the
        // normal worker/orchestrator gate remains unchanged because this is
        // only the model-facing smoke path.
        if (value.force_deep && entry.second.size() >= 2 && directions.size() >= 2) {
            const auto aggregation_snapshot_for_deep = aggregation.snapshot();
            if (aggregation_snapshot_for_deep.retained_samples.size() < 2) {
                std::cerr << "forced Deep requires two retained samples for " << entry.first << '\n';
                host.close();
                return 1;
            }
            const auto deep_case_it = bootstrap_cases_by_delta.find(
                aggregation_snapshot_for_deep.retained_samples.front().delta.id);
            if (deep_case_it == bootstrap_cases_by_delta.end()) {
                std::cerr << "forced Deep sample has no model-facing fixture: " << entry.first << '\n';
                host.close();
                return 1;
            }
            const auto deep_case = deep_case_it->second;
            const auto deep_case_deltas_it = deltas_by_case.find(deep_case.id);
            if (deep_case_deltas_it == deltas_by_case.end() || deep_case_deltas_it->second.empty()) {
                std::cerr << "forced Deep sample has no per-layer model captures: "
                          << deep_case.id << '\n';
                host.close();
                return 1;
            }
            const auto & deep_case_deltas = deep_case_deltas_it->second;
            common_flydelta_experiment_fixture deep_fixture;
            deep_fixture.id = "flydelta://fixture/deep/" + deep_case.id;
            deep_fixture.task_fingerprint = "sha256:dataset-question:" + deep_case.id;
            deep_fixture.model_profile_fingerprint = capture->model_profile_fingerprint;
            deep_fixture.tokenizer_fingerprint = "sha256:dataset-question-tokenizer-v1";
            deep_fixture.template_fingerprint = "template:dataset-question-compact-v1";
            deep_fixture.execution_context_fingerprint = "sha256:dataset-question-context-v1";
            deep_fixture.verifier_revision = "verifier:cozo-native-data-adapter-v1";

            common_flydelta_deep_search_config deep_config;
            deep_config.max_rank = 2;
            deep_config.max_directions = 4;
            deep_config.full_generation_top_k = 3;
            deep_config.coefficients.strategy = value.force_tfo_lite
                ? common_flydelta_coefficient_search_strategy::tfo_lite
                : common_flydelta_coefficient_search_strategy::coordinate;
            deep_config.coefficients.step = 0.05f;
            deep_config.coefficients.max_candidates = 8;
            // Exercise the same bounded wave semantics used by a production
            // coefficient host. The smoke still permits scalar fallback, but
            // TFO must not bypass the batch runner or exceed this wave size.
            deep_config.coefficients.max_batch_arms = 2;
            deep_config.coefficients.max_l2_norm = 0.32f;
            deep_config.coefficients.norm_penalty = 0.05f;
            deep_config.coefficients.leakage_penalty = 0.10f;

            std::vector<common_flydelta_deep_search_direction> deep_directions;
            for (const auto & direction : directions) {
                deep_directions.push_back({direction, false, 0.0f});
            }
            std::shared_ptr<const common_flydelta_hidden_state_capture> deep_baseline_capture;
            std::optional<common_flydelta_decision_margin> deep_baseline_choice_margin;
            std::string deep_margin_error;
            auto run_deep_arm = [&](const common_flydelta_low_rank_basis & basis,
                    const std::vector<float> & coefficients, bool apply_overlay,
                    common_flydelta_counterfactual_trial & trial,
                    common_flydelta_decision_margin & margin,
                    common_flydelta_representation_diagnostics & geometry,
                    bool & geometry_available, std::string & runner_error) {
                common_agent_generation_result generated_result;
                geometry = {};
                geometry_available = false;
                std::shared_ptr<const common_flydelta_activation_result> activation_ptr;
                if (apply_overlay) {
                    common_flydelta_gate_request gate_request;
                    // The model smoke has no persisted recognition artifact.
                    // It uses the same explicit opt-in gate as the existing
                    // Bootstrap worker with a known familiar context.
                    gate_request.explicit_opt_in = true;
                    gate_request.candidate_status = common_flydelta_candidate_status::approved;
                    gate_request.basis_available = true;
                    gate_request.familiarity = 1.0f;
                    gate_request.novelty = 0.0f;
                    gate_request.requested_scale = 1.0f;
                    common_flydelta_activation_request activation_request;
                    activation_request.candidate_id = "flydelta://candidate/deep/" + deep_case.id;
                    activation_request.artifact_id = "flydelta://experimental/deep/" + deep_case.id;
                    activation_request.model_profile_fingerprint = capture->model_profile_fingerprint;
                    activation_request.capture_layout_revision = capture->capture_layout_revision;
                    activation_request.model_n_embd = n_embd;
                    activation_request.model_n_layers = n_layers;
                    activation_request.il_start = 1;
                    activation_request.il_end = static_cast<int32_t>(n_layers - 1);
                    for (const auto & vector : basis.vectors) {
                        activation_request.directions.push_back({basis.layer_index, vector});
                    }
                    activation_request.coefficients = coefficients;
                    gate_request.requested_scale = 1.0f;
                    activation_request.gate_request = gate_request;
                    common_flydelta_gate_config gate_config;
                    gate_config.enabled = true;
                    gate_config.max_scale = 1.0f;
                    common_flydelta_activation_result activation;
                    if (!common_flydelta_prepare_activation(
                            gate_config, activation_request, 64U * 1024U * 1024U,
                            activation, runner_error)) return false;
                    activation_ptr = std::make_shared<const common_flydelta_activation_result>(
                        std::move(activation));
                }
                generated_result = {};
                const bool generated = inference->generate(make_request(
                    value, contract, deep_case.question, capture, activation_ptr), generated_result);
                const auto verdict = verify_model_call(
                    generated_result, deep_case.expected_tool, deep_case.canonical_arguments,
                    deep_case.verification_mode, &deep_case.canonical_execution, host);
                trial = {};
                trial.executed = generated;
                trial.verifier_known = generated &&
                    verdict.value != host_verdict::kind::unresolved_alternative;
                trial.passed = generated && verdict.value == host_verdict::kind::passed;
                trial.quality = trial.passed ? 1.0f : 0.0f;
                trial.overlay_applied = apply_overlay;
                trial.intervention_count = apply_overlay ? basis.vectors.size() : 0;
                trial.evidence_ref = apply_overlay
                    ? "evidence:flydelta-deep-overlay" : "evidence:flydelta-deep-baseline";
                margin = {};
                deep_margin_error.clear();
                if (!deep_case.failed_tool.empty() || !deep_case.failed_continuation.empty()) {
                    common_agent_teacher_forced_choice_request score_request;
                    score_request.context = make_request(
                        value, contract, deep_case.question, capture, activation_ptr);
                    score_request.context.flydelta_capture.reset();
                    score_request.choice_prefix = "{\"name\":\"";
                    score_request.positive_choice = deep_case.expected_tool;
                    score_request.negative_choice = deep_case.failed_tool;
                    score_request.positive_continuation = tool_call_continuation(
                        deep_case.expected_tool, deep_case.canonical_arguments);
                    score_request.negative_continuation = deep_case.failed_continuation;
                    common_agent_teacher_forced_choice_result score_result;
                    if (inference->score_teacher_forced_choice(score_request, score_result) &&
                            score_result.available) {
                        margin.available = true;
                        margin.positive_total_logprob = score_result.positive_total_logprob;
                        margin.negative_total_logprob = score_result.negative_total_logprob;
                        margin.positive_token_count = score_result.positive_token_count;
                        margin.negative_token_count = score_result.negative_token_count;
                    } else {
                        deep_margin_error = score_result.error_message.empty()
                            ? "teacher-forced choice margin unavailable"
                            : score_result.error_message;
                    }
                } else {
                    deep_margin_error = "fixture has no distinct failed tool for decision pair";
                }
                if (!apply_overlay && margin.available) deep_baseline_choice_margin = margin;
                if (!apply_overlay && generated_result.flydelta_capture &&
                        generated_result.flydelta_capture->captured) {
                    deep_baseline_capture = generated_result.flydelta_capture;
                }
                if (apply_overlay && generated_result.flydelta_capture &&
                        generated_result.flydelta_capture->captured && deep_baseline_capture) {
                    // Captures use the layer-input layout: the requested
                    // layer is sampled before its own overlay injection.
                    // Measuring the repair delta at the same layer therefore
                    // reports the pre-injection state and makes every arm
                    // look like a zero shift. Use the first captured
                    // downstream layer, just as the normal region runner
                    // does, so coordinate and TFO diagnostics describe the
                    // actual propagated intervention.
                    const auto measurement_it = std::find_if(deep_case_deltas.begin(),
                        deep_case_deltas.end(), [&](const auto & delta) {
                            return delta.layer_index > basis.layer_index;
                        });
                    if (measurement_it == deep_case_deltas.end()) {
                        runner_error = "forced Deep has no downstream capture for geometry";
                        return false;
                    }
                    if (!common_flydelta_representation_diagnostics_from_captures(
                            *deep_baseline_capture, *generated_result.flydelta_capture,
                            *measurement_it,
                            64U * 1024U * 1024U, geometry, runner_error)) return false;
                    geometry_available = true;
                }
                std::cout << "flydelta_deep_model_output group=" << entry.first
                          << " overlay=" << (apply_overlay ? "yes" : "no")
                          << " output=" << preview(generated_result) << '\n';
                if (!generated && !generated_result.error_message.empty()) {
                    runner_error = generated_result.error_message;
                }
                return generated;
            };

            const common_flydelta_coefficient_search_runner deep_diagnostic_runner =
                [&](const common_flydelta_experiment_fixture & fixture,
                            const common_flydelta_low_rank_basis & basis,
                            const std::vector<float> & coefficients,
                            bool apply_overlay, common_flydelta_counterfactual_trial & trial,
                            common_flydelta_decision_margin & margin,
                            common_flydelta_representation_diagnostics & geometry,
                            bool & geometry_available, std::string & runner_error) {
                        const bool executed = run_deep_arm(basis, coefficients, apply_overlay,
                            trial, margin, geometry, geometry_available, runner_error);
                        if (apply_overlay) {
                            std::cout << "flydelta_deep_diagnostic group=" << entry.first
                                      << " coefficients=";
                            for (size_t index = 0; index < coefficients.size(); ++index) {
                                if (index != 0) std::cout << ',';
                                std::cout << coefficients[index];
                            }
                            std::cout << " executed=" << (executed ? "yes" : "no")
                                      << " outcome=diagnostic";
                            if (margin.available) {
                                const float delta_total = deep_baseline_choice_margin && apply_overlay
                                    ? margin.total_delta() - deep_baseline_choice_margin->total_delta() : 0.0f;
                                const float delta_normalized = deep_baseline_choice_margin && apply_overlay
                                    ? margin.normalized_delta() - deep_baseline_choice_margin->normalized_delta() : 0.0f;
                                std::cout << " margin_total=" << margin.total_delta()
                                          << " margin_normalized=" << margin.normalized_delta()
                                          << " margin_delta_total=" << delta_total
                                          << " margin_delta_normalized=" << delta_normalized;
                            } else {
                                std::cout << " margin_available=no";
                            }
                            if (geometry_available) {
                                std::cout << " cosine=" << geometry.cosine
                                          << " progress=" << geometry.progress
                                          << " leakage=" << geometry.leakage
                                          << " shift_norm=" << geometry.shift_norm;
                            }
                            if (!deep_margin_error.empty()) std::cout << " margin_error=" << deep_margin_error;
                            std::cout << '\n';
                        }
                        return executed;
                    };
            const common_flydelta_coefficient_search_runner deep_full_runner =
                [&](const common_flydelta_experiment_fixture & fixture,
                            const common_flydelta_low_rank_basis & basis,
                            const std::vector<float> & coefficients,
                            bool apply_overlay, common_flydelta_counterfactual_trial & trial,
                            common_flydelta_decision_margin & margin,
                            common_flydelta_representation_diagnostics & geometry,
                            bool & geometry_available, std::string & runner_error) {
                        return run_deep_arm(basis, coefficients, apply_overlay, trial, margin,
                            geometry, geometry_available, runner_error);
                    };
            const auto make_deep_batch_runner = [](
                    const common_flydelta_coefficient_search_runner & scalar_runner) {
                return [scalar_runner](
                        const common_flydelta_experiment_fixture & fixture,
                        const common_flydelta_low_rank_basis & basis,
                        const std::vector<std::vector<float>> & coefficients,
                        std::vector<common_flydelta_counterfactual_trial> & trials,
                        std::vector<common_flydelta_decision_margin> & margins,
                        std::vector<common_flydelta_representation_diagnostics> & geometries,
                        std::vector<bool> & geometry_available,
                        std::string & runner_error) {
                    trials.clear();
                    margins.clear();
                    geometries.clear();
                    geometry_available.clear();
                    trials.reserve(coefficients.size());
                    margins.reserve(coefficients.size());
                    geometries.reserve(coefficients.size());
                    geometry_available.reserve(coefficients.size());
                    for (const auto & values : coefficients) {
                        common_flydelta_counterfactual_trial trial;
                        common_flydelta_decision_margin margin;
                        common_flydelta_representation_diagnostics geometry;
                        bool has_geometry = false;
                        if (!scalar_runner(fixture, basis, values, true, trial, margin,
                                geometry, has_geometry, runner_error)) return false;
                        trials.push_back(std::move(trial));
                        margins.push_back(std::move(margin));
                        geometries.push_back(std::move(geometry));
                        geometry_available.push_back(has_geometry);
                    }
                    return true;
                };
            };
            const auto deep_diagnostic_batch_runner = make_deep_batch_runner(
                deep_diagnostic_runner);
            const auto deep_full_batch_runner = make_deep_batch_runner(deep_full_runner);
            common_flydelta_deep_search_result deep_result;
            const auto deep_started = std::chrono::steady_clock::now();
            if (!common_flydelta_run_deep_search_batched(
                    deep_fixture, deep_config, deep_directions,
                    deep_diagnostic_runner, deep_diagnostic_batch_runner,
                    deep_full_runner, deep_full_batch_runner, deep_result, error)) {
                host.close();
                std::cerr << "forced Deep search failed for " << entry.first << ": " << error << '\n';
                return 1;
            }
            ++deep_search_executed;
            size_t deep_group_full_generation_trials = 0;
            for (const auto & trial : deep_result.coefficient_trials) {
                if (trial.mutation_kind == "full_generation_top_arm") {
                    ++deep_group_full_generation_trials;
                    ++deep_full_generation_trials;
                }
            }
            const size_t deep_group_diagnostic_trials =
                deep_result.coefficient_trials.size() - deep_group_full_generation_trials;
            deep_diagnostic_trials += deep_group_diagnostic_trials;
            std::cout << "flydelta_deep_search group=" << entry.first
                      << " forced=yes strategy="
                      << common_flydelta_coefficient_search_strategy_name(
                          deep_config.coefficients.strategy)
                      << " basis_rank=" << deep_result.basis.vectors.size()
                      << " diagnostic_trials=" << deep_group_diagnostic_trials
                      << " full_generation_trials=" << deep_group_full_generation_trials
                      << " max_batch_arms=" << deep_config.coefficients.max_batch_arms
                      << " batch_runner=enabled_scalar_fallback"
                      << " selected=" << (deep_result.coefficient_selection.selected ? "yes" : "no")
                      << " elapsed_ms=" << std::chrono::duration_cast<std::chrono::milliseconds>(
                          std::chrono::steady_clock::now() - deep_started).count() << '\n';
            for (const auto & trial : deep_result.coefficient_trials) {
                std::cout << "flydelta_deep_arm group=" << entry.first << " coefficients=";
                for (size_t index = 0; index < trial.coefficients.size(); ++index) {
                    if (index != 0) std::cout << ',';
                    std::cout << trial.coefficients[index];
                }
                std::cout << " mutation=" << trial.mutation_kind
                          << " outcome=" << common_flydelta_counterfactual_outcome_name(trial.outcome)
                          << " host_verified=" << (trial.verifier_known ? "yes" : "no")
                          << " search_fitness=" << trial.search_fitness;
                if (trial.geometry_available) {
                    std::cout << " cosine=" << trial.geometry.cosine
                              << " progress=" << trial.geometry.progress
                              << " leakage=" << trial.geometry.leakage
                              << " shift_norm=" << trial.geometry.shift_norm;
                }
                std::cout << '\n';
            }
        }

        // The normal model-facing smoke goes through the same bounded
        // search-pipeline worker as production. Whirlpool supplies the WHERE
        // probes; this host runner executes the fresh model arms. Keeping the
        // runner here makes the smoke model-facing without moving policy into
        // the evaluator or recursively running later phases.
        const auto aggregation_snapshot = aggregation.snapshot();
        if (evidence_depth.depth == common_flydelta_search_depth::bootstrap &&
                !directions.empty() && !aggregation_snapshot.retained_samples.empty()) {
            const auto & sample = aggregation_snapshot.retained_samples.front();
            const auto case_it = bootstrap_cases_by_delta.find(sample.delta.id);
            if (case_it == bootstrap_cases_by_delta.end()) {
                host.close();
                std::cerr << "Bootstrap sample has no model-facing fixture: " << sample.delta.id << '\n';
                return 1;
            }
            const auto bootstrap_case = case_it->second;
            const auto bootstrap_direction = directions.front();
            const auto case_deltas_it = deltas_by_case.find(bootstrap_case.id);
            if (case_deltas_it == deltas_by_case.end() || case_deltas_it->second.empty()) {
                host.close();
                std::cerr << "Bootstrap sample has no per-layer model captures: "
                          << bootstrap_case.id << '\n';
                return 1;
            }
            const auto & case_deltas = case_deltas_it->second;
            common_flydelta_experiment_job job;
            job.id = "flydelta://job/search-pipeline/" + bootstrap_case.id;
            job.kind = common_flydelta_experiment_job_kind::search_pipeline;
            job.seed.id = "evidence://dataset-repair/" + bootstrap_case.id;
            job.seed.behavior_key = bootstrap_case.behavior_key;
            job.seed.source = common_adaptation_evidence_source::tool_repair;
            job.seed.scope.namespace_id = "local";
            job.seed.scope.project_id = "dataset-question-suite";
            job.seed.scope.session_id = "flydelta-dataset-repair";
            job.seed.scope.turn_id = "turn:dataset-repair:" + bootstrap_case.id;
            job.seed.task_fingerprint = "sha256:dataset-question:" + bootstrap_case.id;
            job.seed.model_profile_fingerprint = capture->model_profile_fingerprint;
            job.seed.tokenizer_fingerprint = "sha256:dataset-question-tokenizer-v1";
            job.seed.template_fingerprint = "template:dataset-question-compact-v1";
            job.seed.execution_context_fingerprint = "sha256:dataset-question-context-v1";
            job.seed.baseline_ref = "execution:dataset-repair:" + bootstrap_case.id + ":failed";
            job.seed.candidate_ref = "execution:dataset-repair:" + bootstrap_case.id + ":repaired";
            job.seed.verifier_ref = "verifier:cozo-native-data-adapter-v1";
            job.seed.evidence_ref = "evidence:dataset-repair:" + bootstrap_case.id;
            job.seed.transaction_ids = {
                "learning://dataset-repair/" + bootstrap_case.id + "/failed",
                "learning://dataset-repair/" + bootstrap_case.id + "/repaired"};
            job.capture_manifest_ids = {bootstrap_case.capture_manifest_id};
            job.behavior_delta_ids = {bootstrap_case.delta_id};
            job.alpha_search.candidates = {0.02f};
            job.alpha_search.max_candidates = 1;
            job.code_revision = "flydelta-dataset-question-search-pipeline:v1";

            common_flydelta_evaluator_config evaluator_config;
            common_flydelta_search_pipeline_config pipeline_config;
            pipeline_config.dimension = n_embd;
            pipeline_config.max_directions = 1;
            pipeline_config.use_intervention_region_search = true;
            pipeline_config.use_whirlpool_search = true;
            pipeline_config.scale.initial_scale = 0.02f;
            pipeline_config.scale.growth_factor = 2.0f;
            pipeline_config.scale.max_scale = 0.16f;
            pipeline_config.scale.max_geometric_trials = 4;
            pipeline_config.scale.max_refinement_trials = 0;
            pipeline_config.scale.min_cosine = 0.0f;
            pipeline_config.scale.max_leakage = 10.0f;
            pipeline_config.scale.max_shift_norm = 10.0f;
            pipeline_config.region_max_singleton_layers = std::min<size_t>(
                16, case_deltas.size());
            pipeline_config.region_max_neighborhoods = 4;
            pipeline_config.region_max_trials = 32;
            pipeline_config.region_max_stalled_scales = 2;
            pipeline_config.whirlpool_max_rounds = 2;
            pipeline_config.whirlpool_probes_per_round = 4;
            pipeline_config.whirlpool_max_trials = 8;
            pipeline_config.whirlpool_initial_radius = 2;
            pipeline_config.whirlpool_shrink_factor = 0.5f;
            evaluator_config.pipeline = pipeline_config;
            evaluator_config.max_references = 128;
            common_flydelta_evaluator_callbacks evaluator_callbacks;
            common_flydelta_search_pipeline_result pipeline_result;
            std::string baseline_tool;
            std::map<uint32_t, std::string> candidate_tools;
            common_flydelta_experiment_fixture fixture;
            fixture.id = "flydelta://fixture/search-pipeline/" + bootstrap_case.id;
            fixture.task_fingerprint = job.seed.task_fingerprint;
            fixture.model_profile_fingerprint = job.seed.model_profile_fingerprint;
            fixture.tokenizer_fingerprint = job.seed.tokenizer_fingerprint;
            fixture.template_fingerprint = job.seed.template_fingerprint;
            fixture.execution_context_fingerprint = job.seed.execution_context_fingerprint;
            fixture.verifier_revision = job.seed.verifier_ref;

            common_flydelta_search_pipeline_direction pipeline_direction;
            pipeline_direction.direction = bootstrap_direction;
            for (const auto & delta : case_deltas) {
                if (delta.layer_index <= 0) continue;
                const auto layer = static_cast<uint32_t>(delta.layer_index);
                pipeline_direction.available_layers.push_back(layer);
                pipeline_direction.layer_diagnostics.push_back({layer, 0.0f, 0.0f, 0.0f, 0.0f});
            }
            std::sort(pipeline_direction.available_layers.begin(),
                pipeline_direction.available_layers.end());
            pipeline_direction.available_layers.erase(std::unique(
                pipeline_direction.available_layers.begin(),
                pipeline_direction.available_layers.end()),
                pipeline_direction.available_layers.end());
            pipeline_direction.layer_anchors = {static_cast<uint32_t>(bootstrap_direction.layer_index)};
            if (pipeline_direction.layer_anchors.front() == 0 ||
                    !std::binary_search(pipeline_direction.available_layers.begin(),
                        pipeline_direction.available_layers.end(),
                        pipeline_direction.layer_anchors.front())) {
                pipeline_direction.layer_anchors = {pipeline_direction.available_layers.front()};
            }

            const auto normalized_delta = [&](const common_flydelta_behavior_delta & delta,
                    std::vector<float> & values, std::string & runner_error) {
                double squared = 0.0;
                for (const float value : delta.values) squared += static_cast<double>(value) * value;
                if (!std::isfinite(squared) || squared <= 0.0) {
                    runner_error = "dataset pipeline delta has zero norm";
                    return false;
                }
                const float inverse_norm = 1.0f / static_cast<float>(std::sqrt(squared));
                values.resize(delta.values.size());
                for (size_t index = 0; index < delta.values.size(); ++index) {
                    values[index] = delta.values[index] * inverse_norm;
                }
                return true;
            };

            evaluator_callbacks.run_search_pipeline = [&](const auto & queued_job,
                    auto & output, std::string & runner_error) {
                if (queued_job.behavior_delta_ids.size() != 1 ||
                        queued_job.behavior_delta_ids.front() != bootstrap_case.delta_id) {
                    runner_error = "search-pipeline worker received an unexpected behavior delta";
                    return false;
                }
                std::shared_ptr<const common_flydelta_hidden_state_capture> baseline_capture;
                std::optional<common_flydelta_decision_margin> baseline_choice_margin;
                const auto direction_for_layer = [&](uint32_t layer,
                        common_flydelta_basis_direction & direction, std::string & direction_error) {
                    const auto delta_it = std::find_if(case_deltas.begin(), case_deltas.end(),
                        [&](const auto & delta) { return delta.layer_index == static_cast<int32_t>(layer); });
                    if (delta_it == case_deltas.end()) {
                        direction_error = "search-pipeline arm has no matching layer delta";
                        return false;
                    }
                    direction.layer_index = static_cast<int32_t>(layer);
                    return normalized_delta(*delta_it, direction.values, direction_error);
                };

                common_flydelta_model_host model_host;
                model_host.capabilities.capture = true;
                model_host.capabilities.overlay = true;
                model_host.capabilities.generation = true;
                model_host.capabilities.teacher_forced_scoring = true;
                model_host.capabilities.host_verification = true;
                model_host.run_bounded_arm = [&](const common_flydelta_arm_request & arm_request,
                        common_flydelta_arm_result & arm_result, std::string & arm_error) {
                    std::shared_ptr<const common_flydelta_activation_result> activation_ptr;
                    if (arm_request.apply_overlay) {
                        if (arm_request.layer_indices.empty() ||
                                arm_request.layer_indices.size() != arm_request.coefficients.size()) {
                            arm_error = "dataset model host received an invalid overlay shape";
                            return false;
                        }
                        common_flydelta_activation_request activation_request;
                        activation_request.candidate_id =
                            "flydelta://candidate/search/" + bootstrap_case.id;
                        activation_request.artifact_id =
                            "flydelta://experimental/search/" + bootstrap_case.id;
                        activation_request.model_profile_fingerprint = capture->model_profile_fingerprint;
                        activation_request.capture_layout_revision = capture->capture_layout_revision;
                        activation_request.model_n_embd = n_embd;
                        activation_request.model_n_layers = n_layers;
                        activation_request.il_end = static_cast<int32_t>(n_layers - 1);
                        for (size_t index = 0; index < arm_request.layer_indices.size(); ++index) {
                            common_flydelta_basis_direction overlay_direction;
                            if (!direction_for_layer(
                                    arm_request.layer_indices[index], overlay_direction, arm_error)) {
                                return false;
                            }
                            activation_request.directions.push_back(std::move(overlay_direction));
                            activation_request.coefficients.push_back(arm_request.coefficients[index]);
                        }
                        activation_request.gate_request.explicit_opt_in = true;
                        activation_request.gate_request.candidate_status =
                            common_flydelta_candidate_status::approved;
                        activation_request.gate_request.basis_available = true;
                        activation_request.gate_request.familiarity = 1.0f;
                        activation_request.gate_request.novelty = 0.0f;
                        activation_request.gate_request.requested_scale = arm_request.alpha;
                        common_flydelta_gate_config gate_config;
                        gate_config.enabled = true;
                        gate_config.max_scale = 1.0f;
                        common_flydelta_activation_result activation;
                        if (!common_flydelta_prepare_activation(
                                gate_config, activation_request, 64U * 1024U * 1024U,
                                activation, arm_error)) return false;
                        activation_ptr = std::make_shared<const common_flydelta_activation_result>(
                            std::move(activation));
                    }

                    common_agent_generation_result generated_result;
                    if (!inference->generate(make_request(
                            value, contract, bootstrap_case.question, capture, activation_ptr),
                            generated_result)) {
                        arm_error = generated_result.error_message.empty()
                            ? "dataset model host generation failed" : generated_result.error_message;
                        return false;
                    }
                    const auto verdict = verify_model_call(
                        generated_result, bootstrap_case.expected_tool,
                        bootstrap_case.canonical_arguments, bootstrap_case.verification_mode,
                        &bootstrap_case.canonical_execution, host);
                    if (!arm_request.apply_overlay) {
                        baseline_tool = verdict.selected_tool;
                    } else if (!arm_request.layer_indices.empty()) {
                        candidate_tools[arm_request.layer_indices.front()] = verdict.selected_tool;
                    }
                    arm_result = {};
                    arm_result.arm_id = arm_request.arm_id;
                    arm_result.executed = true;
                    arm_result.requested_alpha = arm_request.alpha;
                    arm_result.executed_alpha = arm_request.alpha;
                    arm_result.generation_available = true;
                    arm_result.host_evaluated = true;
                    arm_result.verifier_known = verdict.value !=
                        host_verdict::kind::unresolved_alternative;
                    arm_result.host_outcome = verdict.value == host_verdict::kind::passed
                        ? common_flydelta_counterfactual_outcome::helped
                        : (arm_result.verifier_known
                            ? common_flydelta_counterfactual_outcome::neutral
                            : common_flydelta_counterfactual_outcome::unknown);
                    arm_result.quality = arm_result.host_outcome ==
                        common_flydelta_counterfactual_outcome::helped ? 1.0f : 0.0f;
                    arm_result.provenance_ref = arm_request.apply_overlay
                        ? "evidence:flydelta-search-overlay"
                        : "evidence:flydelta-search-baseline";
                    arm_result.execution_metrics.available = true;
                    if (generated_result.flydelta_capture &&
                            generated_result.flydelta_capture->captured) {
                        arm_result.execution_metrics.capture_bytes_to_host =
                            generated_result.flydelta_capture->values.size() * sizeof(float);
                    }
                    // This smoke currently calculates geometry through the
                    // CPU reference after capture. Keep that fact explicit so
                    // a future device reduction cannot be mistaken for the
                    // current full-capture path.
                    arm_result.execution_metrics.device_reduction_used = false;

                    arm_result.margin = {};
                    if (bootstrap_case.failed_tool.empty() &&
                            bootstrap_case.failed_continuation.empty()) {
                        arm_error = "fixture has no distinct failed tool for decision pair";
                        return false;
                    }
                    common_agent_teacher_forced_choice_request score_request;
                    score_request.context = make_request(
                        value, contract, bootstrap_case.question, capture, activation_ptr);
                    score_request.context.flydelta_capture.reset();
                    score_request.choice_prefix = "{\"name\":\"";
                    score_request.positive_choice = bootstrap_case.expected_tool;
                    score_request.negative_choice = bootstrap_case.failed_tool;
                    score_request.positive_continuation = tool_call_continuation(
                        bootstrap_case.expected_tool, bootstrap_case.canonical_arguments);
                    score_request.negative_continuation = bootstrap_case.failed_continuation;
                    common_agent_teacher_forced_choice_batch_request score_batch_request;
                    score_batch_request.choices.push_back(std::move(score_request));
                    common_agent_teacher_forced_choice_batch_result score_batch_result;
                    if (inference->score_teacher_forced_choice_batch(
                                score_batch_request, score_batch_result) &&
                            score_batch_result.choices.size() == 1 &&
                            score_batch_result.choices.front().available) {
                        const auto & score_result = score_batch_result.choices.front();
                        arm_result.margin.available = true;
                        arm_result.margin.positive_total_logprob =
                            score_result.positive_total_logprob;
                        arm_result.margin.negative_total_logprob =
                            score_result.negative_total_logprob;
                        arm_result.margin.positive_token_count = score_result.positive_token_count;
                        arm_result.margin.negative_token_count = score_result.negative_token_count;
                    }
                    arm_result.margin_available = arm_result.margin.available;
                    if (!arm_request.apply_overlay && generated_result.flydelta_capture &&
                            generated_result.flydelta_capture->captured) {
                        baseline_capture = generated_result.flydelta_capture;
                        if (arm_result.margin.available) baseline_choice_margin = arm_result.margin;
                    }
                    if (arm_request.apply_overlay && generated_result.flydelta_capture &&
                            generated_result.flydelta_capture->captured && baseline_capture) {
                        const uint32_t measured_after = *std::max_element(
                            arm_request.layer_indices.begin(), arm_request.layer_indices.end());
                        const auto delta_it = std::find_if(case_deltas.begin(), case_deltas.end(),
                            [&](const auto & delta) {
                                return delta.layer_index > static_cast<int32_t>(measured_after);
                            });
                        if (delta_it != case_deltas.end()) {
                            common_flydelta_representation_diagnostics diagnostics;
                            if (!common_flydelta_representation_diagnostics_from_captures(
                                    *baseline_capture, *generated_result.flydelta_capture, *delta_it,
                                    64U * 1024U * 1024U, diagnostics, arm_error)) return false;
                            arm_result.geometry_available = true;
                            arm_result.cosine = diagnostics.cosine;
                            arm_result.progress = diagnostics.progress;
                            arm_result.leakage = diagnostics.leakage;
                            arm_result.shift_norm = diagnostics.shift_norm;
                        }
                    }
                    std::cout << "flydelta_search_arm group=" << entry.first << " layers=";
                    if (arm_request.layer_indices.empty()) std::cout << "baseline";
                    else for (size_t index = 0; index < arm_request.layer_indices.size(); ++index) {
                        if (index != 0) std::cout << ',';
                        std::cout << arm_request.layer_indices[index];
                    }
                    std::cout << " scale=" << arm_request.alpha
                              << " selected_tool=" << verdict.selected_tool
                              << " outcome=" << common_flydelta_counterfactual_outcome_name(
                                  arm_result.host_outcome)
                              << " execution_path=" << common_flydelta_arm_execution_path_name(
                                  arm_result.execution_metrics.execution_path)
                              << " teacher_batch=single-entry"
                              << " diagnostics_bytes_to_host="
                              << arm_result.execution_metrics.diagnostics_bytes_to_host
                              << " margin_available=" << (arm_result.margin.available ? "yes" : "no")
                              << " margin_delta=" << (arm_result.margin.available
                                  ? arm_result.margin.total_delta() : 0.0f)
                              << '\n';
                    return true;
                };
                const auto model_runner = common_flydelta_search_pipeline_runner_from_model_host(
                    model_host, queued_job.id, "context://dataset-question-repair",
                    "intervention://dataset-question-repair/" + bootstrap_case.id);
                const auto model_batch_runner =
                    common_flydelta_search_pipeline_batch_runner_from_model_host(
                        model_host, queued_job.id, "context://dataset-question-repair",
                        "intervention://dataset-question-repair/" + bootstrap_case.id);
                const bool executed = common_flydelta_run_search_pipeline_batched(
                    fixture, pipeline_config, {pipeline_direction}, model_runner,
                    model_batch_runner, pipeline_result, runner_error);
                if (!executed) {
                    if (runner_error.empty()) {
                        runner_error = "FlyDelta model smoke search pipeline returned false";
                    }
                    error = runner_error;
                    return false;
                }
                output = pipeline_result;
                return true;
            };
            common_flydelta_experiment_worker_report worker_report;
            const bool enqueued = common_flydelta_experiment_queue_enqueue(worker_root, job, {}, error);
            const bool worker_executed = enqueued &&
                common_flydelta_experiment_worker_run_evaluator_once(
                    worker_root, {}, evaluator_config, evaluator_callbacks,
                    worker_report, error);
            if (!enqueued && error.empty()) {
                error = "search-pipeline worker could not enqueue job";
            } else if (!worker_executed && error.empty()) {
                error = "search-pipeline worker evaluator returned false";
            } else if (worker_report.state != common_flydelta_experiment_queue_state::succeeded &&
                    error.empty()) {
                error = "search-pipeline worker did not reach succeeded state";
            } else if (pipeline_result.directions.empty() && error.empty()) {
                error = "search-pipeline completed without direction results";
            }
            if (!enqueued || !worker_executed ||
                    worker_report.state != common_flydelta_experiment_queue_state::succeeded ||
                    pipeline_result.directions.empty()) {
                host.close();
                std::cerr << "Search-pipeline worker failed for " << entry.first << ": " << error << '\n';
                return 1;
            }
            std::cout << "flydelta_search_pipeline_worker group=" << entry.first
                      << " state=" << common_flydelta_experiment_queue_state_name(worker_report.state)
                      << " directions=" << pipeline_result.directions.size()
                      << " region_trials=" << pipeline_result.directions.front().region_trials.size()
                      << " selected=" << (pipeline_result.selection.selected ? "yes" : "no")
                      << " baseline_tool=" << baseline_tool
                      << '\n';
            if (value.adaptive_alpha_search) {
                const auto & pipeline_direction_result = pipeline_result.directions.front();
                const common_flydelta_intervention_region_trial * alpha_seed = nullptr;
                if (!pipeline_direction_result.region_trials.empty()) {
                    size_t seed_index = pipeline_direction_result.whirlpool_trace.best_trial_index;
                    if (seed_index >= pipeline_direction_result.region_trials.size()) {
                        seed_index = 0;
                        for (size_t index = 1;
                                index < pipeline_direction_result.region_trials.size(); ++index) {
                            if (pipeline_direction_result.region_trials[index].search_score >
                                    pipeline_direction_result.region_trials[seed_index].search_score) {
                                seed_index = index;
                            }
                        }
                    }
                    alpha_seed = &pipeline_direction_result.region_trials[seed_index];
                }
                if (alpha_seed == nullptr || alpha_seed->candidate.layer_indices.empty()) {
                    host.close();
                    std::cerr << "AdaptiveAlphaSearch has no Whirlpool region seed for "
                              << entry.first << '\n';
                    return 1;
                }

                std::shared_ptr<const common_flydelta_hidden_state_capture> alpha_baseline_capture;
                std::optional<common_flydelta_decision_margin> alpha_baseline_margin;
                const auto alpha_direction_for_layer = [&](uint32_t layer,
                        common_flydelta_basis_direction & direction,
                        std::string & alpha_error) {
                    const auto delta_it = std::find_if(case_deltas.begin(), case_deltas.end(),
                        [&](const auto & delta) { return delta.layer_index == static_cast<int32_t>(layer); });
                    if (delta_it == case_deltas.end()) {
                        alpha_error = "AdaptiveAlphaSearch arm has no matching layer delta";
                        return false;
                    }
                    direction.layer_index = static_cast<int32_t>(layer);
                    return normalized_delta(*delta_it, direction.values, alpha_error);
                };

                const auto alpha_runner = [&](const common_flydelta_experiment_fixture &,
                        float scale, bool apply_overlay,
                        common_flydelta_counterfactual_trial & trial,
                        common_flydelta_decision_margin & margin,
                        common_flydelta_representation_diagnostics & geometry,
                        bool & geometry_available, std::string & alpha_error) {
                    std::shared_ptr<const common_flydelta_activation_result> activation_ptr;
                    if (apply_overlay) {
                        common_flydelta_activation_request activation_request;
                        activation_request.candidate_id =
                            "flydelta://candidate/adaptive-alpha/" + bootstrap_case.id;
                        activation_request.artifact_id =
                            "flydelta://experimental/adaptive-alpha/" + bootstrap_case.id;
                        activation_request.model_profile_fingerprint = capture->model_profile_fingerprint;
                        activation_request.capture_layout_revision = capture->capture_layout_revision;
                        activation_request.model_n_embd = n_embd;
                        activation_request.model_n_layers = n_layers;
                        activation_request.il_end = static_cast<int32_t>(n_layers - 1);
                        for (const uint32_t layer : alpha_seed->candidate.layer_indices) {
                            common_flydelta_basis_direction direction;
                            if (!alpha_direction_for_layer(layer, direction, alpha_error)) return false;
                            activation_request.directions.push_back(std::move(direction));
                            activation_request.coefficients.push_back(1.0f);
                        }
                        activation_request.gate_request.explicit_opt_in = true;
                        activation_request.gate_request.candidate_status =
                            common_flydelta_candidate_status::approved;
                        activation_request.gate_request.basis_available = true;
                        activation_request.gate_request.familiarity = 1.0f;
                        activation_request.gate_request.novelty = 0.0f;
                        const float layer_count = static_cast<float>(
                            alpha_seed->candidate.layer_indices.size());
                        activation_request.gate_request.requested_scale =
                            scale / std::sqrt(std::max(1.0f, layer_count));
                        common_flydelta_gate_config gate_config;
                        gate_config.enabled = true;
                        gate_config.max_scale = 1.0f;
                        common_flydelta_activation_result activation;
                        if (!common_flydelta_prepare_activation(
                                gate_config, activation_request, 64U * 1024U * 1024U,
                                activation, alpha_error)) return false;
                        activation_ptr = std::make_shared<const common_flydelta_activation_result>(
                            std::move(activation));
                    }

                    common_agent_generation_result generated_result;
                    const bool generated = inference->generate(make_request(
                        value, contract, bootstrap_case.question, capture, activation_ptr),
                        generated_result);
                    const auto verdict = verify_model_call(generated_result,
                        bootstrap_case.expected_tool, bootstrap_case.canonical_arguments,
                        bootstrap_case.verification_mode, &bootstrap_case.canonical_execution, host);
                    trial = {};
                    trial.executed = true;
                    trial.verifier_known = generated &&
                        verdict.value != host_verdict::kind::unresolved_alternative;
                    trial.passed = generated && verdict.value == host_verdict::kind::passed;
                    trial.quality = trial.passed ? 1.0f : 0.0f;
                    trial.overlay_applied = apply_overlay;
                    trial.intervention_count = apply_overlay
                        ? alpha_seed->candidate.layer_indices.size() : 0;
                    trial.evidence_ref = apply_overlay
                        ? "evidence:flydelta-adaptive-alpha-overlay"
                        : "evidence:flydelta-adaptive-alpha-baseline";

                    margin = {};
                    std::string margin_error;
                    if (!bootstrap_case.failed_tool.empty() ||
                            !bootstrap_case.failed_continuation.empty()) {
                        common_agent_teacher_forced_choice_request score_request;
                        score_request.context = make_request(
                            value, contract, bootstrap_case.question, capture, activation_ptr);
                        score_request.context.flydelta_capture.reset();
                        score_request.choice_prefix = "{\"name\":\"";
                        score_request.positive_choice = bootstrap_case.expected_tool;
                        score_request.negative_choice = bootstrap_case.failed_tool;
                        score_request.positive_continuation = tool_call_continuation(
                            bootstrap_case.expected_tool, bootstrap_case.canonical_arguments);
                        score_request.negative_continuation = bootstrap_case.failed_continuation;
                        common_agent_teacher_forced_choice_result score_result;
                        if (inference->score_teacher_forced_choice(score_request, score_result) &&
                                score_result.available) {
                            margin.available = true;
                            margin.positive_total_logprob = score_result.positive_total_logprob;
                            margin.negative_total_logprob = score_result.negative_total_logprob;
                            margin.positive_token_count = score_result.positive_token_count;
                            margin.negative_token_count = score_result.negative_token_count;
                        } else {
                            margin_error = score_result.error_message.empty()
                                ? "teacher-forced choice margin unavailable"
                                : score_result.error_message;
                        }
                    } else {
                        margin_error = "fixture has no distinct failed tool for decision pair";
                    }
                    if (!apply_overlay && margin.available) alpha_baseline_margin = margin;
                    if (!apply_overlay && generated_result.flydelta_capture &&
                            generated_result.flydelta_capture->captured) {
                        alpha_baseline_capture = generated_result.flydelta_capture;
                    }

                    geometry = {};
                    geometry_available = false;
                    if (apply_overlay && generated_result.flydelta_capture &&
                            generated_result.flydelta_capture->captured && alpha_baseline_capture) {
                        const uint32_t measured_after = *std::max_element(
                            alpha_seed->candidate.layer_indices.begin(),
                            alpha_seed->candidate.layer_indices.end());
                        const auto delta_it = std::find_if(case_deltas.begin(), case_deltas.end(),
                            [&](const auto & delta) {
                                return delta.layer_index > static_cast<int32_t>(measured_after);
                            });
                        if (delta_it != case_deltas.end()) {
                            common_flydelta_representation_diagnostics diagnostics;
                            if (!common_flydelta_representation_diagnostics_from_captures(
                                    *alpha_baseline_capture, *generated_result.flydelta_capture,
                                    *delta_it, 64U * 1024U * 1024U, diagnostics, alpha_error)) {
                                return false;
                            }
                            geometry_available = true;
                            geometry.cosine = diagnostics.cosine;
                            geometry.progress = diagnostics.progress;
                            geometry.leakage = diagnostics.leakage;
                            geometry.shift_norm = diagnostics.shift_norm;
                        }
                    }
                    std::cout << "flydelta_adaptive_alpha_arm group=" << entry.first
                              << " layers=";
                    for (size_t index = 0; index < alpha_seed->candidate.layer_indices.size(); ++index) {
                        if (index != 0) std::cout << ',';
                        std::cout << alpha_seed->candidate.layer_indices[index];
                    }
                    std::cout << " scale=" << scale
                              << " outcome=" << (trial.passed ? "HELPED" :
                                  (trial.verifier_known ? "NEUTRAL" : "UNKNOWN"));
                    if (margin.available) {
                        const float delta_total = alpha_baseline_margin && apply_overlay
                            ? margin.total_delta() - alpha_baseline_margin->total_delta() : 0.0f;
                        const float delta_normalized = alpha_baseline_margin && apply_overlay
                            ? margin.normalized_delta() - alpha_baseline_margin->normalized_delta() : 0.0f;
                        std::cout << " margin_delta_total=" << delta_total
                                  << " margin_delta_normalized=" << delta_normalized;
                    } else {
                        std::cout << " margin_available=no";
                    }
                    if (geometry_available) {
                        std::cout << " cosine=" << geometry.cosine
                                  << " progress=" << geometry.progress
                                  << " leakage=" << geometry.leakage
                                  << " shift_norm=" << geometry.shift_norm;
                    }
                    if (!margin_error.empty()) std::cout << " margin_error=" << margin_error;
                    std::cout << '\n';
                    return true;
                };

                common_flydelta_alpha_response_search_config alpha_config;
                alpha_config.seed_scale = 0.02f;
                alpha_config.growth_factor = 2.0f;
                alpha_config.max_scale = 0.64f;
                // Keep the model-backed smoke bounded independently of the
                // production V0 budget (5 + 2 + 2). Each alpha arm performs
                // fresh model work, so this smoke only proves the wiring and
                // trace contract; the production search retains the larger
                // budget in its typed configuration.
                alpha_config.max_expansion_trials = 2;
                alpha_config.max_zoom_trials = 1;
                alpha_config.max_min_effective_trials = 1;
                alpha_config.max_expansion_non_improving = 2;
                alpha_config.max_leakage = pipeline_config.scale.max_leakage;
                alpha_config.max_shift_norm = pipeline_config.scale.max_shift_norm;
                alpha_config.dose_policy.max_shift_norm = pipeline_config.scale.max_shift_norm;
                alpha_config.dose_policy.max_leakage = pipeline_config.scale.max_leakage;
                std::vector<common_flydelta_alpha_response_trial> alpha_trials;
                common_flydelta_alpha_response_selection alpha_selection;
                std::string alpha_error;
                const bool alpha_ok = common_flydelta_run_alpha_response_search(
                    fixture, alpha_config, alpha_runner, alpha_trials, alpha_selection, alpha_error);
                if (!alpha_ok) {
                    host.close();
                    std::cerr << "AdaptiveAlphaSearch failed for " << entry.first
                              << ": " << alpha_error << '\n';
                    return 1;
                }
                    std::cout << "flydelta_adaptive_alpha_summary group=" << entry.first
                          << " seed_layer=" << alpha_seed->candidate.anchor_layer_index
                          << " trials=" << alpha_trials.size()
                          << " response_status=" << common_flydelta_alpha_response_status_name(
                              alpha_selection.response_status)
                          << " last_scale=" << alpha_selection.last_scale
                          << " max_reachable_scale=" << alpha_selection.max_reachable_scale
                          << " utility_slope=" << alpha_selection.utility_slope
                          << " range_not_exhausted=" <<
                              (alpha_selection.range_not_exhausted ? "yes" : "no")
                          << " selected=" << (alpha_selection.selected ? "yes" : "no")
                              << " selected_scale=" << alpha_selection.scale
                              << " last_requested_scale=" << alpha_selection.last_requested_scale
                              << " last_executed_scale=" << alpha_selection.last_executed_scale
                              << " last_relative_dose=" << alpha_selection.last_relative_dose
                              << " last_dose_action=" << common_flydelta_dose_action_name(
                                  alpha_selection.last_dose_action)
                              << " last_dose_evaluated=" <<
                                  (alpha_selection.last_dose_evaluated ? "yes" : "no")
                              << " last_dose_safety_limited=" <<
                                  (alpha_selection.last_dose_safety_limited ? "yes" : "no")
                              << " utility=" << alpha_selection.utility
                          << " minimum_effective=" <<
                              (alpha_selection.minimum_effective_available ? "yes" : "no")
                          << " minimum_effective_scale=" << alpha_selection.minimum_effective_scale
                              << '\n';
                for (const auto & alpha_trial : alpha_trials) {
                    std::cout << "flydelta_adaptive_alpha_dose group=" << entry.first
                              << " requested_scale=" << alpha_trial.requested_scale
                              << " executed_scale=" << alpha_trial.scale
                              << " relative_dose=" << alpha_trial.relative_dose
                              << " dose_action=" << common_flydelta_dose_action_name(
                                  alpha_trial.dose_action)
                              << " dose_evaluated=" <<
                                  (alpha_trial.dose_evaluated ? "yes" : "no")
                              << " safety_limited=" <<
                                  (alpha_trial.dose_safety_limited ? "yes" : "no")
                              << " reason=" << alpha_trial.dose_reason << '\n';
                }
                // Keep the printed trace representation in sync with the
                // optional bounded phase. The worker trace is persisted by
                // the worker before this smoke-only follow-up, so this is a
                // diagnostic projection rather than a lifecycle mutation.
                worker_report.trace.alpha_response_available = true;
                worker_report.trace.alpha_response_status = alpha_selection.response_status;
                worker_report.trace.alpha_range_not_exhausted =
                    alpha_selection.range_not_exhausted;
                worker_report.trace.alpha_last_scale = alpha_selection.last_scale;
                worker_report.trace.alpha_last_requested_scale =
                    alpha_selection.last_requested_scale;
                worker_report.trace.alpha_last_executed_scale =
                    alpha_selection.last_executed_scale;
                worker_report.trace.alpha_last_relative_dose =
                    alpha_selection.last_relative_dose;
                worker_report.trace.alpha_last_dose_action = common_flydelta_dose_action_name(
                    alpha_selection.last_dose_action);
                worker_report.trace.alpha_last_dose_evaluated =
                    alpha_selection.last_dose_evaluated;
                worker_report.trace.alpha_last_dose_safety_limited =
                    alpha_selection.last_dose_safety_limited;
                worker_report.trace.alpha_utility_slope = alpha_selection.utility_slope;
                worker_report.trace.alpha_best_margin_delta_normalized =
                    alpha_selection.best_margin_delta_normalized;
                // Emit the optional phase as an explicit trace update. The
                // worker trace is immutable after its bounded slice; this
                // keeps the later AdaptiveAlpha observation visible without
                // pretending it was part of the worker's persisted trace.
                std::cout << "flydelta_trace_update group=" << entry.first
                          << " phase=adaptive_alpha"
                          << " seed_layer=" << alpha_seed->candidate.anchor_layer_index
                          << " trials=" << alpha_trials.size()
                          << " response_status=" << common_flydelta_alpha_response_status_name(
                              alpha_selection.response_status)
                          << " selected=" << (alpha_selection.selected ? "yes" : "no")
                          << " range_not_exhausted=" <<
                              (alpha_selection.range_not_exhausted ? "yes" : "no")
                          << " last_requested_scale=" << alpha_selection.last_requested_scale
                          << " last_executed_scale=" << alpha_selection.last_executed_scale
                          << " utility=" << alpha_selection.utility
                          << '\n';
            }
            std::cout << "flydelta_trace group=" << entry.first
                      << " json=" << common_flydelta_trace_to_json(worker_report.trace) << '\n';
        }
    }
    std::string ledger_error;
    const auto persisted_learning = learning_ledger.list(ledger_error);
    if (!ledger_error.empty()) {
        host.close();
        std::cerr << "could not read persisted repair-echo failures: " << ledger_error << '\n';
        return 1;
    }
    size_t persisted_echo_failures = 0;
    for (const auto & persisted : persisted_learning) {
        for (const auto & signal : persisted.observation.signals) {
            if (signal.type == common_learning_signal_type::repair_echo_failure) {
                ++persisted_echo_failures;
                if (signal.repair_context_json.empty()) {
                    host.close();
                    std::cerr << "persisted repair-echo failure is missing repair context\n";
                    return 1;
                }
            }
        }
    }
    if (persisted_echo_failures != repair_echo_failures) {
        host.close();
        std::cerr << "persisted repair-echo failure count mismatch: expected "
                  << repair_echo_failures << " got " << persisted_echo_failures << '\n';
        return 1;
    }
    host.close();
    std::filesystem::remove_all(worker_root, ignored);
    std::cout << "flydelta_depth_summary layer2_samples=" << layer2_samples
              << " groups=" << samples_by_behavior.size()
              << " deep_groups=" << deep_groups
              << " direction_candidates=" << direction_candidates
              << " forced_deep_searches=" << deep_search_executed
              << " forced_deep_full_generation_trials=" << deep_full_generation_trials
              << " forced_tfo_lite=" << (value.force_tfo_lite ? "yes" : "no")
              << " bootstrap_overlay_evaluated=yes"
              << " deep_search_executed=" << (deep_search_executed != 0 ? "yes" : "no")
              << " repair_echo_persisted=" << persisted_echo_failures
              << " learning_ledger=" << learning_ledger_path << '\n';
    if (remove_learning_ledger) {
        std::error_code ledger_remove_error;
        std::filesystem::remove(learning_ledger_path, ledger_remove_error);
    }
    return 0;
}
