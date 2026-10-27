#include "../agent-flydelta-dataset-repair-host.h"

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
#include "agent/input-resources.h"
#include "agent/tooling/schema/tool-schema-compact.h"
#include "agent/tooling/schema/tool-output-codec.h"
#include "tools/agent/host/agent-host-config.h"
#include "tools/agent/runtime/agent-server-context-host.h"
#include "tools/server/server-context.h"
#include "tools/agent/daemon/agent-daemon-flydelta-internal.h"
#include "tools/agent/resource/agent-resource-store.h"
#include "common.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace {
using ordered_json = nlohmann::ordered_json;

struct options {
    std::string model;
    std::string suite;
    std::string config;
    std::string learning_ledger;
    bool force_deep = false;
    bool force_tfo_lite = false;
    bool low_rank_rank_sweep = false;
    bool dose_comparison = false;
    bool disable_factorized_cvec_batch = false;
    bool production_bindings = false;
    bool production_where = false;
    bool adaptive_alpha_search = false;
    int n_predict = 512;
    int n_threads = 3;
    int n_gpu_layers = 0;
    int n_parallel = 2;
    int n_ctx_per_sequence = 6144;
    std::string backend = "server";
    std::string outcome_guidance_mode;
    std::string scenario_id;
    std::string behavior_key;
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
    if (const char * rank_sweep = std::getenv("LLAMA_AGENT_LOW_RANK_RANK_SWEEP")) {
        value.low_rank_rank_sweep = std::string(rank_sweep) == "1" ||
            std::string(rank_sweep) == "true";
        value.force_deep = value.force_deep || value.low_rank_rank_sweep;
    }
    if (const char * adaptive_alpha = std::getenv("LLAMA_AGENT_ADAPTIVE_ALPHA_SEARCH")) {
        value.adaptive_alpha_search = std::string(adaptive_alpha) == "1" ||
            std::string(adaptive_alpha) == "true";
    }
    if (const char * threads = std::getenv("LLAMA_AGENT_THREADS")) value.n_threads = std::stoi(threads);
    if (const char * parallel = std::getenv("LLAMA_AGENT_N_PARALLEL")) value.n_parallel = std::stoi(parallel);
    if (const char * per_sequence_ctx = std::getenv("LLAMA_AGENT_N_CTX_PER_SEQUENCE")) {
        value.n_ctx_per_sequence = std::stoi(per_sequence_ctx);
    }
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
        else if (argument == "--low-rank-rank-sweep") {
            value.low_rank_rank_sweep = true;
            value.force_deep = true;
        }
        else if (argument == "--dose-comparison") {
            value.dose_comparison = true;
            value.force_deep = true;
        }
        else if (argument == "--disable-factorized-cvec-batch") value.disable_factorized_cvec_batch = true;
        else if (argument == "--adaptive-alpha-search") value.adaptive_alpha_search = true;
        else if (argument == "--production-bindings") value.production_bindings = true;
        else if (argument == "--production-where") { value.production_where = value.production_bindings = value.force_deep = true; }
        else if (argument == "--outcome-guidance-mode") {
            const auto v = next("--outcome-guidance-mode"); if (!v) return false;
            common_flydelta_outcome_guidance_mode parsed;
            std::string mode_error;
            if (!common_flydelta_parse_outcome_guidance_mode(v, parsed, mode_error)) {
                std::cerr << mode_error << '\n'; return false;
            }
            value.outcome_guidance_mode = v;
            value.production_bindings = value.production_where = value.force_deep = true;
        }
        else if (argument == "--n-predict") { const auto v = next("--n-predict"); if (!v) return false; value.n_predict = std::stoi(v); }
        else if (argument == "--threads") { const auto v = next("--threads"); if (!v) return false; value.n_threads = std::stoi(v); }
        else if (argument == "--n-parallel") { const auto v = next("--n-parallel"); if (!v) return false; value.n_parallel = std::stoi(v); }
        else if (argument == "--n-ctx-per-sequence") { const auto v = next("--n-ctx-per-sequence"); if (!v) return false; value.n_ctx_per_sequence = std::stoi(v); }
        else if (argument == "--n-gpu-layers") { const auto v = next("--n-gpu-layers"); if (!v) return false; value.n_gpu_layers = std::stoi(v); }
        else if (argument == "--backend") { const auto v = next("--backend"); if (!v) return false; value.backend = v; }
        else if (argument == "--scenario-id") { const auto v = next("--scenario-id"); if (!v) return false; value.scenario_id = v; }
        else if (argument == "--behavior-key") { const auto v = next("--behavior-key"); if (!v) return false; value.behavior_key = v; }
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
    ordered_json canonical_arguments = ordered_json::object();
    std::string verification_mode = "normalized_call";
    common_tool_execution_result canonical_execution;
    // The actual failed tool is the negative side of this fixture's
    // behavior-specific teacher-forced choice pair. It is deliberately not
    // a global dataset.inspect/statistics.describe assumption.
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

bool read_json(const std::string & path, ordered_json & value, std::string & error) {
    std::ifstream input(path, std::ios::binary);
    if (!input) { error = "could not open suite: " + path; return false; }
    std::ostringstream contents;
    contents << input.rdbuf();
    const std::string text = contents.str();
    value = ordered_json::parse(text, nullptr, false);
    if (!value.is_discarded() && value.is_object() && value.contains("scenarios") &&
            value["scenarios"].is_array()) {
        return true;
    }

    // The replay fixture is JSONL because it is also a durable, append-only
    // learning input. Normalize it to the model smoke's existing suite shape
    // so the host repair, capture and worker pipeline remain shared.
    ordered_json scenarios = ordered_json::array();
    std::istringstream lines(text);
    std::string line;
    size_t line_number = 0;
    while (std::getline(lines, line)) {
        ++line_number;
        if (line.empty()) continue;
        const auto entry = ordered_json::parse(line, nullptr, false);
        if (entry.is_discarded() || !entry.is_object() ||
                !entry.contains("id") || !entry.contains("question") ||
                !entry.contains("expected_tool") || !entry.contains("canonical_repair") ||
                !entry["canonical_repair"].is_object() ||
                !entry["canonical_repair"].contains("arguments")) {
            error = "dataset repair replay is invalid at line " + std::to_string(line_number);
            return false;
        }
        ordered_json scenario = entry;
        ordered_json steps = ordered_json::array();
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

std::string deep_arm_trace_id(const std::string & fixture_id, int32_t layer,
        const std::vector<float> & coefficients, bool full_execution) {
    return fixture_id + "/deep/" + (full_execution ? "full/" : "diagnostic/") +
        "layer-" + std::to_string(layer) + "/" + ordered_json(coefficients).dump();
}

void trace_model_exchange(const std::string & phase, const std::string & id,
        const common_agent_generation_request & request,
        const common_agent_generation_result & result) {
    ordered_json messages = ordered_json::array();
    for (const auto & message : request.messages) {
        messages.push_back({{"role", message.role}, {"content", message.content}});
    }
    const ordered_json exchange = {{"phase", phase}, {"id", id},
        {"output_source", phase == "repair" ? "host_repair_echo"
            : phase == "baseline" ? "original_model_generation"
            : phase.rfind("replay_forced_", 0) == 0 ? "synthetic_replay_capture"
            : "flydelta_arm_execution"},
        {"messages", messages}, {"n_predict", request.options.n_predict},
        {"budget", {{"phase", phase}, {"token_limit", request.options.n_predict},
            {"decoded_tokens", result.decoded_tokens},
            {"exhausted", request.options.n_predict > 0 && result.stop_reason == common_agent_generation_stop_reason::limit},
            {"diagnostic_only", request.options.n_predict == 0}}},
        {"content", result.content}, {"decoded_tokens", result.decoded_tokens},
        {"stop_reason", common_agent_generation_stop_reason_name(result.stop_reason)},
        {"overlay_requested", result.flydelta_runtime.requested},
        {"overlay_applied", result.flydelta_runtime.applied},
        {"factorized_binding", result.flydelta_runtime.factorized_binding},
        {"fallback_reason", result.flydelta_runtime.fallback_reason},
        {"error", result.error_message}};
    // Share stderr's FILE lock with resident diagnostics so a server log
    // cannot splice into the middle of a machine-readable event.
    std::fprintf(stderr, "flydelta_model_exchange %s\n", exchange.dump().c_str());
}

bool parse_model_tool_call(const common_agent_generation_result & result,
        const agent_flydelta_dataset_repair_host & host, ordered_json & parsed, std::string & error) {
    if (!common_agent_generation_succeeded(result)) {
        error = result.error_message;
        return false;
    }
    common_agent_tool_call call;
    if (!host.parse_model_call(result.content, call, error)) return false;
    const ordered_json arguments = ordered_json::parse(call.arguments_json, nullptr, false);
    if (!arguments.is_object()) {
        error = "parsed tool arguments are not an object";
        return false;
    }
    parsed = {{"name", call.name}, {"arguments", arguments}};
    return true;
}

std::string tool_call_continuation(const std::string & tool, const ordered_json & arguments,
        const agent_flydelta_dataset_repair_host & host) {
    std::string error;
    return host.render_model_call(tool, arguments, error);
}

std::string parsed_tool_call_continuation(const common_agent_generation_result & result,
        const agent_flydelta_dataset_repair_host & host) {
    ordered_json parsed;
    std::string error;
    if (!parse_model_tool_call(result, host, parsed, error)) return {};
    return tool_call_continuation(parsed["name"].get<std::string>(), parsed["arguments"], host);
}

bool result_oracle_matches(
        const common_tool_execution_result & actual,
        const common_tool_execution_result & expected) {
    if (!actual.ok || !expected.ok) return false;
    const ordered_json actual_json = ordered_json::parse(actual.output, nullptr, false);
    const ordered_json expected_json = ordered_json::parse(expected.output, nullptr, false);
    if (!actual_json.is_discarded() && !expected_json.is_discarded()) {
        return actual_json == expected_json;
    }
    return actual.output == expected.output;
}

host_verdict verify_model_call(const common_agent_generation_result & result,
        const std::string & expected_tool, const ordered_json & expected_arguments,
        const std::string & verification_mode,
        const common_tool_execution_result * expected_execution,
        agent_flydelta_dataset_repair_host & host) {
    host_verdict verdict;
    if (!common_agent_generation_succeeded(result)) {
        verdict.diagnostic = "model generation failed";
        return verdict;
    }
    if (result.stop_reason == common_agent_generation_stop_reason::limit) {
        verdict.value = host_verdict::kind::unresolved_alternative;
        verdict.diagnostic = "generation_token_budget_exhausted: decoded_tokens=" +
            std::to_string(result.decoded_tokens) + "; tool-call verifier not run";
        return verdict;
    }
    ordered_json parsed;
    std::string parse_error;
    if (!parse_model_tool_call(result, host, parsed, parse_error)) {
        verdict.diagnostic = "host rejected tool-call syntax: " + parse_error;
        return verdict;
    }
    verdict.selected_tool = parsed["name"].get<std::string>();
    ordered_json normalized_arguments;
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
    ordered_json normalized_expected;
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
    request.options.generation_trace = std::getenv("LLAMA_AGENT_RESIDENT_TRACE") != nullptr;
    request.messages = {{"system",
        "Choose the best tool for the request. Return exactly one compact DSL action.\n"
        "Put open! followed by the selected tool name on the first line.\n"
        "On subsequent lines, use only argument names declared by that tool, "
        "followed by a colon and the argument value, one argument per line.\n"
        "Follow the concrete tool examples below. Omit unset or host-resolved arguments.\n" +
        contract}, {"user", user}};
    request.flydelta_capture = capture;
    request.flydelta_activation = std::move(activation);
    return request;
}
}

int main(int argc, char ** argv) {
    const char * dose_environment = std::getenv("LLAMA_AGENT_DOSE_EXPERIMENT");
    const std::string dose_experiment = dose_environment ? dose_environment : "raw";
    if (dose_experiment != "raw" && dose_experiment != "normalized" && dose_experiment != "direction-dose") {
        std::cerr << "invalid LLAMA_AGENT_DOSE_EXPERIMENT\n"; return 2;
    }
    options value;
    if (!parse_args(argc, argv, value)) {
        std::cerr << "usage: " << argv[0]
                  << " --model MODEL --suite SUITE_JSON [--config AGENT_CONFIG]"
                  << " [--backend server]"
                  << " [--scenario-id ID]"
                  << " [--behavior-key KEY]"
                  << " [--max-scenarios N]"
                  << " [--learning-ledger JSONL]"
                  << " [--outcome-guidance-mode off|shadow|experimental]"
                  << " [--force-deep|--force-tfo-lite|--low-rank-rank-sweep|--dose-comparison] [--adaptive-alpha-search]"
                  << " [--threads N] [--n-gpu-layers N] [--n-parallel N]"
                  << " [--n-ctx-per-sequence N]\n";
        return 2;
    }
    if (value.dose_comparison && (!dose_environment || value.low_rank_rank_sweep || value.force_tfo_lite)) {
        std::cerr << "--dose-comparison requires one explicit dose mode and coordinate search\n";
        return 2;
    }
    if (value.model.empty() || !std::filesystem::is_regular_file(value.model)) {
        std::cerr << "dataset repair model smoke skipped: provide --model or LLAMA_AGENT_MODEL\n";
        return 77;
    }
    if (value.suite.empty() || !std::filesystem::is_regular_file(value.suite) ||
            value.n_threads <= 0 || value.n_threads > 3 || value.n_predict <= 0 ||
            value.n_parallel <= 0 || value.n_parallel > 16 || value.n_ctx_per_sequence <= 0 ||
            static_cast<int64_t>(value.n_parallel) * value.n_ctx_per_sequence >
                std::numeric_limits<int32_t>::max()) return 2;

    ordered_json suite;
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
    if (value.backend != "server") {
        std::cerr << "dataset repair model smoke requires the server backend\n";
        return 2;
    }

    agent_flydelta_dataset_repair_host host;
    if (!host.open("model", error)) { std::cerr << error << '\n'; return 1; }

    common_tool_catalog catalog;
    common_tool_bootstrap_result bootstrap;
    if (!catalog.bootstrap("analysis", bootstrap, error)) { host.close(); std::cerr << error << '\n'; return 1; }
    std::string contract = "Available read-only tools:\n";
    std::vector<std::string> described_tools;
    size_t selected_scenarios = 0;
    size_t considered_scenarios = 0;
    for (const auto & scenario : suite["scenarios"]) {
        const auto id = scenario.value("id", "unnamed");
        if (!value.scenario_id.empty() && id != value.scenario_id) continue;
        if (!value.behavior_key.empty() && scenario.value("behavior_key", "") != value.behavior_key) continue;
        if (value.max_scenarios != 0 && considered_scenarios >= value.max_scenarios) break;
        ++considered_scenarios;
        const auto expected = scenario.value("expected_tool", "");
        if (!policy_allows_tool_family(host_config ? &*host_config : nullptr, expected)) continue;
        ++selected_scenarios;
        const auto * definition = catalog.find_definition(expected);
        if (!definition) { host.close(); std::cerr << "unknown expected tool: " << expected << '\n'; return 1; }
        if (std::find(described_tools.begin(), described_tools.end(), expected) != described_tools.end()) continue;
        described_tools.push_back(expected);
        std::string compact_error;
        const auto description = common_render_compact_tool_description(definition->name, definition->description,
            common_tool_model_input_schema(*definition), common_tool_model_result_schema(*definition), compact_error);
        if (!compact_error.empty()) { host.close(); std::cerr << compact_error << '\n'; return 1; }
        contract += "- " + description + "\n";
    }
    if (selected_scenarios == 0) {
        host.close();
        std::cerr << "no scenarios enabled by agent learning policy\n";
        return 2;
    }
    if (!host.select_model_tools(described_tools, error)) {
        host.close(); std::cerr << error << '\n'; return 1;
    }
    contract += common_agent_render_dataset_inventory(host.available_datasets(), 8192, true);

    std::shared_ptr<common_agent_server_context_host> server_host;
    std::unique_ptr<common_agent_inference> inference;
    size_t n_embd = 0;
    size_t n_layers = 0;
    {
        server_host = std::make_shared<common_agent_server_context_host>();
        common_agent_server_context_host_config server_config;
        server_config.context_key.load_key.model = value.model;
        server_config.context_key.load_key.n_gpu_layers = value.n_gpu_layers;
        server_config.context_key.load_key.fit_params = true;
        // The model-backed repair smoke uses the production server-host
        // binding when requested. Keep two resident sequences so FlyDelta
        // waves can use the same per-sequence overlay/batch path as runtime.
        server_config.context_key.n_parallel = value.n_parallel;
        server_config.context_key.n_sequences = value.n_parallel;
        // Repair must see the complete model-facing tool contract and the
        // host repair context in one resident server request. With two
        // parallel slots, 8192 total tokens left only 4096 per request; the
        // replay suite's full compact tool contracts need more room.
        server_config.context_key.n_ctx = value.n_parallel * value.n_ctx_per_sequence;
        server_config.context_key.n_threads = value.n_threads;
        server_config.per_sequence_cvec_batch = true;
        server_config.per_sequence_factorized_cvec_batch =
            value.low_rank_rank_sweep && !value.disable_factorized_cvec_batch;
        std::cout << "flydelta_smoke_factorized_cvec_enabled="
                  << (server_config.per_sequence_factorized_cvec_batch ? "yes" : "no") << '\n';
        server_config.agent_trace = std::getenv("LLAMA_AGENT_RESIDENT_TRACE") != nullptr;
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
        std::cout << "flydelta_smoke_context requested_parallel=" << value.n_parallel
                  << " requested_ctx_per_sequence=" << value.n_ctx_per_sequence
                  << " requested_total_ctx=" << server_config.context_key.n_ctx
                  << " actual_total_ctx=" << llama_n_ctx(context)
                  << " actual_ctx_per_sequence=" << llama_n_ctx(context) / value.n_parallel
                  << " actual_parallel=" << value.n_parallel << '\n';
    }
    if (n_embd == 0 || n_layers <= 2) { host.close(); return 1; }
    // Reject an unstable forced-token prefix before buying any model probes.
    const auto * context = server_host->server().get_llama_context();
    const std::string forced_prefix = agent_flydelta_dataset_dsl_prefix;
    const auto prefix_tokens = common_tokenize(context, forced_prefix, false, true);
    for (const auto & name : described_tools) {
        const auto complete_tokens = common_tokenize(context, forced_prefix + " " + name, false, true);
        if (prefix_tokens.empty() || complete_tokens.size() < prefix_tokens.size() ||
                !std::equal(prefix_tokens.begin(), prefix_tokens.end(), complete_tokens.begin())) {
            host.close(); std::cerr << "DSL teacher prefix crosses a token boundary for " << name << '\n'; return 1;
        }
    }
    std::cout << "flydelta_teacher_prefix prefix=" << std::quoted(forced_prefix)
              << " stable=yes tokens=" << prefix_tokens.size() << std::endl;

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
    if (dose_environment) {
        const char * fingerprint = std::getenv("LLAMA_AGENT_DOSE_MODEL_FINGERPRINT");
        if (!fingerprint || !*fingerprint) {
            host.close(); std::cerr << "dose comparison requires a model content fingerprint\n"; return 2;
        }
        capture->model_profile_fingerprint = fingerprint;
    }
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
    std::map<std::string, common_flydelta_hidden_state_capture> dose_calibration_captures;
    size_t passed = 0, unresolved = 0, failures = 0, repaired = 0, synthetic_repairs = 0,
        repair_echo_failures = 0;
    const bool replay_mode = suite.value("replay_mode", false);
    size_t processed_scenarios = 0;
    for (const auto & scenario : suite["scenarios"]) {
        const auto id = scenario.value("id", "unnamed");
        if (!value.scenario_id.empty() && id != value.scenario_id) continue;
        if (!value.behavior_key.empty() && scenario.value("behavior_key", "") != value.behavior_key) continue;
        if (value.max_scenarios != 0 && processed_scenarios >= value.max_scenarios) break;
        ++processed_scenarios;
        const auto expected = scenario.value("expected_tool", "");
        const auto behavior_key = scenario.value(
            "behavior_key", "tool_use/dataset/structured_call_repair");
        if (!policy_allows_tool_family(host_config ? &*host_config : nullptr, expected)) continue;
        const auto steps = scenario.value("plan", ordered_json::object()).value("steps", ordered_json::array());
        if (steps.size() != 1) { host.close(); std::cerr << "scenario has invalid plan: " << id << '\n'; return 1; }
        common_tool_execution_result canonical_execution;
        if (!host.execute_step(steps.front(), canonical_execution, error)) {
            std::cout << "scenario=" << id << " host_verifier=unavailable error=" << error << '\n';
            continue;
        }
        const ordered_json canonical_repair = {{"name", expected}, {"arguments", steps.front()["args"]}};
        const std::string verification_mode = scenario.value("verification_mode", "normalized_call");
        if (verification_mode != "tool_only" && verification_mode != "normalized_call" &&
                verification_mode != "result_oracle") {
            host.close();
            std::cerr << "scenario has invalid verification_mode: " << id << '\n';
            return 1;
        }
        std::cout << "flydelta_fixture " << ordered_json({{"id", id}, {"expected", canonical_repair},
            {"canonical_output", canonical_execution.output}, {"verification_mode", verification_mode}}).dump()
                  << std::endl;

        common_agent_generation_result failed_result;
        const auto failed_request = make_request(value, contract, scenario.value("question", ""), capture);
        const bool generated = inference->generate(failed_request, failed_result);
        trace_model_exchange("baseline", id, failed_request, failed_result);
        const auto failed_verdict = verify_model_call(
            failed_result, expected, canonical_repair["arguments"], verification_mode,
            &canonical_execution, host);
        std::cout << "flydelta_host_verdict " << ordered_json({{"id", id}, {"phase", "baseline"},
            {"outcome_source", "original_model_generation"},
            {"selected_tool", failed_verdict.selected_tool}, {"diagnostic", failed_verdict.diagnostic},
            {"passed", failed_verdict.value == host_verdict::kind::passed}}).dump() << std::endl;
        std::string decision_failed_tool = failed_verdict.selected_tool;
        std::string decision_failed_continuation = parsed_tool_call_continuation(failed_result, host);
        std::string decision_contrast_source = "parsed_model_call";
        // Scoring an invalid DSL answer is still a valid teacher-forced
        // comparison. Preserve the observed continuation instead of replacing
        // the argument error with a fixture's unrelated invented tool.
        common_agent_teacher_forced_choice_request observed_contrast;
        std::string observed_contrast_error;
        if (agent_flydelta_dataset_dsl_contrast(observed_contrast,
                tool_call_continuation(expected, canonical_repair["arguments"], host),
                failed_result.content, observed_contrast_error)) {
            decision_failed_continuation = failed_result.content;
            decision_contrast_source = "observed_output";
            if (decision_failed_tool.empty()) {
                std::istringstream observed_header(failed_result.content);
                std::string marker;
                observed_header >> marker >> decision_failed_tool;
            }
        }
        // Replay fixtures deliberately carry a host-constructed negative call.
        // Use it only when Qwen's failed output cannot be parsed into a
        // model-facing choice pair; this keeps experimental margin scoring
        // available without turning the synthetic fixture into learning
        // credit.
        if (replay_mode && decision_failed_continuation.empty() &&
                scenario.contains("failed_output") && scenario["failed_output"].is_object()) {
            const auto & failed_fixture = scenario["failed_output"];
            const auto fixture_tool = failed_fixture.value("name", "");
            const ordered_json * fixture_arguments = nullptr;
            if (failed_fixture.contains("arguments")) fixture_arguments = &failed_fixture["arguments"];
            else if (failed_fixture.contains("args")) fixture_arguments = &failed_fixture["args"];
            if (!fixture_tool.empty() && fixture_arguments != nullptr && fixture_arguments->is_object()) {
                decision_failed_tool = fixture_tool;
                decision_failed_continuation = tool_call_continuation(fixture_tool, *fixture_arguments, host);
                decision_contrast_source = "synthetic_fixture_fallback";
            }
        }
        std::cout << "flydelta_decision_contrast fixture=" << id
                  << " negative_source=" << decision_contrast_source
                  << " positive_source=host_canonical learning_credit=none\n";
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
        const std::string canonical_dsl_call = tool_call_continuation(
            expected, canonical_repair["arguments"], host);
        common_agent_tool_call parsed_canonical_dsl_call;
        std::string canonical_dsl_error;
        const bool canonical_dsl_parsed = host.parse_model_call(
            canonical_dsl_call, parsed_canonical_dsl_call, canonical_dsl_error);
        ordered_json parsed_canonical_arguments = ordered_json::value_t::discarded;
        if (canonical_dsl_parsed) {
            parsed_canonical_arguments = ordered_json::parse(
                parsed_canonical_dsl_call.arguments_json, nullptr, false);
        }
        if (!canonical_dsl_parsed || parsed_canonical_dsl_call.name != expected ||
                parsed_canonical_arguments != canonical_repair["arguments"]) {
            host.close();
            std::cerr << "could not round-trip canonical DSL repair for " << id
                      << ": " << (canonical_dsl_error.empty()
                          ? "parsed tool or arguments differed"
                          : canonical_dsl_error) << '\n';
            return 1;
        }
        const std::string repair_request = "The host rejected the prior tool-call attempt (" +
            failed_verdict.diagnostic + "). The host constructed the canonical repair below. " +
            "Return this single compact DSL call unchanged. Do not return JSON, markdown, or explanation: " +
            canonical_dsl_call;
        common_agent_generation_result repaired_result;
        const auto repaired_request = make_request(value, contract, repair_request, capture);
        // Replay already owns an executed canonical repair. Capture that
        // continuation directly; a free echo adds cost and can corrupt the
        // target without providing any model-recovery evidence.
        const bool repair_generated = !replay_mode && inference->generate(repaired_request, repaired_result);
        host_verdict repaired_verdict;
        if (!replay_mode) {
            trace_model_exchange("repair", id, repaired_request, repaired_result);
            repaired_verdict = verify_model_call(repaired_result, expected,
                canonical_repair["arguments"], verification_mode, &canonical_execution, host);
            std::cout << "flydelta_host_verdict " << ordered_json({{"id", id}, {"phase", "repair"},
                {"outcome_source", "host_repair"},
                {"selected_tool", repaired_verdict.selected_tool}, {"diagnostic", repaired_verdict.diagnostic},
                {"passed", repaired_verdict.value == host_verdict::kind::passed}}).dump() << std::endl;
        } else {
            std::cout << "flydelta_host_repair " << ordered_json({{"id", id}, {"canonical_call", canonical_dsl_call},
                {"model_generation", false}, {"source", "host_canonical"}, {"learning_credit", false}}).dump() << std::endl;
        }
        const bool capture_pair = failed_result.flydelta_capture && repaired_result.flydelta_capture &&
            failed_result.flydelta_capture->captured && repaired_result.flydelta_capture->captured;
        // Replay captures an observed baseline and a host-canonical target.
        // These are experimental model representations, not observed recovery
        // or HELPED evidence.
        const bool synthetic_repair = replay_mode && generated && failed_result.flydelta_capture &&
            failed_result.flydelta_capture->captured;
        if ((!repair_generated || repaired_verdict.value != host_verdict::kind::passed || !capture_pair) &&
                !synthetic_repair) {
            const auto echo_evidence_id = "evidence:dataset-repair:" + id + ":repair-echo-failure";
            const ordered_json echo_context = {
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
            // evidence. The matched captures provide experimental WHAT.
            // Baseline and repair generation used different user prompts.
            // Their boundary delta would measure the added repair instruction,
            // rather than representations of bad/good continuations of one
            // task. Replay both continuations under the original prefix.
            const std::vector<common_agent_generation_request> capture_requests = {
                agent_flydelta_dataset_capture_context(failed_request, failed_result.content),
                agent_flydelta_dataset_capture_context(failed_request, canonical_dsl_call),
            };
            std::vector<common_agent_generation_result> capture_results;
            if (!inference->generate_batch(capture_requests, capture_results) || capture_results.size() != 2 ||
                    !capture_results[0].flydelta_capture || !capture_results[1].flydelta_capture ||
                    !capture_results[0].flydelta_capture->captured || !capture_results[1].flydelta_capture->captured) {
                host.close(); std::cerr << "matched replay capture failed for " << id << '\n'; return 1;
            }
            trace_model_exchange("replay_forced_baseline", id, capture_requests[0], capture_results[0]);
            trace_model_exchange("replay_forced_preferred", id, capture_requests[1], capture_results[1]);
            std::cout << "flydelta_replay_capture fixture=" << id
                      << " mode=host.forced_context.v1 common_prefix=original_task"
                      << " baseline=observed_output preferred=host_canonical learning_credit=none\n";
            const auto & baseline_capture = *capture_results[0].flydelta_capture;
            dose_calibration_captures[id] = baseline_capture;
            const auto & preferred_capture = *capture_results[1].flydelta_capture;
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
            manifest.captured_bytes = (baseline_capture.values.size() +
                preferred_capture.values.size()) * sizeof(float);
            if (!common_flydelta_capture_manifest_validate(
                    manifest, 64U * 1024U * 1024U, error)) {
                host.close();
                std::cerr << "synthetic replay manifest failed: " << error << '\n';
                return 1;
            }
            std::vector<common_flydelta_behavior_delta> replay_deltas;
            if (!common_flydelta_behavior_deltas_from_captures(
                    manifest, baseline_capture, preferred_capture,
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
                      << " selected=" << expected << " preferred_source=host_canonical"
                      << " failed_output=" << preview(failed_result)
                      << " preferred_output=" << ordered_json(canonical_dsl_call).dump() << '\n';
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
        dose_calibration_captures[id] = *failed_result.flydelta_capture;
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
    size_t deep_verified_overlay_trials = 0;
    size_t direction_candidates = 0;
    bool bootstrap_overlay_evaluated = false;
    const auto worker_root = std::filesystem::temp_directory_path() /
        "llama-agent-flydelta-dataset-question-bootstrap-worker";
    std::error_code ignored;
    std::filesystem::remove_all(worker_root, ignored);
    std::cout << "flydelta_dataset_question_repair_model_smoke"
              << " tool_output_format=dsl"
              << " baseline_passed=" << passed
              << " baseline_failures=" << failures
              << " unresolved=" << unresolved
              << " config=" << (value.config.empty() ? "none" : value.config)
              << " selected_scenarios=" << selected_scenarios
              << " fixture_counterfactual=" << (failures != 0 ? "yes" : "no")
              << " host_certified_repairs=" << repaired
              << " synthetic_repair_observations=" << synthetic_repairs
              << " repair_echo_failures=" << repair_echo_failures
              << " model_evidence=" <<
                    ((repaired != 0) ? "host_certified" :
                     (synthetic_repairs != 0 ? "experimental_only" : "inconclusive"))
              << " learning_credit=" << (repaired != 0 ? "eligible" : "none")
              << '\n';
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
        const size_t learning_eligible_samples = static_cast<size_t>(std::count_if(
            entry.second.begin(), entry.second.end(), [](const auto & sample) {
                return sample.credit.outcome == common_flydelta_counterfactual_outcome::helped &&
                    sample.credit.eligible_for_learning;
            }));
        std::cout << "flydelta_partition group=" << entry.first
                  << " observations=" << entry.second.size()
                  << " compatible=" << evidence_depth.compatible_samples
                  << " learning_eligible=" << learning_eligible_samples
                  << " experimental=" << evidence_depth.experimental_samples
                  << " rejected=" << evidence_depth.incompatible_samples
                  << " depth=" << common_flydelta_search_depth_name(evidence_depth.depth)
                  << " shallow_ready=" << (evidence_depth.shallow_ready ? "yes" : "no")
                  << " deep_ready=" << (evidence_depth.deep_ready ? "yes" : "no")
                  << " effective_rank=" << evidence_depth.effective_rank
                  << " stable_rank=" << evidence_depth.stable_rank
                  << " median_alignment=" << evidence_depth.median_alignment
                  << " condition_number=" << evidence_depth.condition_number
                  << " retained_explained_energy=" << evidence_depth.retained_explained_energy
                  << " smallest_retained_singular_value="
                  << evidence_depth.smallest_retained_singular_value
                  << " pairwise_cosines=";
        for (size_t index = 0; index < evidence_depth.pairwise_cosines.size(); ++index) {
            if (index != 0) std::cout << ',';
            std::cout << evidence_depth.pairwise_cosines[index];
        }
        std::cout
                  << " geometry_stable=" << (evidence_depth.geometry_stable ? "yes" : "no")
                  << " forced_deep_requested=" << (value.force_deep ? "yes" : "no")
                  << " forced_deep_override=" << (value.force_deep &&
                        evidence_depth.depth != common_flydelta_search_depth::deep ? "yes" : "no")
                  << " forced_deep_override_reason=" << (value.force_deep &&
                        evidence_depth.depth != common_flydelta_search_depth::deep
                            ? "smoke_only_evidence_depth_bypass" : "none")
                  << " region_budget=" << search_budget.max_region_trials
                  << " coefficient_budget=" << search_budget.max_coefficient_trials
                  << " tfo_lite=" << (search_budget.allow_tfo_lite ? "yes" : "no")
                  << " direction_candidates=" << directions.size() << '\n';
        for (const auto outcome : {common_flydelta_counterfactual_outcome::helped,
                common_flydelta_counterfactual_outcome::harmed,
                common_flydelta_counterfactual_outcome::neutral,
                common_flydelta_counterfactual_outcome::unknown}) {
            const size_t count = static_cast<size_t>(std::count_if(entry.second.begin(), entry.second.end(),
                [&](const auto & sample) { return sample.credit.outcome == outcome; }));
            if (count == 0) continue;
            const size_t eligible_count = static_cast<size_t>(std::count_if(
                entry.second.begin(), entry.second.end(), [&](const auto & sample) {
                    return sample.credit.outcome == outcome && sample.credit.eligible_for_learning;
                }));
            std::cout << "flydelta_learning_credit group=" << entry.first
                      << " source=" << (replay_mode ? "synthetic_replay" : "host_verified_repair")
                      << " outcome_source=" << (replay_mode ? "synthetic_replay" : "host_repair")
                      << " sample_outcome=" << common_flydelta_counterfactual_outcome_name(outcome)
                      << " samples=" << count
                      << " eligible=" << eligible_count
                      << " denied=" << (count - eligible_count)
                      << " decision_reason=" << (eligible_count > 0
                            ? "host_verified_helped_credit_explicit"
                            : replay_mode ? "synthetic_replay_is_diagnostic_only"
                            : outcome == common_flydelta_counterfactual_outcome::helped
                                ? "credit_flag_not_set"
                                : "outcome_is_not_helped")
                      << '\n';
        }
        for (const auto & sample : entry.second) {
            const bool eligible = sample.credit.outcome == common_flydelta_counterfactual_outcome::helped &&
                sample.credit.eligible_for_learning;
            const char * denial_reason = eligible ? "none"
                : replay_mode ? "synthetic_replay_is_diagnostic_only"
                : sample.credit.outcome != common_flydelta_counterfactual_outcome::helped
                    ? "outcome_is_not_helped"
                    : "learning_eligibility_flag_not_set";
            const ordered_json credit_event = {
                {"group", entry.first}, {"sample_id", sample.delta.id},
                {"fixture_id", sample.credit.fixture_id},
                {"source", replay_mode ? "synthetic_replay" : "host_verified_repair"},
                {"outcome_source", replay_mode ? "synthetic_replay" : "host_repair"},
                {"intervention_verified", false},
                {"outcome", common_flydelta_counterfactual_outcome_name(sample.credit.outcome)},
                {"quality_delta", sample.credit.quality_delta},
                {"eligible_for_learning", eligible}, {"denial_reason", denial_reason},
                {"layer", sample.delta.layer_index}, {"host_evidence_ref", sample.delta.host_evidence_ref},
            };
            std::fprintf(stderr, "flydelta_learning_credit_sample %s\n", credit_event.dump().c_str());
        }
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
        const bool guidance_smoke_requested = !value.outcome_guidance_mode.empty() &&
            value.production_where;
        const size_t required_directions = value.dose_comparison ? 1 : 2;
        if ((value.force_deep && entry.second.size() >= 2 && directions.size() >= required_directions) ||
                (guidance_smoke_requested && !entry.second.empty() && !directions.empty())) {
            const auto aggregation_snapshot_for_deep = aggregation.snapshot();
            const bool guidance_only_smoke = guidance_smoke_requested &&
                aggregation_snapshot_for_deep.retained_samples.size() < 2;
            if (aggregation_snapshot_for_deep.retained_samples.empty() ||
                    (aggregation_snapshot_for_deep.retained_samples.size() < 2 &&
                     !guidance_only_smoke)) {
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
            deep_config.max_rank = value.dose_comparison ? 1 : 2;
            deep_config.max_directions = 4;
            deep_config.full_generation_top_k = value.low_rank_rank_sweep ? 1 : 3;
            deep_config.coefficients.strategy = value.force_tfo_lite
                ? common_flydelta_coefficient_search_strategy::tfo_lite
                : common_flydelta_coefficient_search_strategy::coordinate;
            deep_config.coefficients.step = 0.05f;
            deep_config.coefficients.max_candidates = value.low_rank_rank_sweep ? 4 : 8;
            // Exercise the same bounded wave semantics used by a production
            // coefficient host. The smoke still permits scalar fallback, but
            // TFO must not bypass the batch runner or exceed this wave size.
            deep_config.coefficients.max_batch_arms = static_cast<size_t>(
                std::max(1, value.n_parallel - 1));
            deep_config.coefficients.max_l2_norm = 0.32f;
            deep_config.coefficients.norm_penalty = 0.05f;
            deep_config.coefficients.leakage_penalty = 0.10f;

            std::vector<common_flydelta_deep_search_direction> deep_directions;
            for (const auto & direction : directions) {
                deep_directions.push_back({direction, false, 0.0f});
            }
            std::shared_ptr<const common_flydelta_hidden_state_capture> deep_diagnostic_baseline_capture;
            std::shared_ptr<const common_flydelta_hidden_state_capture> deep_full_baseline_capture;
            std::optional<common_flydelta_decision_margin> deep_baseline_choice_margin;
            std::string deep_margin_error;
            std::shared_ptr<const std::vector<float>> deep_factorized_basis;
            agent_catalogued_resource_store production_resources(std::make_shared<agent_in_memory_blob_store>(),
                std::make_unique<agent_in_memory_resource_catalog>());
            std::shared_ptr<agent_daemon_flydelta_internal::daemon_flydelta_resource_provider> production_provider;
            common_flydelta_experiment_job production_job;
            if (value.production_bindings) {
                using namespace agent_daemon_flydelta_internal;
                daemon_options daemon_config;
                daemon_config.model = value.model;
                daemon_config.n_predict = value.n_predict;
                daemon_config.n_threads = value.n_threads;
                daemon_config.adaptation_flydelta_model_profile_fingerprint = capture->model_profile_fingerprint;
                daemon_config.adaptation_flydelta_capture_layout_revision = capture->capture_layout_revision;
                if (!value.outcome_guidance_mode.empty()) {
                    common_flydelta_parse_outcome_guidance_mode(value.outcome_guidance_mode,
                        daemon_config.adaptation_flydelta_outcome_guidance.mode, error);
                    daemon_config.adaptation_flydelta_outcome_guidance.matched_evaluation = false;
                }
                production_provider = daemon_flydelta_make_resource_provider(daemon_config, server_host,
                    &production_resources, {}, std::make_shared<common_learning_in_memory_lifecycle_store>(), error);
                if (!production_provider) { host.close(); std::cerr << error << '\n'; return 1; }
                if (!value.outcome_guidance_mode.empty()) {
                    std::cout << "flydelta_guidance_smoke requested_mode=" << value.outcome_guidance_mode
                              << " provider_mode=" << common_flydelta_outcome_guidance_mode_name(
                                    production_provider->outcome_guidance_config.mode)
                              << " matched_evaluation=off qualification=production_gate_preserved\n";
                }
                const auto put = [&](const ordered_json & data, std::string & uri) {
                    agent_resource_put_request request;
                    request.name = "flydelta-production-smoke.json";
                    request.mime_type = "application/json";
                    request.text = data.dump();
                    request.scope = common_runtime_resource_scope::session;
                    request.namespace_id = production_provider->authority.namespace_id;
                    request.session_id = production_provider->authority.session_id;
                    agent_resource_descriptor descriptor;
                    if (!production_resources.put_text(request, descriptor, error)) return false;
                    uri = descriptor.uri;
                    return true;
                };
                production_job.id = "flydelta-production-smoke:" + deep_case.id;
                production_job.seed.candidate_ref = "flydelta://experimental/repair-replay";
                production_job.seed.behavior_key = entry.first;
                production_job.seed.fixture_group_id = "dataset-question-guidance:" + deep_case.id;
                production_job.seed.verifier_contract_revision = "dataset-question-verifier:v1";
                production_job.seed.evidence_ref = deep_case_deltas.front().host_evidence_ref;
                production_job.capture_manifest_ids = {deep_case.capture_manifest_id};
                production_job.seed.model_profile_fingerprint = capture->model_profile_fingerprint;
                production_job.seed.task_fingerprint = deep_fixture.task_fingerprint;
                production_job.seed.tokenizer_fingerprint = deep_fixture.tokenizer_fingerprint;
                production_job.seed.template_fingerprint = deep_fixture.template_fingerprint;
                production_job.seed.execution_context_fingerprint = deep_fixture.execution_context_fingerprint;
                ordered_json tools = ordered_json::array();
                for (const auto & name : described_tools) {
                    const auto * definition = catalog.find_definition(name);
                    if (!definition) { host.close(); return 1; }
                    tools.push_back({{"name", name}, {"description", definition->description},
                        {"parameters", ordered_json::parse(common_tool_model_input_schema(*definition))}});
                }
                const auto context = make_request(value, contract, deep_case.question, {});
                ordered_json messages = ordered_json::array();
                for (const auto & message : context.messages) messages.push_back({{"role", message.role}, {"content", message.content}});
                ordered_json fixture_data = {{"verification_mode", "normalized_call"},
                    {"expected_decision", {{"name", deep_case.expected_tool}, {"arguments", deep_case.canonical_arguments}}}};
                common_agent_teacher_forced_choice_request contrast;
                std::string contrast_error;
                if (agent_flydelta_dataset_dsl_contrast(contrast,
                        tool_call_continuation(deep_case.expected_tool, deep_case.canonical_arguments, host),
                        deep_case.failed_continuation, contrast_error)) {
                    fixture_data["choice_prefix"] = contrast.choice_prefix;
                    fixture_data["positive_continuation"] = contrast.positive_continuation;
                    fixture_data["negative_continuation"] = contrast.negative_continuation;
                }
                if (!put({{"messages", messages}, {"model_tools", tools}, {"n_predict", value.n_predict}},
                        production_job.seed.baseline_ref) || !put(fixture_data,
                        production_job.seed.verifier_ref)) { host.close(); std::cerr << error << '\n'; return 1; }
                deep_fixture.id = production_job.seed.verifier_ref;
                std::cout << "flydelta_production_bindings group=" << entry.first
                          << " active=yes learning_credit=disabled source=experimental_replay\n";
                if (value.production_where) {
                    std::vector<common_flydelta_basis_direction> source;
                    for (const auto & delta : deep_case_deltas) source.push_back({delta.layer_index, delta.values});
                    if (!daemon_flydelta_register_composed_directions(production_provider,
                            production_job.seed.candidate_ref, std::move(source), "flydelta:experimental-where-replay",
                            production_job.seed.candidate_ref, error)) { host.close(); std::cerr << error << '\n'; return 1; }
                    common_flydelta_search_pipeline_config where_config;
                    where_config.dimension = n_embd;
                    where_config.max_directions = 1;
                    where_config.whirlpool_max_rounds = 1;
                    where_config.whirlpool_max_trials = where_config.region_max_trials = 4;
                    // Guidance-mode evaluation deliberately gives the adviser
                    // a bounded selection choice. The production default is
                    // untouched; all four layer candidates remain legal, but
                    // only two are executed in this smoke round.
                    where_config.whirlpool_probes_per_round = guidance_smoke_requested ? 2 : 4;
                    common_flydelta_search_pipeline_result where_result;
                    if (!daemon_flydelta_run_search_pipeline(production_provider, production_job,
                            where_config, where_result, error)) { host.close(); std::cerr << error << '\n'; return 1; }
                    if (guidance_smoke_requested) {
                        for (const auto & where_direction : where_result.directions) {
                            const auto & whirlpool = where_direction.whirlpool_trace;
                            ordered_json whirlpool_json = {
                                {"initial_centre_source", whirlpool.initial_centre_source},
                                {"initial_centre_decision_id",
                                    whirlpool.initial_centre_guidance_decision_id},
                                {"initial_centre_original_layers",
                                    whirlpool.initial_centre_guidance_original_layers},
                                {"initial_centre_posterior_layers",
                                    whirlpool.initial_centre_guidance_posterior_layers},
                                {"initial_centre_hypothetical_layers",
                                    whirlpool.initial_centre_guidance_hypothetical_layers},
                                {"rounds", ordered_json::array()},
                            };
                            for (const auto & round : whirlpool.rounds) {
                                whirlpool_json["rounds"].push_back({
                                    {"decision_id", round.guidance_decision_id},
                                    {"original_layers", round.guidance_original_layers},
                                    {"posterior_layers", round.guidance_posterior_layers},
                                    {"hypothetical_layers", round.guidance_hypothetical_layers},
                                    {"selected_layers", round.guidance_selected_layers},
                                    {"executed_layers", round.probed_layers},
                                    {"history_fingerprint", round.guidance_history_fingerprint},
                                    {"fallback_reason", round.guidance_fallback_reason},
                                });
                            }
                            std::cout << "flydelta_guidance_smoke_whirlpool_trace mode="
                                      << common_flydelta_outcome_guidance_mode_name(
                                          production_provider->outcome_guidance_config.mode)
                                      << " group=" << entry.first
                                      << " json=" << whirlpool_json.dump() << '\n';
                        }
                    }
                    if (guidance_only_smoke) {
                        std::cout << "flydelta_guidance_smoke_deep=unavailable"
                                  << " reason=insufficient_independent_repair_samples"
                                  << " whirlpool_probe_limit="
                                  << where_config.whirlpool_probes_per_round << '\n';
                        continue;
                    }
                    common_flydelta_search_continuation continuation;
                    if (!common_flydelta_select_search_continuation(where_result, continuation, error) ||
                            continuation.region.layer_indices.empty()) {
                        std::cout << "flydelta_low_rank_rank_sweep_skipped group=" << entry.first
                                  << " reason=no_safe_production_where_continuation\n";
                        continue;
                    }
                    const auto layer = continuation.region.anchor_layer_index;
                    const auto selected_delta = std::find_if(deep_case_deltas.begin(), deep_case_deltas.end(),
                        [&](const auto & delta) { return delta.layer_index == static_cast<int32_t>(layer); });
                    if (selected_delta == deep_case_deltas.end()) { host.close(); return 1; }
                    auto selected_direction = deep_directions.front();
                    selected_direction.direction.layer_index = layer;
                    selected_direction.direction.values = selected_delta->values;
                    selected_direction.direction.kind = common_flydelta_direction_kind::raw_repair;
                    selected_direction.direction.extraction_id = selected_delta->id;
                    selected_direction.direction.origin = "production_where_raw_repair";
                    selected_direction.direction.source_samples = 1;
                    selected_direction.direction.retained_samples = 1;
                    deep_directions = {std::move(selected_direction)};
                    std::cout << "flydelta_production_where group=" << entry.first << " layer=" << layer
                              << " host_helped=" << (continuation.host_helped ? "yes" : "no")
                              << " continuation=reference_only learning_credit=disabled\n";
                }
            }
            auto prepare_deep_arm = [&](const common_flydelta_low_rank_basis & basis,
                    const std::vector<float> & coefficients, bool apply_overlay,
                    bool allow_factorized_overlay,
                    common_agent_generation_request & request, std::string & runner_error) {
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
                    activation_request.coefficients = coefficients;
                    activation_request.allow_factorized_overlay =
                        value.low_rank_rank_sweep && !value.disable_factorized_cvec_batch &&
                        allow_factorized_overlay;
                    if (activation_request.allow_factorized_overlay) {
                        if (!deep_factorized_basis) {
                            auto shared = std::make_shared<std::vector<float>>();
                            shared->reserve(n_embd * basis.vectors.size());
                            for (const auto & vector : basis.vectors) shared->insert(shared->end(), vector.begin(), vector.end());
                            deep_factorized_basis = std::move(shared);
                        }
                        activation_request.factorized_basis = deep_factorized_basis;
                        activation_request.factorized_rank = static_cast<int32_t>(basis.vectors.size());
                        activation_request.factorized_layer_index = basis.layer_index;
                    } else {
                        for (const auto & vector : basis.vectors) {
                            activation_request.directions.push_back({basis.layer_index, vector});
                        }
                    }
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
                request = make_request(value, contract, deep_case.question, capture, activation_ptr);
                return true;
            };
            auto run_deep_arm = [&](const common_flydelta_low_rank_basis & basis,
                    const std::vector<float> & coefficients, bool apply_overlay, bool full_execution,
                    common_flydelta_counterfactual_trial & trial,
                    common_flydelta_decision_margin & margin,
                    common_flydelta_representation_diagnostics & geometry,
                    bool & geometry_available, std::string & runner_error) {
                if (production_provider) {
                    std::vector<common_flydelta_counterfactual_trial> trials;
                    std::vector<common_flydelta_decision_margin> margins;
                    std::vector<common_flydelta_representation_diagnostics> geometries;
                    std::vector<bool> flags;
                    if (!agent_daemon_flydelta_internal::daemon_flydelta_run_coefficient_arm_batch(
                            production_provider, production_job, deep_fixture, basis, {coefficients}, apply_overlay,
                            trials, margins, geometries, flags, runner_error, full_execution,
                            full_execution ? "smoke-full" : "smoke-diagnostic",
                            common_flydelta_experiment_phase::deep_controls)) return false;
                    trial = trials.front(); margin = margins.front(); geometry = geometries.front();
                    geometry_available = flags.front();
                    return true;
                }
                common_agent_generation_result generated_result;
                geometry = {};
                geometry_available = false;
                common_agent_generation_request generation_request;
                if (!prepare_deep_arm(basis, coefficients, apply_overlay, false,
                        generation_request, runner_error)) return false;
                if (!full_execution) generation_request.options.n_predict = 0;
                generated_result = {};
                const bool generated = inference->generate(generation_request, generated_result);
                trace_model_exchange(full_execution ? "deep_full" : "deep_diagnostic",
                    deep_arm_trace_id(deep_case.id, basis.layer_index, coefficients, full_execution),
                    generation_request, generated_result);
                if (generated && apply_overlay && !generated_result.flydelta_runtime.applied) {
                    runner_error = "Deep overlay was requested but runtime did not confirm graph application";
                    return false;
                }
                const auto verdict = full_execution ? verify_model_call(
                    generated_result, deep_case.expected_tool, deep_case.canonical_arguments,
                    deep_case.verification_mode, &deep_case.canonical_execution, host) : host_verdict{};
                trial = {};
                trial.executed = generated;
                trial.arm_id = deep_arm_trace_id(deep_case.id, basis.layer_index,
                    coefficients, full_execution);
                trial.host_evaluated = full_execution && generated;
                trial.verifier_known = full_execution && generated &&
                    verdict.value != host_verdict::kind::unresolved_alternative;
                trial.passed = full_execution && generated && verdict.value == host_verdict::kind::passed;
                trial.quality = trial.passed ? 1.0f : 0.0f;
                trial.overlay_applied = generated_result.flydelta_runtime.applied;
                trial.intervention_count = apply_overlay ? basis.vectors.size() : 0;
                trial.evidence_ref = apply_overlay
                    ? "evidence:flydelta-deep-overlay" : "evidence:flydelta-deep-baseline";
                trial.expected_decision_summary = deep_case.expected_tool + " " +
                    deep_case.canonical_arguments.dump();
                trial.observed_decision_summary = preview(generated_result);
                trial.verifier_reason = full_execution
                    ? (generated ? verdict.diagnostic : generated_result.error_message)
                    : "diagnostic_only_host_verifier_not_run";
                margin = {};
                deep_margin_error.clear();
                if (!deep_case.failed_tool.empty() || !deep_case.failed_continuation.empty()) {
                    common_agent_teacher_forced_choice_request score_request;
                    score_request.context = generation_request;
                    score_request.context.flydelta_capture.reset();
                    score_request.positive_choice = deep_case.expected_tool;
                    score_request.negative_choice = deep_case.failed_tool;
                    const bool contrast_valid = agent_flydelta_dataset_dsl_contrast(score_request,
                        tool_call_continuation(deep_case.expected_tool, deep_case.canonical_arguments, host),
                        deep_case.failed_continuation, deep_margin_error);
                    common_agent_teacher_forced_choice_result score_result;
                    if (contrast_valid && inference->score_teacher_forced_choice(score_request, score_result) &&
                            score_result.available) {
                        margin.available = true;
                        margin.positive_total_logprob = score_result.positive_total_logprob;
                        margin.negative_total_logprob = score_result.negative_total_logprob;
                        margin.positive_token_count = score_result.positive_token_count;
                        margin.negative_token_count = score_result.negative_token_count;
                    } else {
                        if (deep_margin_error.empty()) deep_margin_error = score_result.error_message.empty()
                            ? "teacher-forced choice margin unavailable"
                            : score_result.error_message;
                    }
                } else {
                    deep_margin_error = "fixture has no distinct failed tool for decision pair";
                }
                if (!apply_overlay && margin.available) deep_baseline_choice_margin = margin;
                auto & deep_baseline_capture = full_execution
                    ? deep_full_baseline_capture : deep_diagnostic_baseline_capture;
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
                          << " phase=" << (full_execution ? "full" : "diagnostic")
                          << " overlay=" << (apply_overlay ? "yes" : "no")
                          << " response=" << ordered_json({{"content", generated_result.content},
                              {"decoded_tokens", generated_result.decoded_tokens},
                              {"stop_reason", common_agent_generation_stop_reason_name(generated_result.stop_reason)},
                              {"verifier_known", trial.verifier_known},
                              {"verifier_diagnostic", full_execution ? verdict.diagnostic : "not_evaluated"}}).dump()
                          << '\n';
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
                        const bool executed = run_deep_arm(basis, coefficients, apply_overlay, false,
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
                        return run_deep_arm(basis, coefficients, apply_overlay, true, trial, margin,
                            geometry, geometry_available, runner_error);
                    };
            bool deep_device_batch_used = false;
            bool deep_backend_batch_used = false;
            const auto make_deep_batch_runner = [&](const bool full_execution) {
                return [&, full_execution](
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
                    if (coefficients.empty()) {
                        runner_error = "forced Deep batch has no coefficient arms";
                        return false;
                    }
                    if (production_provider) {
                        const size_t resident_parallelism = production_provider->host
                            ? production_provider->host->context_key().n_parallel : 0;
                        const size_t candidate_capacity = full_execution
                            ? std::max<size_t>(1, resident_parallelism)
                            : (resident_parallelism > 1 ? resident_parallelism - 1 : 1);
                        for (size_t begin = 0, part = 0; begin < coefficients.size(); ++part) {
                            const size_t count = std::min(candidate_capacity,
                                coefficients.size() - begin);
                            std::vector<std::vector<float>> chunk(
                                coefficients.begin() + static_cast<std::ptrdiff_t>(begin),
                                coefficients.begin() + static_cast<std::ptrdiff_t>(begin + count));
                            std::vector<common_flydelta_counterfactual_trial> part_trials;
                            std::vector<common_flydelta_decision_margin> part_margins;
                            std::vector<common_flydelta_representation_diagnostics> part_geometries;
                            std::vector<bool> part_geometry_available;
                            const std::string wave = std::string(full_execution
                                ? "smoke-full-part-" : "smoke-diagnostic-part-") +
                                std::to_string(part);
                            if (!agent_daemon_flydelta_internal::daemon_flydelta_run_coefficient_arm_batch(
                                    production_provider, production_job, deep_fixture, basis, chunk, true,
                                    part_trials, part_margins, part_geometries,
                                    part_geometry_available, runner_error, full_execution, wave,
                                    common_flydelta_experiment_phase::deep_controls,
                                    deep_config.coefficients.strategy)) return false;
                            trials.insert(trials.end(),
                                std::make_move_iterator(part_trials.begin()),
                                std::make_move_iterator(part_trials.end()));
                            margins.insert(margins.end(),
                                std::make_move_iterator(part_margins.begin()),
                                std::make_move_iterator(part_margins.end()));
                            geometries.insert(geometries.end(),
                                std::make_move_iterator(part_geometries.begin()),
                                std::make_move_iterator(part_geometries.end()));
                            geometry_available.insert(geometry_available.end(),
                                part_geometry_available.begin(), part_geometry_available.end());
                            begin += count;
                        }
                        if (trials.size() != coefficients.size() ||
                                margins.size() != coefficients.size() ||
                                geometries.size() != coefficients.size() ||
                                geometry_available.size() != coefficients.size()) {
                            runner_error = "forced Deep segmented batch returned incomplete results";
                            return false;
                        }
                        deep_device_batch_used = deep_device_batch_used || std::any_of(trials.begin(), trials.end(),
                            [](const auto & trial) { return trial.device_batch; });
                        deep_backend_batch_used = deep_backend_batch_used || std::any_of(trials.begin(), trials.end(),
                            [](const auto & trial) { return trial.backend_batch; });
                        return true;
                    }
                    std::vector<common_agent_generation_request> generation_requests;
                    generation_requests.reserve(coefficients.size());
                    for (const auto & values : coefficients) {
                        common_agent_generation_request request;
                        if (!prepare_deep_arm(basis, values, true,
                                coefficients.size() >= 2, request, runner_error)) return false;
                        if (!full_execution) request.options.n_predict = 0;
                        generation_requests.push_back(std::move(request));
                    }
                    std::vector<common_agent_generation_result> generated_results;
                    if (!inference->generate_batch(generation_requests, generated_results) ||
                            generated_results.size() != coefficients.size()) {
                        runner_error = "forced Deep native batch generation returned incomplete results";
                        return false;
                    }
                    for (const auto & result : generated_results) {
                        if (!result.flydelta_runtime.applied) {
                            runner_error = "Deep batch overlay was not confirmed by the decode graph";
                            return false;
                        }
                    }
                    bool wave_device = false;
                    bool wave_backend = false;
                    for (const auto & result : generated_results) {
                        // A physical wave can fragment after one sequence
                        // finishes. Record whether any fragment actually
                        // contained the multi-sequence batch; requiring every
                        // final result to report the original size would
                        // misclassify a valid singleton tail as scalar-only.
                        wave_device = wave_device || result.flydelta_device_batch;
                        wave_backend = wave_backend ||
                            result.flydelta_runtime.physical_batch_size >= 2;
                    }
                    deep_device_batch_used = deep_device_batch_used || wave_device;
                    deep_backend_batch_used = deep_backend_batch_used || wave_backend;
                    if (coefficients.size() > 1 && !wave_backend) {
                        runner_error = "forced Deep batch did not receive per-arm backend batch execution";
                        return false;
                    }
                    common_agent_teacher_forced_choice_batch_request score_batch;
                    score_batch.choices.reserve(coefficients.size());
                    for (size_t index = 0; index < coefficients.size(); ++index) {
                        common_agent_teacher_forced_choice_request score_request;
                        score_request.sequence_id = "deep-" + std::to_string(index);
                        score_request.context = generation_requests[index];
                        score_request.context.flydelta_capture.reset();
                        score_request.positive_choice = deep_case.expected_tool;
                        score_request.negative_choice = deep_case.failed_tool;
                        if (!agent_flydelta_dataset_dsl_contrast(score_request,
                                tool_call_continuation(deep_case.expected_tool, deep_case.canonical_arguments, host),
                                deep_case.failed_continuation, runner_error)) return false;
                        score_batch.choices.push_back(std::move(score_request));
                    }
                    common_agent_teacher_forced_choice_batch_result score_results;
                    if (!inference->score_teacher_forced_choice_batch(score_batch, score_results) ||
                            score_results.choices.size() != coefficients.size()) {
                        runner_error = score_results.error_message.empty()
                            ? "forced Deep batch teacher scoring returned incomplete results"
                            : score_results.error_message;
                        return false;
                    }
                    trials.reserve(coefficients.size());
                    margins.reserve(coefficients.size());
                    geometries.reserve(coefficients.size());
                    geometry_available.reserve(coefficients.size());
                    for (size_t index = 0; index < coefficients.size(); ++index) {
                        const auto & generated_result = generated_results[index];
                        const std::string arm_trace_id = deep_arm_trace_id(deep_case.id,
                            basis.layer_index, coefficients[index], full_execution);
                        trace_model_exchange(full_execution ? "deep_full" : "deep_diagnostic",
                            arm_trace_id,
                            generation_requests[index], generated_result);
                        const auto verdict = full_execution ? verify_model_call(generated_result,
                            deep_case.expected_tool, deep_case.canonical_arguments,
                            deep_case.verification_mode, &deep_case.canonical_execution, host) : host_verdict{};
                        common_flydelta_counterfactual_trial trial;
                        trial.executed = common_agent_generation_succeeded(generated_result);
                        trial.arm_id = arm_trace_id;
                        trial.host_evaluated = full_execution && trial.executed;
                        trial.verifier_known = full_execution && trial.executed &&
                            verdict.value != host_verdict::kind::unresolved_alternative;
                        trial.passed = full_execution && trial.executed && verdict.value == host_verdict::kind::passed;
                        trial.quality = trial.passed ? 1.0f : 0.0f;
                        trial.overlay_applied = generated_result.flydelta_runtime.applied;
                        trial.intervention_count = basis.vectors.size();
                        trial.evidence_ref = "evidence:flydelta-deep-overlay";
                        trial.expected_decision_summary = deep_case.expected_tool + " " +
                            deep_case.canonical_arguments.dump();
                        trial.observed_decision_summary = preview(generated_result);
                        trial.verifier_reason = full_execution
                            ? (trial.executed ? verdict.diagnostic : generated_result.error_message)
                            : "diagnostic_only_host_verifier_not_run";
                        common_flydelta_decision_margin margin;
                        const auto & scored = score_results.choices[index];
                        margin.available = scored.available;
                        margin.positive_total_logprob = scored.positive_total_logprob;
                        margin.negative_total_logprob = scored.negative_total_logprob;
                        margin.positive_token_count = scored.positive_token_count;
                        margin.negative_token_count = scored.negative_token_count;
                        common_flydelta_representation_diagnostics geometry;
                        bool has_geometry = false;
                        const auto & deep_baseline_capture = full_execution
                            ? deep_full_baseline_capture : deep_diagnostic_baseline_capture;
                        if (generated_result.flydelta_capture && generated_result.flydelta_capture->captured &&
                                deep_baseline_capture) {
                            const auto measurement_it = std::find_if(deep_case_deltas.begin(),
                                deep_case_deltas.end(), [&](const auto & delta) {
                                    return delta.layer_index > basis.layer_index;
                                });
                            if (measurement_it == deep_case_deltas.end() ||
                                    !common_flydelta_representation_diagnostics_from_captures(
                                        *deep_baseline_capture, *generated_result.flydelta_capture,
                                        *measurement_it, 64U * 1024U * 1024U,
                                        geometry, runner_error)) {
                                if (runner_error.empty()) runner_error = "forced Deep batch geometry unavailable";
                                return false;
                            }
                            has_geometry = true;
                        }
                        std::cout << "flydelta_deep_batch_response group=" << entry.first
                                  << " phase=" << (full_execution ? "full" : "diagnostic")
                                  << " arm=" << index << " response="
                                  << ordered_json({{"content", generated_result.content},
                                      {"decoded_tokens", generated_result.decoded_tokens},
                                      {"stop_reason", common_agent_generation_stop_reason_name(generated_result.stop_reason)},
                                      {"verifier_known", trial.verifier_known},
                                      {"verifier_diagnostic", full_execution ? verdict.diagnostic : "not_evaluated"}}).dump()
                                  << '\n';
                        trials.push_back(std::move(trial));
                        margins.push_back(std::move(margin));
                        geometries.push_back(std::move(geometry));
                        geometry_available.push_back(has_geometry);
                    }
                    return true;
                };
            };
            const auto deep_diagnostic_batch_runner = make_deep_batch_runner(false);
            const auto deep_full_batch_runner = make_deep_batch_runner(true);
            // A rank sweep reuses the ordinary bounded Deep/TFO seam.  It is
            // deliberately report-only: the host still owns outcomes and a
            // higher rank is never treated as an improvement merely because
            // its diagnostic score changed.
            std::vector<size_t> requested_ranks;
            if (value.low_rank_rank_sweep) {
                // Experimental rank is geometry, not learning authority.
                // Use distinct observed capture pairs at the selected layer;
                // derived trimmed/whitened variants cannot manufacture rank.
                std::vector<common_flydelta_deep_search_direction> observed_directions;
                std::vector<common_flydelta_direction_candidate> observed_candidates;
                std::set<std::string> capture_pairs;
                for (const auto & sample : aggregation_snapshot_for_deep.retained_samples) {
                    if (!capture_pairs.insert(sample.delta.capture_manifest_id).second) continue;
                    const auto source_case = bootstrap_cases_by_delta.find(sample.delta.id);
                    if (source_case == bootstrap_cases_by_delta.end()) continue;
                    const auto source_deltas = deltas_by_case.find(source_case->second.id);
                    if (source_deltas == deltas_by_case.end()) continue;
                    const auto source_delta = std::find_if(source_deltas->second.begin(), source_deltas->second.end(),
                        [&](const auto & delta) { return delta.layer_index == deep_directions.front().direction.layer_index; });
                    if (source_delta == source_deltas->second.end()) continue;
                    auto observed = deep_directions.front().direction;
                    observed.kind = common_flydelta_direction_kind::raw_repair;
                    observed.values = source_delta->values;
                    observed.source_samples = observed.retained_samples = 1;
                    observed.median_alignment = 1.0f;
                    observed.experimental_only = true;
                    observed.extraction_id = source_delta->id;
                    observed.origin = "observed_capture_pair_rank_probe";
                    observed_candidates.push_back(observed);
                    observed_directions.push_back({std::move(observed), false, 0.0f});
                    if (observed_directions.size() == deep_config.max_directions) break;
                }
                common_flydelta_low_rank_basis observed_basis;
                if (!common_flydelta_build_low_rank_basis(n_embd, 4, observed_candidates, observed_basis, error)) {
                    host.close(); std::cerr << "experimental capture-pair rank failed: " << error << '\n'; return 1;
                }
                const size_t observed_rank = observed_basis.vectors.size();
                deep_directions.clear();
                for (size_t component = 0; component < observed_rank; ++component) {
                    auto observed = observed_directions.front().direction;
                    observed.values = observed_basis.vectors[component];
                    observed.source_samples = observed.retained_samples = observed_candidates.size();
                    observed.origin = "observed_capture_pair_subspace";
                    observed.extraction_id = "experimental-subspace:" + deep_case.id + ":" + std::to_string(component);
                    deep_directions.push_back({std::move(observed), false, 0.0f});
                }
                std::cout << "flydelta_experimental_rank group=" << entry.first
                          << " capture_pairs=" << observed_candidates.size()
                          << " effective_rank=" << observed_rank
                          << " learning_effective_rank=" << evidence_depth.effective_rank
                          << " learning_credit=none\n";
                requested_ranks.push_back(1);
                for (const size_t requested_rank : {size_t{2}, size_t{4}}) {
                    if (observed_rank >= requested_rank &&
                            deep_directions.size() >= requested_rank) {
                        requested_ranks.push_back(requested_rank);
                    } else {
                        std::cout << "flydelta_low_rank_rank_unavailable group=" << entry.first
                                  << " requested_rank=" << requested_rank
                                  << " evidence_effective_rank=" << evidence_depth.effective_rank
                                  << " observed_capture_rank=" << observed_rank
                                  << " direction_candidates=" << deep_directions.size()
                                  << " reason=insufficient_independent_evidence\n";
                    }
                }
            } else {
                requested_ranks.push_back(deep_config.max_rank);
            }
            for (const size_t requested_rank : requested_ranks) {
                deep_factorized_basis.reset();
                deep_device_batch_used = false;
                deep_backend_batch_used = false;
                common_flydelta_deep_search_config rank_config = deep_config;
                rank_config.max_rank = requested_rank;
                if (dose_environment) {
                    if ((!value.low_rank_rank_sweep && !value.dose_comparison) || value.force_tfo_lite) {
                        host.close(); std::cerr << "dose comparison requires coordinate search mode\n"; return 2;
                    }
                    auto & coefficients = rank_config.coefficients;
                    coefficients.record_intervention_norms = true;
                    coefficients.max_candidates = 16;
                    // No retries: baseline + 15 proposals is the same inference
                    // budget in all three conditions; unavailable arms are explicit.
                    coefficients.max_dose_retries = 0;
                    if (dose_experiment != "raw") {
                        std::vector<common_flydelta_hidden_state_capture> calibration_captures;
                        std::vector<std::string> calibration_ids;
                        for (const auto & item : dose_calibration_captures) {
                            if (item.first == deep_case.id) continue;
                            calibration_captures.push_back(item.second); calibration_ids.push_back(item.first);
                        }
                        if (!common_flydelta_calibrate_residual_dose(calibration_captures,
                                deep_directions.front().direction.layer_index,
                                "dose-calibration:" + deep_case.id,
                                "generation_boundary", coefficients.residual_calibration, error)) {
                            host.close(); std::cerr << "dose calibration unavailable: " << error << '\n'; return 1;
                        }
                        if (dose_experiment != "normalized") {
                            std::vector<std::vector<common_flydelta_hidden_state_capture>> sequences;
                            for (const auto & captured : calibration_captures) sequences.push_back({captured});
                            if (!common_flydelta_calibrate_residual_sequences(sequences, calibration_ids,
                                    deep_directions.front().direction.layer_index, "dose-calibration:" + deep_case.id,
                                    "generation_boundary", coefficients.residual_calibration, error)) {
                                host.close(); std::cerr << "sequence calibration unavailable: " << error << '\n'; return 1;
                            }
                            auto & calibration = coefficients.residual_calibration;
                            const auto runtime_json = ordered_json{{"n_ctx", server_host->params().n_ctx},
                                {"n_parallel", server_host->params().n_parallel}, {"n_gpu_layers", server_host->params().n_gpu_layers},
                                {"cache_type_k", static_cast<int>(server_host->params().cache_type_k)},
                                {"cache_type_v", static_cast<int>(server_host->params().cache_type_v)}}.dump();
                            calibration.runtime_configuration_fingerprint = "sha256:" + hash_sha256_hex(runtime_json.data(), runtime_json.size());
                            calibration.intervention_site = "layer-input:additive-cvec:v1";
                            calibration.runtime_layout_revision = "dataset-question-smoke-cvec:v1";
                            const auto dataset_json = ordered_json{{"cases", calibration_ids},
                                {"measurements", calibration.sample_norms}, {"fixture_revision", "dataset-question:v1"}}.dump();
                            calibration.dataset_revision = "sha256:" + hash_sha256_hex(dataset_json.data(), dataset_json.size());
                            coefficients.injection_intervention_site = calibration.intervention_site;
                            coefficients.injection_runtime_layout_revision = calibration.runtime_layout_revision;
                            coefficients.injection_runtime_configuration_fingerprint = calibration.runtime_configuration_fingerprint;
                        }
                        coefficients.residual_calibration.tokenizer_fingerprint = deep_fixture.tokenizer_fingerprint;
                        coefficients.residual_calibration.template_fingerprint = deep_fixture.template_fingerprint;
                        coefficients.injection_capture_layout_revision = capture->capture_layout_revision;
                        coefficients.injection_token_selection = "generation_boundary";
                        coefficients.experimental_dose_mode = dose_experiment == "normalized"
                            ? common_flydelta_coefficient_dose_mode::residual_fraction
                            : common_flydelta_coefficient_dose_mode::direction_dose;
                        const auto & c = coefficients.residual_calibration;
                        std::cout << "flydelta_dose_calibration " << ordered_json({
                            {"schema_version", c.schema_version}, {"reference", c.reference},
                            {"model_profile_fingerprint", c.model_profile_fingerprint},
                            {"tokenizer_fingerprint", c.tokenizer_fingerprint},
                            {"template_fingerprint", c.template_fingerprint},
                            {"capture_layout_revision", c.capture_layout_revision},
                            {"intervention_site", c.intervention_site}, {"measurement_stage", c.measurement_stage},
                            {"runtime_configuration_fingerprint", c.runtime_configuration_fingerprint},
                            {"runtime_layout_revision", c.runtime_layout_revision}, {"dataset_revision", c.dataset_revision},
                            {"aggregation", c.aggregation}, {"sequence_ids", c.sequence_ids},
                            {"token_selection", c.token_selection}, {"layer", c.layer_index},
                            {"median_norm", c.median_norm}, {"sample_count", c.sample_count},
                            {"dimension", c.dimension}, {"sample_norms", c.sample_norms},
                            {"capture_position", common_flydelta_capture_position_name(c.position)},
                            {"token_index", c.token_index},
                            {"source_scenarios", calibration_ids}, {"excluded_evaluation_scenario", deep_case.id},
                            {"learning_credit_granted", false}}).dump() << std::endl;
                    }
                }
                common_flydelta_deep_search_result deep_result;
                const auto deep_started = std::chrono::steady_clock::now();
                if (!common_flydelta_run_deep_search_batched(
                        deep_fixture, rank_config, deep_directions,
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
                        if (trial.executed && trial.host_evaluated && trial.overlay_applied) {
                            ++deep_verified_overlay_trials;
                        }
                    }
                }
                const size_t deep_group_diagnostic_trials =
                    deep_result.coefficient_trials.size() - deep_group_full_generation_trials;
                deep_diagnostic_trials += deep_group_diagnostic_trials;
                const size_t arm_count = deep_result.coefficient_trials.size();
                const size_t effective_rank = deep_result.basis.vectors.size();
                const size_t flattened_bytes = arm_count * n_embd * sizeof(float);
                const size_t shared_basis_bytes =
                    (n_embd * effective_rank + arm_count * effective_rank) * sizeof(float);
                std::cout << "flydelta_low_rank_rank_sweep group=" << entry.first
                          << " requested_rank=" << requested_rank
                          << " effective_rank=" << effective_rank
                          << " evidence_effective_rank=" << evidence_depth.effective_rank
                          << " retained_explained_energy=" << evidence_depth.retained_explained_energy
                          << " smallest_retained_singular_value="
                          << evidence_depth.smallest_retained_singular_value
                          << " condition_number=" << evidence_depth.condition_number
                          << " strategy="
                          << common_flydelta_coefficient_search_strategy_name(
                              rank_config.coefficients.strategy)
                          << " diagnostic_trials=" << deep_group_diagnostic_trials
                          << " full_generation_trials=" << deep_group_full_generation_trials
                          << " max_batch_arms=" << rank_config.coefficients.max_batch_arms
                          << " execution_batch=" << (deep_device_batch_used
                                ? "device_batch" : deep_backend_batch_used
                                ? "backend_batch" : "none")
                          << " flattened_overlay_bytes=" << flattened_bytes
                          << " shared_basis_model_bytes=" << shared_basis_bytes
                          << " selected=" << (deep_result.coefficient_selection.selected ? "yes" : "no")
                          << " elapsed_ms=" << std::chrono::duration_cast<std::chrono::milliseconds>(
                              std::chrono::steady_clock::now() - deep_started).count() << '\n';
                for (size_t trial_index = 0;
                        trial_index < deep_result.coefficient_trials.size(); ++trial_index) {
                    const auto & trial = deep_result.coefficient_trials[trial_index];
                    const bool full_generation = trial.mutation_kind == "full_generation_top_arm";
                    const bool learning_credit = full_generation && trial.executed &&
                        trial.host_evaluated && trial.verifier_known &&
                        trial.outcome == common_flydelta_counterfactual_outcome::helped;
                    const char * credit_reason = !full_generation
                        ? "diagnostic_only_no_full_generation_oracle_result"
                        : !trial.executed ? "candidate_generation_failed"
                        : !trial.host_evaluated ? "host_verifier_not_run"
                        : !trial.verifier_known ? "host_verifier_unknown"
                        : trial.outcome == common_flydelta_counterfactual_outcome::helped
                            ? "baseline_failed_candidate_passed"
                            : trial.outcome == common_flydelta_counterfactual_outcome::harmed
                                ? "baseline_passed_candidate_failed"
                                : trial.outcome == common_flydelta_counterfactual_outcome::neutral
                                    ? "baseline_and_candidate_passed"
                                    : "baseline_and_candidate_did_not_establish_helped";
                    ordered_json arm = {
                        {"group", entry.first}, {"fixture_id", deep_fixture.id},
                        {"evaluation_scenario_id", deep_case.id},
                        {"task", deep_case.question}, {"arm_id", trial.arm_id},
                        {"raw_output_trace_id", trial.arm_id},
                        {"requested_rank", requested_rank}, {"trial_index", trial_index},
                        {"basis_identity", common_flydelta_coefficient_basis_identity(deep_result.basis)},
                        {"coefficients", trial.coefficients}, {"layer", deep_result.layer_index},
                        {"dose_experiment", dose_experiment},
                        {"calibration_ref", trial.residual_calibration_ref},
                        {"requested_coefficient_norm", trial.requested_strength},
                        {"executed_coefficient_norm", trial.executed_strength},
                        {"requested_residual_fraction", trial.requested_residual_fraction},
                        {"applied_residual_fraction", trial.applied_residual_fraction},
                        {"requested_intervention_norm", trial.requested_intervention_norm},
                        {"applied_intervention_norm", trial.applied_intervention_norm},
                        {"behavior_delta_relative_displacement", trial.relative_dose},
                        {"dose_reason", trial.dose_reason},
                        {"mutation", trial.mutation_kind},
                        {"executed", trial.executed}, {"host_evaluated", trial.host_evaluated},
                        {"verifier_known", trial.verifier_known},
                        {"verifier_passed", full_generation && trial.verifier_known &&
                            trial.outcome != common_flydelta_counterfactual_outcome::harmed &&
                            trial.outcome != common_flydelta_counterfactual_outcome::unknown},
                        {"outcome", common_flydelta_counterfactual_outcome_name(trial.outcome)},
                        {"counterfactual_credit_eligible", learning_credit},
                        {"counterfactual_credit_reason", credit_reason},
                        {"learning_credit_granted", false},
                        {"learning_credit_grant_reason", "rank_smoke_does_not_persist_learning_credit"},
                        {"expected_tool", deep_case.expected_tool},
                        {"expected_arguments", deep_case.canonical_arguments},
                        {"expected_decision", trial.expected_decision_summary},
                        {"observed_decision", trial.observed_decision_summary},
                        {"verifier_reason", trial.verifier_reason},
                        {"evidence_ref", trial.evidence_ref},
                        {"search_fitness", trial.search_fitness},
                    };
                    if (!trial.intervention_norms_available) {
                        arm["requested_intervention_norm"] = nullptr;
                        arm["applied_intervention_norm"] = nullptr;
                    }
                    if (trial.residual_calibration_ref.empty()) {
                        arm["requested_residual_fraction"] = nullptr;
                        arm["applied_residual_fraction"] = nullptr;
                    }
                    if (trial.margin.available) {
                        arm["margin_normalized"] = trial.margin.normalized_delta();
                    }
                    if (trial.geometry_available) {
                        arm["geometry"] = {{"cosine", trial.geometry.cosine},
                            {"progress", trial.geometry.progress},
                            {"leakage", trial.geometry.leakage},
                            {"shift_norm", trial.geometry.shift_norm}};
                    }
                    std::fprintf(stderr, "flydelta_low_rank_arm %s\n", arm.dump().c_str());
                }
            }
        }

        // The rank smoke is intentionally scoped to the Deep rank comparison.
        // Running the ordinary Whirlpool region pipeline as well repeats
        // expensive model calls without contributing to the rank measurement.
        // Groups without two compatible directions cannot support comparison.
        if (value.low_rank_rank_sweep || value.dose_comparison) {
            if (entry.second.size() < 2 || directions.size() < required_directions) {
                std::cout << "flydelta_low_rank_rank_sweep_skipped group=" << entry.first
                          << " reason=insufficient_compatible_directions"
                          << " observations=" << entry.second.size()
                          << " direction_candidates=" << directions.size() << '\n';
            }
            continue;
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

                std::optional<host_verdict> trace_baseline_verdict;
                std::string trace_baseline_arm_id;
                std::string trace_baseline_output;
                const auto prepare_arm = [&](const common_flydelta_arm_request & arm_request,
                        common_agent_generation_request & generation_request,
                        std::string & arm_error) {
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

                    generation_request = make_request(
                        value, contract, bootstrap_case.question, capture, activation_ptr);
                    return true;
                };

                const auto finalize_arm = [&](const common_flydelta_arm_request & arm_request,
                        const common_agent_generation_result & generated_result,
                        common_flydelta_arm_result & arm_result,
                        std::string & arm_error) {
                    if (std::getenv("LLAMA_AGENT_RESIDENT_TRACE") != nullptr) {
                        // Diagnostic arms use n_predict=0. Observe their actual free
                        // response in a separate bounded replay of the same prompt
                        // and overlay, without admitting it into search or credit.
                        common_agent_generation_request trace_request;
                        common_agent_generation_result trace_result = generated_result;
                        std::string trace_error;
                        const auto trace_start = std::chrono::steady_clock::now();
                        bool trace_ok = prepare_arm(arm_request, trace_request, trace_error);
                        bool additional_model_call = false;
                        if (trace_ok && !arm_request.request_generation) {
                            trace_request.flydelta_capture.reset();
                            additional_model_call = true;
                            trace_ok = inference->generate(trace_request, trace_result);
                        } else if (trace_ok) {
                            common_agent_server_context_host_apply_flydelta_arm_execution_contract(
                                arm_request, trace_request);
                        }
                        const bool generation_ok = trace_ok && common_agent_generation_succeeded(trace_result);
                        const bool overlay_ok = !arm_request.apply_overlay || trace_result.flydelta_runtime.applied;
                        const bool verified = generation_ok && overlay_ok;
                        host_verdict trace_verdict;
                        trace_verdict.value = host_verdict::kind::unresolved_alternative;
                        if (verified) {
                            trace_verdict = verify_model_call(trace_result, bootstrap_case.expected_tool,
                                bootstrap_case.canonical_arguments, bootstrap_case.verification_mode,
                                &bootstrap_case.canonical_execution, host);
                        }
                        if (!arm_request.apply_overlay) {
                            trace_baseline_arm_id = arm_request.arm_id;
                            trace_baseline_output = trace_result.content;
                            trace_baseline_verdict = verified ? std::optional<host_verdict>(trace_verdict)
                                : std::nullopt;
                        }
                        trace_model_exchange(arm_request.apply_overlay ? "flydelta_trace_candidate"
                            : "flydelta_trace_baseline", arm_request.arm_id, trace_request, trace_result);
                        const ordered_json event = {
                            {"job_id", arm_request.job_id}, {"arm_id", arm_request.arm_id},
                            {"fixture_id", bootstrap_case.id}, {"question", bootstrap_case.question},
                            {"layer_indices", arm_request.layer_indices},
                            {"coefficients", arm_request.coefficients}, {"alpha", arm_request.alpha},
                            {"trace_only", true}, {"outcome_source", "trace_only_arm_verification"},
                            {"additional_model_calls", additional_model_call ? 1 : 0},
                            {"elapsed_ms", std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - trace_start).count()},
                            {"baseline_arm_id", trace_baseline_arm_id},
                            {"before_output", trace_baseline_output}, {"output", trace_result.content},
                            {"expected_call", {{"name", bootstrap_case.expected_tool},
                                {"arguments", bootstrap_case.canonical_arguments}}},
                            {"baseline_verifier_known", trace_baseline_verdict.has_value() &&
                                trace_baseline_verdict->value != host_verdict::kind::unresolved_alternative},
                            {"baseline_passed", trace_baseline_verdict.has_value() &&
                                trace_baseline_verdict->value == host_verdict::kind::passed},
                            {"verifier_known", verified &&
                                trace_verdict.value != host_verdict::kind::unresolved_alternative},
                            {"passed", verified && trace_verdict.value == host_verdict::kind::passed},
                            {"overlay_requested", arm_request.apply_overlay},
                            {"overlay_applied", trace_result.flydelta_runtime.applied},
                            {"verifier_reason", trace_verdict.diagnostic},
                            {"error", !trace_error.empty() ? trace_error
                                : !generation_ok ? (trace_result.error_message.empty()
                                    ? "trace generation failed" : trace_result.error_message)
                                : !overlay_ok ? "trace overlay application was not confirmed" : ""},
                        };
                        std::fprintf(stderr, "flydelta_arm_response %s\n", event.dump().c_str());
                    }
                    if (!common_agent_generation_succeeded(generated_result)) {
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
                    arm_result.margin_available = false;
                    if (!arm_request.apply_overlay && generated_result.flydelta_capture &&
                            generated_result.flydelta_capture->captured) {
                        baseline_capture = generated_result.flydelta_capture;
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
                    return true;
                };

                const auto score_teacher_forced_margin_batch = [&](
                        const std::vector<common_flydelta_arm_request> & arms,
                        const std::vector<common_agent_generation_request> & contexts,
                        common_agent_inference & scorer,
                        std::vector<common_flydelta_decision_margin> & margins,
                        std::string & score_error) {
                    score_error.clear();
                    margins.assign(arms.size(), {});
                    if (arms.size() != contexts.size()) {
                        score_error = "dataset repair smoke received mismatched arm and score contexts";
                        return false;
                    }
                    // A free-form baseline can be a perfectly valid failed
                    // generation yet provide no canonical negative decision.
                    // That is an unavailable *fixture* margin, not a batch
                    // execution failure. Replay fixtures below supply an
                    // explicit host-owned contrast when the smoke needs to
                    // exercise teacher-score batching.
                    if (bootstrap_case.failed_tool.empty() ||
                            bootstrap_case.failed_continuation.empty()) {
                        return true;
                    }

                    common_agent_teacher_forced_choice_batch_request score_batch;
                    std::vector<size_t> score_indices;
                    score_batch.choices.reserve(arms.size());
                    score_indices.reserve(arms.size());
                    for (size_t index = 0; index < arms.size(); ++index) {
                        if (!arms[index].request_teacher_forced_margin) continue;
                        common_agent_teacher_forced_choice_request score_request;
                        score_request.sequence_id = arms[index].arm_id;
                        // Rebuild the immutable score context. Generation
                        // requests are transport input and may have been
                        // consumed by a backend implementation; the host
                        // keeps the semantic arm->context mapping here.
                        if (!prepare_arm(arms[index], score_request.context, score_error)) return false;
                        score_request.context.flydelta_capture.reset();
                        score_request.positive_choice = bootstrap_case.expected_tool;
                        score_request.negative_choice = bootstrap_case.failed_tool;
                        if (!agent_flydelta_dataset_dsl_contrast(score_request,
                                tool_call_continuation(bootstrap_case.expected_tool, bootstrap_case.canonical_arguments, host),
                                bootstrap_case.failed_continuation, score_error)) return false;
                        score_batch.choices.push_back(std::move(score_request));
                        score_indices.push_back(index);
                    }
                    if (score_batch.choices.empty()) return true;

                    common_agent_teacher_forced_choice_batch_result score_result;
                    if (!scorer.score_teacher_forced_choice_batch(score_batch, score_result)) {
                        score_error = score_result.error_message.empty()
                            ? "dataset repair smoke teacher-forced batch scoring failed"
                            : score_result.error_message;
                        return false;
                    }
                    if (score_result.choices.size() != score_indices.size()) {
                        score_error = "dataset repair smoke teacher-forced batch returned incomplete scores";
                        return false;
                    }
                    for (size_t index = 0; index < score_indices.size(); ++index) {
                        const auto & choice = score_result.choices[index];
                        auto & margin = margins[score_indices[index]];
                        margin.available = choice.available;
                        margin.positive_total_logprob = choice.positive_total_logprob;
                        margin.negative_total_logprob = choice.negative_total_logprob;
                        margin.positive_token_count = choice.positive_token_count;
                        margin.negative_token_count = choice.negative_token_count;
                    }
                    return true;
                };

                std::shared_ptr<const common_flydelta_model_host> model_host;
                if (value.backend == "server") {
                    common_agent_server_flydelta_binding_callbacks callbacks;
                    callbacks.primitives.capture = true;
                    callbacks.primitives.overlay = true;
                    callbacks.primitives.generation = true;
                    callbacks.primitives.teacher_forced_scoring = true;
                    callbacks.primitives.host_verification = true;
                    callbacks.prepare_arm = prepare_arm;
                    callbacks.finalize_arm = finalize_arm;
                    callbacks.score_teacher_forced_margin_batch = score_teacher_forced_margin_batch;
                    callbacks.register_evaluator = [](
                            common_flydelta_evaluator_config &,
                            common_flydelta_evaluator_callbacks &,
                            std::string & register_error) {
                        register_error.clear();
                        return true;
                    };
                    auto binding = common_agent_server_flydelta_binding_from_callbacks(
                        std::move(callbacks));
                    model_host = common_agent_server_context_host_make_flydelta_model_host(
                        server_host, std::move(binding), runner_error);
                    if (!model_host) return false;
                    if (!model_host->capabilities.bounded_arm_batch) {
                        runner_error = "dataset repair smoke server host did not advertise FlyDelta batch capability";
                        return false;
                    }
                } else {
                    auto scalar_host = std::make_shared<common_flydelta_model_host>();
                    scalar_host->capabilities.capture = true;
                    scalar_host->capabilities.overlay = true;
                    scalar_host->capabilities.generation = true;
                    scalar_host->capabilities.teacher_forced_scoring = true;
                    scalar_host->capabilities.host_verification = true;
                    scalar_host->run_bounded_arm = [&, prepare_arm, finalize_arm,
                            score_teacher_forced_margin_batch](
                            const common_flydelta_arm_request & arm_request,
                            common_flydelta_arm_result & arm_result,
                            std::string & arm_error) {
                        common_agent_generation_request generation_request;
                        if (!prepare_arm(arm_request, generation_request, arm_error)) return false;
                        common_agent_generation_result generated_result;
                        if (!inference->generate(generation_request, generated_result)) {
                            arm_error = generated_result.error_message.empty()
                                ? "dataset model host generation failed" : generated_result.error_message;
                            return false;
                        }
                        if (!finalize_arm(arm_request, generated_result, arm_result, arm_error)) return false;
                        std::vector<common_flydelta_decision_margin> margins;
                        if (!score_teacher_forced_margin_batch(
                                {arm_request}, {generation_request}, *inference, margins, arm_error)) {
                            return false;
                        }
                        if (margins.size() != 1) {
                            arm_error = "dataset repair smoke scalar teacher scoring returned no margin";
                            return false;
                        }
                        arm_result.margin = margins.front();
                        arm_result.margin_available = arm_result.margin.available;
                        arm_result.margin_total = arm_result.margin.total_delta();
                        arm_result.margin_normalized = arm_result.margin.normalized_delta();
                        return true;
                    };
                    model_host = std::move(scalar_host);
                }

                const auto model_runner = common_flydelta_search_pipeline_runner_from_model_host(
                    *model_host, queued_job.id, "context://dataset-question-repair",
                    "intervention://dataset-question-repair/" + bootstrap_case.id,
                    true, true, false, false, 64U * 1024U * 1024U, 0);
                const auto model_batch_runner =
                    common_flydelta_search_pipeline_batch_runner_from_model_host(
                        *model_host, queued_job.id, "context://dataset-question-repair",
                        "intervention://dataset-question-repair/" + bootstrap_case.id,
                        true, true, false, false, 64U * 1024U * 1024U, 0);
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
            bootstrap_overlay_evaluated = bootstrap_overlay_evaluated || worker_executed;
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
            // The server path must preserve the same fixture-specific
            // teacher-forced margin that scalar execution exposes. A missing
            // margin is a smoke failure, not a reason to silently downgrade
            // the search objective merely because a logical arm wave was
            // batched.
            if (value.backend == "server" &&
                    !bootstrap_case.failed_tool.empty() &&
                    !bootstrap_case.failed_continuation.empty()) {
                const auto & region_trials = pipeline_result.directions.front().region_trials;
                const bool all_margins_available = !region_trials.empty() &&
                    std::all_of(region_trials.begin(), region_trials.end(),
                        [](const common_flydelta_intervention_region_trial & trial) {
                            return trial.margin.available;
                        });
                if (!all_margins_available) {
                    host.close();
                    std::cerr << "Search-pipeline server batch did not preserve teacher-forced margins for "
                              << entry.first << '\n';
                    return 1;
                }
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
                        score_request.positive_choice = bootstrap_case.expected_tool;
                        score_request.negative_choice = bootstrap_case.failed_tool;
                        const bool contrast_valid = agent_flydelta_dataset_dsl_contrast(score_request,
                            tool_call_continuation(bootstrap_case.expected_tool, bootstrap_case.canonical_arguments, host),
                            bootstrap_case.failed_continuation, margin_error);
                        common_agent_teacher_forced_choice_result score_result;
                        if (contrast_valid && inference->score_teacher_forced_choice(score_request, score_result) &&
                                score_result.available) {
                            margin.available = true;
                            margin.positive_total_logprob = score_result.positive_total_logprob;
                            margin.negative_total_logprob = score_result.negative_total_logprob;
                            margin.positive_token_count = score_result.positive_token_count;
                            margin.negative_token_count = score_result.negative_token_count;
                        } else {
                            if (margin_error.empty()) margin_error = score_result.error_message.empty()
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
              << " verified_overlay_trials=" << deep_verified_overlay_trials
              << " forced_tfo_lite=" << (value.force_tfo_lite ? "yes" : "no")
              << " bootstrap_overlay_evaluated=" << (bootstrap_overlay_evaluated ? "yes" : "no")
              << " deep_search_executed=" << (deep_search_executed != 0 ? "yes" : "no")
              << " repair_echo_persisted=" << persisted_echo_failures
              << " learning_ledger=" << learning_ledger_path << '\n';
    if (value.force_deep && deep_search_executed == 0) {
        std::cout << "flydelta_low_rank_rank_sweep_skipped reason=no_eligible_baseline_capture_pairs"
                  << " phase=baseline unresolved=" << unresolved
                  << " evaluation_status=unavailable learning_credit=none\n";
    }
    if (remove_learning_ledger) {
        std::error_code ledger_remove_error;
        std::filesystem::remove(learning_ledger_path, ledger_remove_error);
    }
    if (value.dose_comparison && deep_verified_overlay_trials == 0) {
        std::cerr << "dose comparison incomplete: no full-generation arm was applied and host evaluated\n";
        return 1;
    }
    return 0;
}
