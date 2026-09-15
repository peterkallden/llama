#include "agent-flydelta-dataset-repair-host.h"

#include "agent/adaptation/flydelta/flydelta-capture.h"
#include "agent/adaptation/flydelta/flydelta-direction-search.h"
#include "agent/adaptation/flydelta/flydelta-evidence.h"
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
#include <vector>

namespace {
using json = nlohmann::ordered_json;

struct options {
    std::string model;
    std::string suite;
    int n_predict = 128;
    int n_threads = 3;
    int n_gpu_layers = 0;
};

struct host_verdict {
    enum class kind { passed, certified_failure, unresolved_alternative } value = kind::certified_failure;
    std::string selected_tool;
    std::string diagnostic;
};

bool parse_args(int argc, char ** argv, options & value) {
    if (const char * model = std::getenv("LLAMA_AGENT_MODEL")) value.model = model;
    if (const char * suite = std::getenv("LLAMA_AGENT_DATASET_QUESTION_SUITE")) value.suite = suite;
    if (const char * threads = std::getenv("LLAMA_AGENT_THREADS")) value.n_threads = std::stoi(threads);
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        const auto next = [&](const char * name) -> const char * {
            if (index + 1 >= argc) { std::cerr << "missing value for " << name << '\n'; return nullptr; }
            return argv[++index];
        };
        if (argument == "--model") { const auto v = next("--model"); if (!v) return false; value.model = v; }
        else if (argument == "--suite") { const auto v = next("--suite"); if (!v) return false; value.suite = v; }
        else if (argument == "--n-predict") { const auto v = next("--n-predict"); if (!v) return false; value.n_predict = std::stoi(v); }
        else if (argument == "--threads") { const auto v = next("--threads"); if (!v) return false; value.n_threads = std::stoi(v); }
        else if (argument == "--n-gpu-layers") { const auto v = next("--n-gpu-layers"); if (!v) return false; value.n_gpu_layers = std::stoi(v); }
        else if (argument == "--help" || argument == "-h") return false;
        else { std::cerr << "unknown argument: " << argument << '\n'; return false; }
    }
    return true;
}

bool read_json(const std::string & path, json & value, std::string & error) {
    std::ifstream input(path, std::ios::binary);
    if (!input) { error = "could not open suite: " + path; return false; }
    std::ostringstream contents;
    contents << input.rdbuf();
    value = json::parse(contents.str(), nullptr, false);
    if (value.is_discarded() || !value.is_object() || !value.contains("scenarios") ||
            !value["scenarios"].is_array()) {
        error = "dataset repair suite is invalid";
        return false;
    }
    return true;
}

std::string preview(const common_agent_generation_result & result) {
    if (!common_agent_generation_succeeded(result)) return "<generation failed: " + result.error_message + ">";
    std::string output = result.content;
    for (char & character : output) if (character == '\n' || character == '\r' || character == '\t') character = ' ';
    if (output.size() > 300) output.resize(300);
    return output;
}

host_verdict verify_model_call(const common_agent_generation_result & result,
        const std::string & expected_tool, agent_flydelta_dataset_repair_host & host) {
    host_verdict verdict;
    if (!common_agent_generation_succeeded(result)) {
        verdict.diagnostic = "model generation failed";
        return verdict;
    }
    const json parsed = json::parse(result.content, nullptr, false);
    if (!parsed.is_object() || !parsed.contains("name") || !parsed["name"].is_string() ||
            !parsed.contains("arguments") || !parsed["arguments"].is_object()) {
        verdict.diagnostic = "host rejected a non-canonical tool-call object";
        return verdict;
    }
    verdict.selected_tool = parsed["name"].get<std::string>();
    common_tool_execution_result execution;
    std::string error;
    if (!host.execute_call(verdict.selected_tool, parsed["arguments"], execution, error)) {
        verdict.diagnostic = "host rejected tool call: " + error;
        return verdict;
    }
    if (verdict.selected_tool != expected_tool) {
        // A different executable read-only tool can still be semantically
        // useful. This fixture has no equivalence oracle, so it remains
        // UNKNOWN rather than creating false tool-repair evidence.
        verdict.value = host_verdict::kind::unresolved_alternative;
        verdict.diagnostic = "different executable tool; semantic equivalence is unknown";
        return verdict;
    }
    verdict.value = host_verdict::kind::passed;
    verdict.diagnostic = "host executed the expected canonical tool";
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
        const std::string & user, const std::shared_ptr<const common_flydelta_hidden_state_capture_request> & capture) {
    common_agent_generation_request request;
    request.purpose = common_agent_generation_purpose::tool_followup;
    request.options.n_predict = value.n_predict;
    request.options.n_threads = value.n_threads;
    request.messages = {{"system", "You are a host-controlled dataset tool selector. Return only one canonical JSON object with name and arguments, no markdown.\n" + contract}, {"user", user}};
    request.flydelta_capture = capture;
    return request;
}
}

int main(int argc, char ** argv) {
    options value;
    if (!parse_args(argc, argv, value)) {
        std::cerr << "usage: " << argv[0] << " --model MODEL --suite SUITE_JSON [--threads N] [--n-gpu-layers N]\n";
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
    agent_flydelta_dataset_repair_host host;
    if (!host.open("model", error)) { std::cerr << error << '\n'; return 1; }

    common_tool_catalog catalog;
    common_tool_bootstrap_result bootstrap;
    if (!catalog.bootstrap("analysis", bootstrap, error)) { host.close(); std::cerr << error << '\n'; return 1; }
    std::string contract = "Available read-only tools:\n";
    for (const auto & scenario : suite["scenarios"]) {
        const auto expected = scenario.value("expected_tool", "");
        const auto * definition = catalog.find_definition(expected);
        if (!definition) { host.close(); std::cerr << "unknown expected tool: " << expected << '\n'; return 1; }
        std::string compact_error;
        const auto description = common_render_compact_tool_description(definition->name, definition->description,
            common_tool_model_input_schema(*definition), common_tool_model_result_schema(*definition), compact_error);
        if (!compact_error.empty()) { host.close(); std::cerr << compact_error << '\n'; return 1; }
        if (contract.find("\n- " + expected + "\n") == std::string::npos) contract += "- " + description + "\n";
    }

    common_agent_model_selection selection;
    selection.profile_id = "flydelta-dataset-question-repair";
    selection.base_model_id = "generation-base";
    selection.backend = "cli";
    selection.path = value.model;
    selection.context_size_tokens = 2048;
    selection.load_policy = "resident";
    common_agent_runtime_cli_model_loader loader({value.n_gpu_layers, value.n_threads, true});
    std::shared_ptr<common_agent_runtime_resident_model> resident;
    if (!loader.load(selection, resident, error)) { host.close(); std::cerr << error << '\n'; return 1; }
    const auto loaded = common_agent_runtime_loaded_model_cast(resident);
    if (!loaded || !loaded->model || !loaded->chat_templates) { host.close(); return 1; }
    auto inference = make_llama_cli_agent_inference(loaded->model, loaded->chat_templates.get());
    const size_t n_embd = static_cast<size_t>(llama_model_n_embd(loaded->model));
    const size_t n_layers = static_cast<size_t>(llama_model_n_layer(loaded->model));
    if (n_embd == 0 || n_layers <= 2) { host.close(); return 1; }

    auto capture = std::make_shared<common_flydelta_hidden_state_capture_request>();
    capture->enabled = true;
    for (uint32_t layer = 1; layer < n_layers && layer <= 4; ++layer) capture->layer_indices.push_back(layer);
    capture->token_index = -1;
    capture->position = common_flydelta_capture_position::generation_boundary;
    capture->max_bytes = 4U * 1024U * 1024U;
    capture->model_profile_fingerprint = "sha256:flydelta-dataset-question-repair";
    capture->capture_layout_revision = "layer-input:generation-boundary:v1";

    std::vector<common_flydelta_contrast_sample> samples;
    size_t passed = 0, unresolved = 0, failures = 0, repaired = 0;
    for (const auto & scenario : suite["scenarios"]) {
        const auto id = scenario.value("id", "unnamed");
        const auto expected = scenario.value("expected_tool", "");
        const auto steps = scenario.value("plan", json::object()).value("steps", json::array());
        if (steps.size() != 1) { host.close(); std::cerr << "scenario has invalid plan: " << id << '\n'; return 1; }
        common_tool_execution_result canonical_execution;
        if (!host.execute_step(steps.front(), canonical_execution, error)) {
            std::cout << "scenario=" << id << " host_verifier=unavailable error=" << error << '\n';
            continue;
        }

        common_agent_generation_result failed_result;
        const bool generated = inference->generate(make_request(value, contract, scenario.value("question", ""), capture), failed_result);
        const auto failed_verdict = verify_model_call(failed_result, expected, host);
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
        const json canonical_repair = {{"name", expected}, {"arguments", steps.front()["args"]}};
        const std::string repair_request = "The host rejected the prior tool-call attempt (" +
            failed_verdict.diagnostic + "). The host constructed the canonical repair below. " +
            "Return this JSON object unchanged, including both name and arguments, with no markdown or explanation: " +
            canonical_repair.dump();
        common_agent_generation_result repaired_result;
        const bool repair_generated = inference->generate(make_request(value, contract, repair_request, capture), repaired_result);
        const auto repaired_verdict = verify_model_call(repaired_result, expected, host);
        const bool capture_pair = failed_result.flydelta_capture && repaired_result.flydelta_capture &&
            failed_result.flydelta_capture->captured && repaired_result.flydelta_capture->captured;
        if (!repair_generated || repaired_verdict.value != host_verdict::kind::passed || !capture_pair) {
            std::cout << "scenario=" << id << " host_outcome=repair_failed selected=" << repaired_verdict.selected_tool
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
                "sha256:dataset-question:" + id, "tool_use/dataset/structured_call_repair",
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
        for (const auto & delta : deltas) if (delta.layer_index == 2) samples.push_back({delta, credit});
        ++repaired;
        std::cout << "scenario=" << id << " host_outcome=host_certified_repair"
                  << " transition=" << transition.id << " selected=" << repaired_verdict.selected_tool
                  << " failed_output=" << preview(failed_result)
                  << " repaired_output=" << preview(repaired_result) << '\n';
    }
    host.close();

    common_flydelta_direction_search_config direction_config;
    direction_config.dimension = n_embd;
    direction_config.layer_index = 2;
    direction_config.min_samples = 6;
    direction_config.max_samples = 32;
    direction_config.min_median_alignment = 0.25f;
    direction_config.trim_fraction = 0.20f;
    direction_config.variance_ridge = 0.001f;
    direction_config.source = common_adaptation_evidence_source::tool_repair;
    direction_config.behavior_key = "tool_use/dataset/structured_call_repair";
        direction_config.model_profile_fingerprint = capture->model_profile_fingerprint;
        direction_config.execution_context_fingerprint =
            "sha256:dataset-question-context-v1";
    direction_config.capture_layout_revision = capture->capture_layout_revision;
    std::vector<common_flydelta_direction_candidate> directions;
    if (!samples.empty() && !common_flydelta_build_direction_candidates(direction_config, samples, directions, error)) {
        std::cerr << "direction search failed: " << error << '\n';
        return 1;
    }
    std::cout << "flydelta_dataset_question_repair_model_smoke"
              << " passed=" << passed << " failures=" << failures << " unresolved=" << unresolved
              << " host_certified_repairs=" << repaired << " layer2_samples=" << samples.size()
              << " deep_ready=" << (samples.size() >= direction_config.min_samples ? "yes" : "no")
              << " direction_candidates=" << directions.size() << '\n';
    return 0;
}
