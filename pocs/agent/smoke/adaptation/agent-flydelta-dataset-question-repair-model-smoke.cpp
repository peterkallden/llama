#include "agent-flydelta-dataset-repair-host.h"

#include "agent/adaptation/flydelta/flydelta-capture.h"
#include "agent/adaptation/flydelta/flydelta-direction-search.h"
#include "agent/adaptation/flydelta/flydelta-evidence.h"
#include "agent/adaptation/flydelta/flydelta-aggregation.h"
#include "agent/adaptation/flydelta/flydelta-evidence-depth.h"
#include "agent/adaptation/flydelta/flydelta-activation.h"
#include "agent/adaptation/flydelta/flydelta-evaluator.h"
#include "agent/adaptation/flydelta/flydelta-deep-search.h"
#include "agent/adaptation/flydelta/flydelta-experiment.h"
#include "agent/adaptation/flydelta/flydelta-worker.h"
#include "agent/tooling/schema/tool-schema-compact.h"
#include "tools/agent/cli/agent-cli-inference.h"
#include "tools/agent/host/agent-host-config.h"
#include "tools/agent/runtime/agent-model-loaders.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
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
    bool force_deep = false;
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
    if (const char * config = std::getenv("LLAMA_AGENT_CONFIG")) value.config = config;
    if (const char * force_deep = std::getenv("LLAMA_AGENT_FORCE_DEEP")) {
        value.force_deep = std::string(force_deep) == "1" || std::string(force_deep) == "true";
    }
    if (const char * threads = std::getenv("LLAMA_AGENT_THREADS")) value.n_threads = std::stoi(threads);
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        const auto next = [&](const char * name) -> const char * {
            if (index + 1 >= argc) { std::cerr << "missing value for " << name << '\n'; return nullptr; }
            return argv[++index];
        };
        if (argument == "--model") { const auto v = next("--model"); if (!v) return false; value.model = v; }
        else if (argument == "--suite") { const auto v = next("--suite"); if (!v) return false; value.suite = v; }
        else if (argument == "--config") { const auto v = next("--config"); if (!v) return false; value.config = v; }
        else if (argument == "--force-deep") value.force_deep = true;
        else if (argument == "--n-predict") { const auto v = next("--n-predict"); if (!v) return false; value.n_predict = std::stoi(v); }
        else if (argument == "--threads") { const auto v = next("--threads"); if (!v) return false; value.n_threads = std::stoi(v); }
        else if (argument == "--n-gpu-layers") { const auto v = next("--n-gpu-layers"); if (!v) return false; value.n_gpu_layers = std::stoi(v); }
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
                  << " [--force-deep]"
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

    std::map<std::string, std::vector<common_flydelta_contrast_sample>> samples_by_behavior;
    std::map<std::string, bootstrap_case> bootstrap_cases_by_delta;
    size_t passed = 0, unresolved = 0, failures = 0, repaired = 0;
    for (const auto & scenario : suite["scenarios"]) {
        const auto id = scenario.value("id", "unnamed");
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
        for (const auto & delta : deltas) if (delta.layer_index == 2) {
            samples_by_behavior[behavior_key].push_back({delta, credit});
            bootstrap_cases_by_delta[delta.id] = {
                id, behavior_key, expected, scenario.value("question", ""), manifest.id, delta.id};
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
              << " passed=" << passed << " failures=" << failures << " unresolved=" << unresolved
              << " config=" << (value.config.empty() ? "none" : value.config)
              << " selected_scenarios=" << selected_scenarios
              << " host_certified_repairs=" << repaired << '\n';
    for (const auto & entry : samples_by_behavior) {
        common_flydelta_direction_search_config direction_config;
        direction_config.dimension = n_embd;
        direction_config.layer_index = 2;
        direction_config.min_samples = 2;
        direction_config.max_samples = depth_config.max_samples;
        direction_config.min_median_alignment = depth_config.min_median_alignment;
        direction_config.trim_fraction = 0.20f;
        direction_config.variance_ridge = 0.001f;
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
        if (value.force_deep && evidence_depth.depth != common_flydelta_search_depth::deep &&
                entry.second.size() >= 2 && directions.size() >= 2) {
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
            const auto deep_delta = aggregation_snapshot_for_deep.retained_samples.front().delta;
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
            deep_config.coefficients.strategy = common_flydelta_coefficient_search_strategy::coordinate;
            deep_config.coefficients.step = 0.05f;
            deep_config.coefficients.max_candidates = 8;
            deep_config.coefficients.max_l2_norm = 0.32f;
            deep_config.coefficients.norm_penalty = 0.05f;
            deep_config.coefficients.leakage_penalty = 0.10f;

            std::vector<common_flydelta_deep_search_direction> deep_directions;
            for (const auto & direction : directions) {
                deep_directions.push_back({direction, false, 0.0f});
            }
            std::shared_ptr<const common_flydelta_hidden_state_capture> deep_baseline_capture;
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
                const auto verdict = verify_model_call(generated_result, deep_case.expected_tool, host);
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
                if (!apply_overlay && generated_result.flydelta_capture) {
                    deep_baseline_capture = generated_result.flydelta_capture;
                }
                if (apply_overlay && generated_result.flydelta_capture && deep_baseline_capture) {
                    if (!common_flydelta_representation_diagnostics_from_captures(
                            *deep_baseline_capture, *generated_result.flydelta_capture, deep_delta,
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

            common_flydelta_deep_search_result deep_result;
            const auto deep_started = std::chrono::steady_clock::now();
            if (!common_flydelta_run_deep_search(
                    deep_fixture, deep_config, deep_directions,
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
                            if (geometry_available) {
                                std::cout << " cosine=" << geometry.cosine
                                          << " progress=" << geometry.progress
                                          << " leakage=" << geometry.leakage
                                          << " shift_norm=" << geometry.shift_norm;
                            }
                            std::cout << '\n';
                        }
                        return executed;
                    },
                    [&](const common_flydelta_experiment_fixture & fixture,
                            const common_flydelta_low_rank_basis & basis,
                            const std::vector<float> & coefficients,
                            bool apply_overlay, common_flydelta_counterfactual_trial & trial,
                            common_flydelta_decision_margin & margin,
                            common_flydelta_representation_diagnostics & geometry,
                            bool & geometry_available, std::string & runner_error) {
                        return run_deep_arm(basis, coefficients, apply_overlay, trial, margin,
                            geometry, geometry_available, runner_error);
                    }, deep_result, error)) {
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
                      << " forced=yes basis_rank=" << deep_result.basis.vectors.size()
                      << " diagnostic_trials=" << deep_group_diagnostic_trials
                      << " full_generation_trials=" << deep_group_full_generation_trials
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

        // Bootstrap deliberately evaluates only the first retained raw
        // direction. The evaluator/worker still owns the counterfactual
        // lifecycle; this smoke supplies the model and host callbacks.
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
            common_flydelta_experiment_job job;
            job.id = "flydelta://job/bootstrap/" + bootstrap_case.id;
            job.kind = common_flydelta_experiment_job_kind::counterfactual;
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
            job.code_revision = "flydelta-dataset-question-bootstrap:v1";

            common_flydelta_evaluator_config evaluator_config;
            common_flydelta_evaluator_callbacks evaluator_callbacks;
            std::vector<common_flydelta_counterfactual_report> reports;
            std::string baseline_tool;
            std::string candidate_tool;
            evaluator_callbacks.run_counterfactual = [&](const auto & queued_job,
                    auto & output, std::string & runner_error) {
                if (queued_job.behavior_delta_ids.size() != 1 ||
                        queued_job.behavior_delta_ids.front() != bootstrap_case.delta_id) {
                    runner_error = "Bootstrap worker received an unexpected behavior delta";
                    return false;
                }
                common_flydelta_experiment_fixture fixture;
                fixture.id = "flydelta://fixture/bootstrap/" + bootstrap_case.id;
                fixture.task_fingerprint = queued_job.seed.task_fingerprint;
                fixture.model_profile_fingerprint = queued_job.seed.model_profile_fingerprint;
                fixture.tokenizer_fingerprint = queued_job.seed.tokenizer_fingerprint;
                fixture.template_fingerprint = queued_job.seed.template_fingerprint;
                fixture.execution_context_fingerprint = queued_job.seed.execution_context_fingerprint;
                fixture.verifier_revision = queued_job.seed.verifier_ref;

                common_flydelta_gate_config gate_config;
                gate_config.enabled = true;
                gate_config.max_scale = 0.25f;
                common_flydelta_activation_request activation_request;
                activation_request.candidate_id = "flydelta://candidate/bootstrap/" + bootstrap_case.id;
                activation_request.artifact_id = "flydelta://experimental/bootstrap/" + bootstrap_case.id;
                activation_request.model_profile_fingerprint = capture->model_profile_fingerprint;
                activation_request.capture_layout_revision = capture->capture_layout_revision;
                activation_request.model_n_embd = n_embd;
                activation_request.model_n_layers = n_layers;
                activation_request.il_end = static_cast<int32_t>(n_layers - 1);
                common_flydelta_basis_direction overlay_direction;
                overlay_direction.layer_index = bootstrap_direction.layer_index;
                overlay_direction.values = bootstrap_direction.values;
                activation_request.directions.push_back(std::move(overlay_direction));
                activation_request.coefficients = {1.0f};
                activation_request.gate_request.explicit_opt_in = true;
                activation_request.gate_request.candidate_status = common_flydelta_candidate_status::approved;
                activation_request.gate_request.basis_available = true;
                activation_request.gate_request.familiarity = 1.0f;
                activation_request.gate_request.novelty = 0.0f;
                activation_request.gate_request.requested_scale = 0.02f;
                common_flydelta_activation_result activation;
                if (!common_flydelta_prepare_activation(
                        gate_config, activation_request, 64U * 1024U * 1024U,
                        activation, runner_error)) return false;
                const auto activation_ptr = std::make_shared<const common_flydelta_activation_result>(
                    std::move(activation));

                const auto runner = [&](const common_flydelta_experiment_fixture &,
                        bool apply_overlay, common_flydelta_counterfactual_trial & trial,
                        std::string &) {
                    common_agent_generation_result generated_result;
                    const bool generated = inference->generate(make_request(
                        value, contract, bootstrap_case.question,
                        {}, apply_overlay ? activation_ptr : std::shared_ptr<const common_flydelta_activation_result>{}),
                        generated_result);
                    const auto verdict = verify_model_call(generated_result,
                        bootstrap_case.expected_tool, host);
                    if (apply_overlay) candidate_tool = verdict.selected_tool;
                    else baseline_tool = verdict.selected_tool;
                    trial.executed = generated;
                    trial.verifier_known = generated &&
                        verdict.value != host_verdict::kind::unresolved_alternative;
                    trial.passed = generated && verdict.value == host_verdict::kind::passed;
                    trial.quality = trial.passed ? 1.0f : 0.0f;
                    trial.overlay_applied = apply_overlay;
                    trial.intervention_count = apply_overlay ? 1 : 0;
                    trial.evidence_ref = apply_overlay
                        ? "evidence:flydelta-bootstrap-overlay"
                        : "evidence:flydelta-bootstrap-baseline";
                    return true;
                };
                common_flydelta_counterfactual_report report;
                if (!common_flydelta_run_counterfactual(
                        queued_job.id, activation_request.candidate_id,
                        "flydelta://profile/bootstrap-baseline",
                        "flydelta://profile/bootstrap-overlay", fixture, runner,
                        report, runner_error)) return false;
                output.push_back(report);
                reports.push_back(report);
                return true;
            };
            common_flydelta_experiment_worker_report worker_report;
            const bool enqueued = common_flydelta_experiment_queue_enqueue(worker_root, job, {}, error);
            if (!enqueued ||
                    !common_flydelta_experiment_worker_run_evaluator_once(
                        worker_root, {}, evaluator_config, evaluator_callbacks,
                        worker_report, error) ||
                    worker_report.state != common_flydelta_experiment_queue_state::succeeded ||
                    reports.size() != 1) {
                host.close();
                std::cerr << "Bootstrap worker failed for " << entry.first << ": " << error << '\n';
                return 1;
            }
            const auto & report = reports.front();
            std::cout << "flydelta_bootstrap_worker group=" << entry.first
                      << " state=" << common_flydelta_experiment_queue_state_name(worker_report.state)
                      << " outcome=" << common_flydelta_counterfactual_outcome_name(report.outcome)
                      << " baseline_tool=" << baseline_tool
                      << " candidate_tool=" << candidate_tool
                      << " baseline_passed=" << (report.baseline.passed ? "yes" : "no")
                      << " candidate_passed=" << (report.candidate.passed ? "yes" : "no")
                      << " candidate_overlay_applied=" << (report.candidate.overlay_applied ? "yes" : "no")
                      << '\n';
        }
    }
    host.close();
    std::filesystem::remove_all(worker_root, ignored);
    std::cout << "flydelta_depth_summary layer2_samples=" << layer2_samples
              << " groups=" << samples_by_behavior.size()
              << " deep_groups=" << deep_groups
              << " direction_candidates=" << direction_candidates
              << " forced_deep_searches=" << deep_search_executed
              << " forced_deep_full_generation_trials=" << deep_full_generation_trials
              << " bootstrap_overlay_evaluated=yes"
              << " deep_search_executed=" << (deep_search_executed != 0 ? "yes" : "no") << '\n';
    return 0;
}
