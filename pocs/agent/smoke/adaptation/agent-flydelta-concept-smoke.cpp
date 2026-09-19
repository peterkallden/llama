#include "agent/adaptation/flydelta/flydelta-concept.h"
#include "agent/adaptation/flydelta/flydelta-activation.h"
#include "agent/adaptation/flydelta/flydelta-hidden-state-hook.h"
#include "agent/adaptation/flydelta/flydelta-model-adapter.h"
#include "agent/adaptation/flydelta/flydelta-search-pipeline.h"
#include "tools/agent/cli/agent-cli-generation.h"
#include "tools/agent/cli/agent-cli-inference.h"
#include "tools/agent/runtime/agent-model-loaders.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

struct options {
    std::string model;
    int n_predict = 64;
    int n_threads = 3;
    int n_gpu_layers = 0;
};

struct concept_family {
    const char * key;
    const char * behavior;
    const char * expected_tool;
    const char * procedure;
};

struct concept_evaluation_fixture {
    const char * id;
    const char * family;
    const char * split;
    const char * request;
    const char * expected_tool;
    const char * decision_negative_tool;
};

const char * detected_tool(const common_agent_generation_result & result);
common_agent_generation_request request(
        const options & value,
        const std::string & instruction,
        const std::shared_ptr<const common_flydelta_hidden_state_capture_request> & capture);

const char * extraction_request(const concept_family & family, size_t index) {
    if (std::string(family.key) == "grouped-aggregation") {
        return index == 0
            ? "Calculate total amount grouped by region."
            : "Find the sum of sales amount for every region.";
    }
    if (std::string(family.key) == "validated-filter") {
        return index == 0
            ? "Return rows where status is failed."
            : "Keep only records whose region is North.";
    }
    return index == 0
        ? "Return the largest orders ordered by amount."
        : "List the five longest requests by duration.";
}

std::string generation_preview(const common_agent_generation_result & result) {
    std::string preview = result.content.substr(0, 160);
    for (char & value : preview) {
        if (static_cast<unsigned char>(value) < 0x20 || value == 0x7f) value = ' ';
    }
    return preview;
}

const char * canonical_conditioned_call(const concept_family & family, size_t index) {
    if (std::string(family.key) == "grouped-aggregation") {
        return index == 0
            ? "{\"name\":\"data.aggregate\",\"arguments\":{\"dataset\":\"sales.csv\",\"group_by\":[\"region\"],\"measure\":\"amount\"}}"
            : "{\"name\":\"data.aggregate\",\"arguments\":{\"dataset\":\"sales.csv\",\"group_by\":[\"region\"],\"measure\":\"sales_amount\"}}";
    }
    if (std::string(family.key) == "validated-filter") {
        return index == 0
            ? "{\"name\":\"data.filter\",\"arguments\":{\"dataset\":\"sales.csv\",\"predicate\":\"status == failed\"}}"
            : "{\"name\":\"data.filter\",\"arguments\":{\"dataset\":\"sales.csv\",\"predicate\":\"region == North\"}}";
    }
    return index == 0
        ? "{\"name\":\"data.query\",\"arguments\":{\"dataset\":\"sales.csv\",\"order_by\":\"amount DESC\",\"limit\":5}}"
        : "{\"name\":\"data.query\",\"arguments\":{\"dataset\":\"sales.csv\",\"order_by\":\"duration DESC\",\"limit\":5}}";
}

constexpr concept_family k_families[] = {
    {"grouped-aggregation", "dataset_grouped_aggregation", "data.aggregate",
        "For a grouped numeric summary, identify the group field and use data.aggregate "
        "with the requested measure; do not answer with a descriptive statistics tool."},
    {"validated-filter", "dataset_validated_filter", "data.filter",
        "For a filtered table request, derive the predicate from the request and use "
        "data.filter; do not substitute sampling or descriptive statistics."},
    {"ordered-query", "dataset_ordered_query", "data.query",
        "For a request for records ordered by a measure, build a data.query with the "
        "ordering and limit; do not substitute value counts or a sample."},
};

constexpr concept_evaluation_fixture k_evaluation_fixtures[] = {
    {"grouped-seen", "grouped-aggregation", "seen",
        "Calculate total amount grouped by region.", "data.aggregate", "data.describe"},
    {"grouped-holdout", "grouped-aggregation", "holdout",
        "Count failed requests for each service.", "data.aggregate", "data.describe"},
    {"grouped-transfer", "grouped-aggregation", "transfer",
        "Average duration per endpoint.", "data.aggregate", "data.describe"},
    {"grouped-contrast", "grouped-aggregation", "contrastive",
        "Describe the status column.", "data.describe", "data.aggregate"},
    {"filter-seen", "validated-filter", "seen",
        "Return rows where status is failed.", "data.filter", "data.describe"},
    {"filter-holdout", "validated-filter", "holdout",
        "Keep records from the North region.", "data.filter", "data.describe"},
    {"filter-transfer", "validated-filter", "transfer",
        "Return services with latency above 100 ms.", "data.filter", "data.describe"},
    {"filter-contrast", "validated-filter", "contrastive",
        "How many rows have status failed?", "data.aggregate", "data.filter"},
    {"query-seen", "ordered-query", "seen",
        "Return the largest orders ordered by amount.", "data.query", "data.describe"},
    {"query-holdout", "ordered-query", "holdout",
        "Sort failures by timestamp.", "data.query", "data.describe"},
    {"query-transfer", "ordered-query", "transfer",
        "List the five longest requests.", "data.query", "data.describe"},
    {"query-contrast", "ordered-query", "contrastive",
        "Describe the status column.", "data.describe", "data.query"},
};

const concept_family * find_concept_family(const std::string & key) {
    for (const auto & family : k_families) {
        if (family.key == key) return &family;
    }
    return nullptr;
}

bool parse_args(int argc, char ** argv, options & value) {
    if (const char * model = std::getenv("LLAMA_AGENT_MODEL")) value.model = model;
    if (const char * threads = std::getenv("LLAMA_AGENT_THREADS")) value.n_threads = std::atoi(threads);
    for (int index = 1; index < argc; ++index) {
        const std::string arg = argv[index];
        auto next = [&](const char * name) -> const char * {
            if (index + 1 >= argc) {
                std::cerr << "missing value for " << name << '\n';
                return nullptr;
            }
            return argv[++index];
        };
        if (arg == "--model") {
            const char * path = next("--model");
            if (!path) return false;
            value.model = path;
        } else if (arg == "--n-predict") {
            const char * count = next("--n-predict");
            if (!count) return false;
            value.n_predict = std::atoi(count);
        } else if (arg == "--threads") {
            const char * count = next("--threads");
            if (!count) return false;
            value.n_threads = std::atoi(count);
        } else if (arg == "--n-gpu-layers") {
            const char * count = next("--n-gpu-layers");
            if (!count) return false;
            value.n_gpu_layers = std::atoi(count);
        } else if (arg == "--help" || arg == "-h") {
            return false;
        } else {
            std::cerr << "unknown argument: " << arg << '\n';
            return false;
        }
    }
    return true;
}

common_flydelta_concept_spec make_spec(
        const concept_family & family,
        const std::string & profile) {
    common_flydelta_concept_spec spec;
    spec.concept_key = std::string("flydelta://concept/") + family.key;
    spec.extraction_id = std::string("flydelta://extraction/") + family.key + "/smoke-v1";
    spec.behavior_key = family.behavior;
    spec.procedure_ref = std::string("procedure://") + family.key + "/v1";
    spec.verifier_ref = std::string("verifier://dataset-tool/") + family.expected_tool + "/v1";
    spec.model_profile_fingerprint = profile;
    spec.tokenizer_fingerprint = "sha256:concept-smoke-tokenizer";
    spec.template_fingerprint = "sha256:concept-smoke-template";
    spec.capture_layout_revision = "generation-boundary-layer-input:v1";
    spec.scope_fingerprint = "sha256:concept-smoke-local-scope";
    spec.host_approved = true;
    spec.redaction_attested = true;
    return spec;
}

std::vector<float> add(
        const std::vector<float> & left,
        const std::vector<float> & right) {
    std::vector<float> result(left.size(), 0.0f);
    for (size_t index = 0; index < result.size(); ++index) result[index] = left[index] + right[index];
    return result;
}

std::vector<common_flydelta_concept_trajectory> synthetic_trajectories(
        const concept_family & family,
        const std::string & profile,
        const std::vector<float> & concept) {
    const std::vector<std::vector<float>> baselines = {
        {0.10f, 0.20f, 0.30f, 0.40f},
        {0.20f, 0.10f, 0.40f, 0.30f},
        {0.30f, 0.40f, 0.10f, 0.20f},
    };
    const std::vector<std::vector<float>> nuisances = {
        {0.20f, -0.10f, 0.05f, 0.10f},
        {0.10f, 0.05f, -0.10f, 0.20f},
        {-0.05f, 0.10f, 0.20f, -0.10f},
    };
    std::vector<common_flydelta_concept_trajectory> result;
    for (size_t index = 0; index < baselines.size(); ++index) {
        const auto control = add(baselines[index], nuisances[index]);
        auto conditioned = add(control, concept);
        // Small, varied concept-conditioned examples keep the family from
        // being a literal duplicate while preserving a shared direction.
        conditioned[0] += static_cast<float>(index) * 0.01f;
        common_flydelta_concept_trajectory trajectory;
        trajectory.id = std::string("trajectory://") + family.key + "/" + std::to_string(index);
        trajectory.fixture_ref = std::string("fixture://") + family.key + "/" + std::to_string(index);
        trajectory.baseline_capture_ref = trajectory.fixture_ref + "/baseline";
        trajectory.conditioned_capture_ref = trajectory.fixture_ref + "/conditioned";
        trajectory.control_capture_ref = trajectory.fixture_ref + "/control";
        trajectory.semantic_anchor = "generation_boundary";
        trajectory.layer_index = 21;
        trajectory.baseline = baselines[index];
        trajectory.conditioned = conditioned;
        trajectory.control = control;
        trajectory.aligned = true;
        trajectory.conditioned_host_verified = true;
        result.push_back(std::move(trajectory));
    }
    (void) profile;
    return result;
}

bool validate_evaluation_manifest() {
    for (const auto & family : k_families) {
        size_t seen = 0;
        size_t holdout = 0;
        size_t transfer = 0;
        size_t contrastive = 0;
        for (const auto & fixture : k_evaluation_fixtures) {
            if (std::string(fixture.family) != family.key) continue;
            if (std::string(fixture.split) == "seen") ++seen;
            else if (std::string(fixture.split) == "holdout") ++holdout;
            else if (std::string(fixture.split) == "transfer") ++transfer;
            else if (std::string(fixture.split) == "contrastive") ++contrastive;
            else return false;
        }
        if (seen != 1 || holdout != 1 || transfer != 1 || contrastive != 1) return false;
        std::cout << "concept_evaluation_family=" << family.key
                  << " seen=1 holdout=1 transfer=1 contrastive=1\n";
    }
    for (size_t left = 0; left < std::size(k_evaluation_fixtures); ++left) {
        for (size_t right = left + 1; right < std::size(k_evaluation_fixtures); ++right) {
            if (std::string(k_evaluation_fixtures[left].id) ==
                    k_evaluation_fixtures[right].id) return false;
        }
    }
    return true;
}

const concept_evaluation_fixture * find_evaluation_fixture(const std::string & id) {
    for (const auto & fixture : k_evaluation_fixtures) {
        if (fixture.id == id) return &fixture;
    }
    return nullptr;
}

common_flydelta_experiment_fixture make_experiment_fixture(
        const concept_evaluation_fixture & source,
        const std::string & profile) {
    common_flydelta_experiment_fixture fixture;
    fixture.id = source.id;
    fixture.task_fingerprint = std::string("sha256:concept-task/") + source.id;
    fixture.model_profile_fingerprint = profile;
    fixture.tokenizer_fingerprint = "sha256:concept-smoke-tokenizer";
    fixture.template_fingerprint = "sha256:concept-smoke-template";
    fixture.execution_context_fingerprint = "sha256:concept-dataset-tools";
    fixture.verifier_revision = "concept-tool-routing:v1";
    return fixture;
}

bool make_experimental_activation(
        const common_flydelta_concept_candidate & candidate,
        const common_flydelta_direction_candidate & direction,
        size_t model_n_embd,
        size_t model_n_layers,
        float alpha,
        common_flydelta_activation_result & activation,
        std::string & error) {
    common_flydelta_gate_request gate_request;
    gate_request.explicit_opt_in = true;
    // This is smoke authority for a bounded experiment only. The candidate
    // remains experimental and never receives learning credit here.
    gate_request.candidate_status = common_flydelta_candidate_status::approved;
    gate_request.basis_available = true;
    gate_request.familiarity = 1.0f;
    gate_request.novelty = 0.0f;
    gate_request.requested_scale = alpha;

    common_flydelta_activation_request request;
    request.candidate_id = std::string("flydelta://concept-candidate/") + candidate.extraction_id;
    request.artifact_id = std::string("flydelta://concept-experimental/") + candidate.extraction_id;
    request.model_profile_fingerprint = candidate.model_profile_fingerprint;
    request.capture_layout_revision = candidate.capture_layout_revision;
    request.model_n_embd = model_n_embd;
    request.model_n_layers = model_n_layers;
    request.il_start = 1;
    request.il_end = static_cast<int32_t>(model_n_layers - 1);
    request.directions.push_back({direction.layer_index, direction.values});
    request.coefficients = {1.0f};
    request.gate_request = gate_request;

    common_flydelta_gate_config gate_config;
    gate_config.enabled = true;
    gate_config.max_scale = 1.0f;
    return common_flydelta_prepare_activation(
        gate_config, request, 64U * 1024U * 1024U, activation, error);
}

bool run_concept_model_evaluation(
        const options & value,
        llama_model * model,
        const common_chat_templates * templates,
        common_agent_inference & inference,
        uint32_t layer,
        const common_flydelta_concept_candidate & candidate,
        const common_flydelta_direction_candidate & direction,
        size_t model_n_embd,
        size_t model_n_layers,
        const std::string & family_key,
        const std::string & profile) {
    std::string error;
    common_flydelta_model_host model_host;
    model_host.capabilities.capture = true;
    model_host.capabilities.overlay = true;
    model_host.capabilities.generation = true;
    model_host.capabilities.teacher_forced_scoring = true;
    model_host.capabilities.host_verification = true;
    model_host.register_evaluator = [](
            common_flydelta_evaluator_config &, common_flydelta_evaluator_callbacks &,
            std::string &) { return true; };

    auto capture_request = std::make_shared<common_flydelta_hidden_state_capture_request>();
    capture_request->enabled = true;
    capture_request->layer_indices = {layer};
    capture_request->position = common_flydelta_capture_position::generation_boundary;
    capture_request->token_index = -1;
    capture_request->max_bytes = 4U * 1024U * 1024U;
    capture_request->model_profile_fingerprint = profile;
    capture_request->capture_layout_revision = candidate.capture_layout_revision;

    model_host.run_bounded_arm = [&](const common_flydelta_arm_request & arm_request,
            common_flydelta_arm_result & arm_result, std::string & runner_error) {
        const auto * source = find_evaluation_fixture(arm_request.fixture_ref);
        if (source == nullptr) {
            runner_error = "concept model host cannot resolve fixture reference";
            return false;
        }
        const auto * family = find_concept_family(family_key);
        if (family == nullptr || source->family != family_key) {
            runner_error = "concept model host received a fixture from another behavior family";
            return false;
        }
        const char * positive_tool = family->expected_tool;
        const char * negative_tool = std::string(source->expected_tool) != positive_tool
            ? source->expected_tool : source->decision_negative_tool;
        if (negative_tool == nullptr || std::string(negative_tool) == positive_tool) {
            runner_error = "concept model host could not resolve a distinct decision pair";
            return false;
        }
        std::shared_ptr<const common_flydelta_activation_result> activation;
        if (arm_request.apply_overlay) {
            common_flydelta_activation_result prepared;
            if (!make_experimental_activation(candidate, direction, model_n_embd,
                    model_n_layers, arm_request.alpha, prepared, runner_error)) return false;
            activation = std::make_shared<const common_flydelta_activation_result>(
                std::move(prepared));
        }

        auto generation_request = request(value, source->request, capture_request);
        generation_request.flydelta_activation = activation;
        common_agent_generation_result generated;
        if (!inference.generate(
                common_agent_generation_request(generation_request), generated) ||
                !common_agent_generation_succeeded(generated)) {
            runner_error = generated.error_message.empty()
                ? "concept model host generation failed" : generated.error_message;
            return false;
        }

        common_flydelta_decision_margin margin;
        if (!score_chat_choice_margin(
                model, templates, generation_request.messages, generation_request.tools,
                generation_request.tool_choice, generation_request.options,
                "{\"name\":\"", positive_tool, negative_tool, margin,
                nullptr, generation_request.json_schema, {}, {},
                activation ? activation->overlay : common_flydelta_static_overlay{},
                &runner_error)) return false;

        const std::string actual_tool = detected_tool(generated);
        const bool helped = actual_tool == source->expected_tool;
        arm_result = {};
        arm_result.arm_id = arm_request.arm_id;
        arm_result.executed = true;
        arm_result.requested_alpha = arm_request.alpha;
        arm_result.executed_alpha = arm_request.alpha;
        arm_result.margin = margin;
        arm_result.margin_available = margin.available;
        arm_result.generation_available = true;
        arm_result.host_evaluated = true;
        arm_result.verifier_known = true;
        arm_result.host_outcome = helped
            ? common_flydelta_counterfactual_outcome::helped
            : common_flydelta_counterfactual_outcome::unknown;
        arm_result.quality = helped ? 1.0f : 0.0f;
        arm_result.generation_ref = std::string("generation://concept/") + source->id;
        arm_result.provenance_ref = std::string("evidence:concept/") + source->split;
        return true;
    };

    if (!common_flydelta_model_host_validate(model_host, error)) {
        std::cerr << "concept model host validation failed: " << error << '\n';
        return false;
    }

    const auto runner = common_flydelta_search_pipeline_runner_from_model_host(
        model_host, std::string("flydelta://job/concept/") + candidate.extraction_id,
        "context://concept-model", "intervention://concept/" + candidate.extraction_id,
        true, true, true, true, 4U * 1024U * 1024U, static_cast<size_t>(value.n_predict));

    const auto * concept_family = find_concept_family(family_key);
    if (concept_family == nullptr) {
        std::cerr << "concept model evaluation received an unknown behavior family\n";
        return false;
    }

    size_t evaluated = 0;
    size_t helped = 0;
    for (const auto & source : k_evaluation_fixtures) {
        if (source.family != family_key) continue;
        const auto fixture = make_experiment_fixture(source, profile);
        common_flydelta_direction_candidate pipeline_direction = direction;
        common_flydelta_layer_candidate layer_candidate;
        layer_candidate.layer_indices = {layer};
        layer_candidate.anchor_layer_index = layer;
        layer_candidate.total_scale = 0.02f;
        layer_candidate.per_layer_scale = 0.02f;
        layer_candidate.source = common_flydelta_layer_search_candidate_source::diagnostic_singleton;
        common_flydelta_counterfactual_trial baseline_trial;
        common_flydelta_counterfactual_trial candidate_trial;
        common_flydelta_decision_margin baseline_margin;
        common_flydelta_decision_margin candidate_margin;
        common_flydelta_scale_geometry baseline_geometry;
        common_flydelta_scale_geometry candidate_geometry;
        if (!runner(fixture, pipeline_direction, nullptr, 0.0f, false,
                baseline_trial, baseline_margin, baseline_geometry, error) ||
                !runner(fixture, pipeline_direction, &layer_candidate, 0.02f, true,
                candidate_trial, candidate_margin, candidate_geometry, error)) {
            std::cerr << "concept model evaluation failed fixture=" << source.id
                      << ": " << error << '\n';
            return false;
        }
        const float margin_delta = candidate_margin.available && baseline_margin.available
            ? candidate_margin.total_delta() - baseline_margin.total_delta() : 0.0f;
        const auto outcome = common_flydelta_classify_counterfactual(
            baseline_trial, candidate_trial);
        const bool candidate_helped = outcome == common_flydelta_counterfactual_outcome::helped;
        const char * negative_tool = std::string(source.expected_tool) !=
                concept_family->expected_tool
            ? source.expected_tool : source.decision_negative_tool;
        ++evaluated;
        if (candidate_helped) ++helped;
        std::cout << "concept_model_evaluation fixture=" << source.id
                  << " split=" << source.split
                  << " expected_tool=" << source.expected_tool
                  << " decision_pair=" << concept_family->expected_tool
                  << "/" << negative_tool
                  << " outcome=" << common_flydelta_counterfactual_outcome_name(outcome)
                  << " margin_available=" << (candidate_margin.available ? "yes" : "no")
                  << " margin_delta_total=" << margin_delta
                  << " experimental_only=yes learning_credit=no\n";
    }
    std::cout << "concept_model_candidate_evaluation candidate_kind="
              << common_flydelta_concept_candidate_kind_name(candidate.kind)
              << " evaluated=" << evaluated
              << " helped=" << helped
              << " learning_credit=no\n";
    return true;
}

bool run_offline() {
    std::string error;
    if (!validate_evaluation_manifest()) {
        std::cerr << "concept evaluation fixture manifest is invalid\n";
        return false;
    }
    const std::string profile = "sha256:concept-offline-model";
    size_t family_count = 0;
    for (const auto & family : k_families) {
        const auto spec = make_spec(family, profile);
        common_flydelta_concept_build_config config;
        config.dimension = 4;
        std::vector<common_flydelta_concept_candidate> candidates;
        if (!common_flydelta_build_concept_candidates(
                spec, config, synthetic_trajectories(
                    family, profile,
                    family_count == 0 ? std::vector<float>{1.0f, 0.5f, 0.2f, 0.1f} :
                    family_count == 1 ? std::vector<float>{0.1f, 1.0f, 0.4f, 0.2f} :
                    std::vector<float>{0.2f, 0.1f, 1.0f, 0.5f}), candidates, error) ||
                candidates.size() != 3) {
            std::cerr << "concept offline build failed: " << error << '\n';
            return false;
        }
        for (const auto & candidate : candidates) {
            common_flydelta_direction_candidate direction;
            if (!common_flydelta_concept_candidate_to_direction(
                    candidate, direction, error) || !direction.experimental_only ||
                    direction.origin != "host_taught_extracted") {
                std::cerr << "concept offline direction admission failed: " << error << '\n';
                return false;
            }
        }
        std::cout << "concept_family=" << family.key
                  << " candidates=" << candidates.size()
                  << " source_trajectories=" << candidates.front().source_trajectories
                  << " retained_trajectories=" << candidates.front().retained_trajectories
                  << " median_alignment=" << candidates.front().median_alignment
                  << " experimental_only=yes learning_eligible=no\n";
        ++family_count;
    }

    auto invalid = synthetic_trajectories(k_families[0], profile, {1.0f, 0.5f, 0.2f, 0.1f});
    invalid.front().conditioned_host_verified = false;
    common_flydelta_concept_build_config config;
    config.dimension = 4;
    std::vector<common_flydelta_concept_candidate> ignored;
    if (common_flydelta_build_concept_candidates(
            make_spec(k_families[0], profile), config, invalid, ignored, error)) {
        std::cerr << "concept contract accepted an unverified trajectory\n";
        return false;
    }
    std::cout << "concept_contract_unverified_rejected=yes\n"
              << "flydelta_concept_smoke=passed mode=offline families=" << family_count << '\n';
    return true;
}

const char * detected_tool(const common_agent_generation_result & result) {
    if (!common_agent_generation_succeeded(result)) return "generation_failure";
    constexpr const char * tools[] = {
        "data.aggregate", "data.filter", "data.query", "data.describe", "data.inspect"};
    for (const char * tool : tools) {
        if (result.content.find(std::string("\"name\":\"") + tool) != std::string::npos ||
                result.content.find(std::string("\"name\": \"") + tool) != std::string::npos) {
            return tool;
        }
    }
    return "none";
}

const char * conditioned_status(
        const common_agent_generation_result & baseline,
        const common_agent_generation_result & conditioned,
        const concept_family & family,
        size_t index) {
    const char * conditioned_name = detected_tool(conditioned);
    if (conditioned.content.find(canonical_conditioned_call(family, index)) != std::string::npos) {
        return "VERIFIED_POSITIVE";
    }
    if (std::string(conditioned_name) == family.expected_tool) return "NON_CANONICAL_POSITIVE";
    if (std::string(conditioned_name) == "generation_failure") return "GENERATION_FAILURE";
    if (std::string(conditioned_name) == "none") return "NO_POSITIVE";
    if (std::string(conditioned_name) == detected_tool(baseline)) return "SAME_DECISION";
    return "HOST_REJECTED_POSITIVE";
}

common_agent_generation_request request(
        const options & value,
        const std::string & instruction,
        const std::shared_ptr<const common_flydelta_hidden_state_capture_request> & capture) {
    common_agent_generation_request result;
    result.purpose = common_agent_generation_purpose::tool_followup;
    result.options.n_predict = value.n_predict;
    result.options.n_threads = value.n_threads;
    result.messages = {
        {"system", "You route dataset requests. Available tools are data.aggregate, data.filter, "
            "data.query, data.describe, and data.inspect. Return exactly one JSON tool object "
            "with a name field and no markdown."},
        {"user", instruction},
    };
    result.flydelta_capture = capture;
    return result;
}

bool slice_capture(
        const common_flydelta_hidden_state_capture & capture,
        uint32_t layer,
        std::vector<float> & values) {
    const auto found = std::find(capture.layer_indices.begin(), capture.layer_indices.end(), layer);
    if (found == capture.layer_indices.end() || capture.n_embd == 0) return false;
    const size_t offset = static_cast<size_t>(found - capture.layer_indices.begin()) * capture.n_embd;
    if (offset + capture.n_embd > capture.values.size()) return false;
    values.assign(capture.values.begin() + offset, capture.values.begin() + offset + capture.n_embd);
    return true;
}

bool run_model(const options & value) {
    if (value.model.empty() || !std::filesystem::is_regular_file(value.model)) {
        std::cerr << "FlyDelta concept model smoke skipped: provide --model or LLAMA_AGENT_MODEL\n";
        return true;
    }
    if (value.n_threads <= 0 || value.n_threads > 3) {
        std::cerr << "concept model smoke requires threads in range 1..3\n";
        return false;
    }
    common_agent_model_selection selection;
    selection.profile_id = "flydelta-host-taught-concept";
    selection.base_model_id = "generation-base";
    selection.backend = "cli";
    selection.path = value.model;
    selection.context_size_tokens = 2048;
    selection.load_policy = "resident";
    common_agent_runtime_cli_model_loader loader({value.n_gpu_layers, value.n_threads, true});
    std::shared_ptr<common_agent_runtime_resident_model> resident;
    std::string error;
    if (!loader.load(selection, resident, error)) {
        std::cerr << "concept model smoke could not load model: " << error << '\n';
        return false;
    }
    const auto loaded = common_agent_runtime_loaded_model_cast(resident);
    if (!loaded || !loaded->model || !loaded->chat_templates) {
        std::cerr << "concept model smoke received incomplete model\n";
        return false;
    }
    auto inference = make_llama_cli_agent_inference(loaded->model, loaded->chat_templates.get());
    const uint32_t layer = static_cast<uint32_t>(std::max<int>(1,
        std::min<int>(static_cast<int>(llama_model_n_layer(loaded->model)) - 1, 21)));
    const uint32_t layers[] = {layer};
    auto capture = std::make_shared<common_flydelta_hidden_state_capture_request>();
    capture->enabled = true;
    capture->layer_indices.assign(std::begin(layers), std::end(layers));
    capture->position = common_flydelta_capture_position::generation_boundary;
    capture->token_index = -1;
    capture->max_bytes = 4U * 1024U * 1024U;
    capture->model_profile_fingerprint = "sha256:concept-qwen-model";
    capture->capture_layout_revision = "generation-boundary-layer-input:v1";

    size_t resolved_families = 0;
    size_t captured_pairs = 0;
    for (const auto & family : k_families) {
        std::vector<common_flydelta_concept_trajectory> trajectories;
        for (size_t index = 0; index < 2; ++index) {
            const std::string request_text = extraction_request(family, index);
            const std::string baseline_text =
                "Choose the correct dataset operation for this request: " + request_text;
            const std::string conditioned_text =
                std::string(family.procedure) + " Request: " + request_text +
                " Return exactly this canonical JSON object and no markdown or explanation: " +
                canonical_conditioned_call(family, index);
            const std::string control_text =
                "Choose the correct dataset operation for this request. Be concise and return JSON only. Request: " +
                request_text;
            common_agent_generation_result baseline;
            common_agent_generation_result conditioned;
            common_agent_generation_result control;
            const bool baseline_ok = inference->generate(
                request(value, baseline_text, capture), baseline);
            const bool conditioned_ok = inference->generate(
                request(value, conditioned_text, capture), conditioned);
            const bool control_ok = inference->generate(
                request(value, control_text, capture), control);
            if (!baseline_ok || !conditioned_ok || !control_ok ||
                    !common_agent_generation_succeeded(baseline) ||
                    !common_agent_generation_succeeded(conditioned) ||
                    !common_agent_generation_succeeded(control)) {
                std::cout << "concept_model_trajectory family=" << family.key
                          << " index=" << index
                          << " status=" << conditioned_status(baseline, conditioned, family, index)
                          << " baseline_tool=" << detected_tool(baseline)
                          << " conditioned_tool=" << detected_tool(conditioned)
                          << " control_tool=" << detected_tool(control)
                          << " conditioned_preview=\"" << generation_preview(conditioned) << "\""
                          << " captures=no\n";
                continue;
            }
            if (!baseline.flydelta_capture || !conditioned.flydelta_capture ||
                    !control.flydelta_capture || !conditioned.flydelta_capture->captured ||
                    !control.flydelta_capture->captured || !baseline.flydelta_capture->captured) {
                std::cout << "concept_model_trajectory family=" << family.key
                          << " index=" << index
                          << " status=CAPTURE_FAILURE"
                          << " baseline_tool=" << detected_tool(baseline)
                          << " conditioned_tool=" << detected_tool(conditioned)
                          << " control_tool=" << detected_tool(control)
                          << " conditioned_preview=\"" << generation_preview(conditioned) << "\""
                          << " captures=no\n";
                continue;
            }
            std::vector<float> baseline_values;
            std::vector<float> conditioned_values;
            std::vector<float> control_values;
            if (!slice_capture(*baseline.flydelta_capture, layer, baseline_values) ||
                    !slice_capture(*conditioned.flydelta_capture, layer, conditioned_values) ||
                    !slice_capture(*control.flydelta_capture, layer, control_values)) continue;
            common_flydelta_concept_trajectory trajectory;
            trajectory.id = std::string("model-trajectory://") + family.key + "/" + std::to_string(index);
            trajectory.fixture_ref = std::string("model-fixture://") + family.key + "/" + std::to_string(index);
            trajectory.baseline_capture_ref = trajectory.fixture_ref + "/baseline";
            trajectory.conditioned_capture_ref = trajectory.fixture_ref + "/conditioned";
            trajectory.control_capture_ref = trajectory.fixture_ref + "/control";
            trajectory.semantic_anchor = "generation_boundary";
            trajectory.layer_index = static_cast<int32_t>(layer);
            trajectory.baseline = std::move(baseline_values);
            trajectory.conditioned = std::move(conditioned_values);
            trajectory.control = std::move(control_values);
            trajectory.aligned = true;
            trajectory.conditioned_host_verified =
                std::string(conditioned_status(baseline, conditioned, family, index)) ==
                "VERIFIED_POSITIVE";
            std::cout << "concept_model_trajectory family=" << family.key
                      << " index=" << index
                      << " status=" << conditioned_status(baseline, conditioned, family, index)
                      << " baseline_tool=" << detected_tool(baseline)
                      << " conditioned_tool=" << detected_tool(conditioned)
                      << " control_tool=" << detected_tool(control)
                      << " conditioned_preview=\"" << generation_preview(conditioned) << "\""
                      << " conditioned_host_verified="
                      << (trajectory.conditioned_host_verified ? "yes" : "no")
                      << " captures=yes\n";
            if (trajectory.conditioned_host_verified) {
                trajectories.push_back(std::move(trajectory));
                ++captured_pairs;
            }
        }
        if (trajectories.size() < 2) {
            std::cout << "concept_model_family=" << family.key
                      << " status=unresolved reason=insufficient_host_verified_conditioned_pairs\n";
            continue;
        }
        common_flydelta_concept_build_config config;
        config.dimension = trajectories.front().baseline.size();
        std::vector<common_flydelta_concept_candidate> candidates;
        if (!common_flydelta_build_concept_candidates(
                make_spec(family, "sha256:concept-qwen-model"), config,
                trajectories, candidates, error)) {
            std::cerr << "concept model build failed for " << family.key << ": " << error << '\n';
            return false;
        }
        for (const auto & candidate : candidates) {
            common_flydelta_direction_candidate direction;
            if (!common_flydelta_concept_candidate_to_direction(
                    candidate, direction, error) || !direction.experimental_only ||
                    direction.origin != "host_taught_extracted") {
                std::cerr << "concept model direction admission failed: " << error << '\n';
                return false;
            }
        }
        std::cout << "concept_model_family=" << family.key
                  << " status=extracted candidates=" << candidates.size()
                  << " layer=" << candidates.front().layer_index
                  << " median_alignment=" << candidates.front().median_alignment
                  << " promotion=no\n";
        const auto candidate_it = std::find_if(candidates.begin(), candidates.end(),
            [](const common_flydelta_concept_candidate & candidate) {
                return candidate.kind == common_flydelta_concept_candidate_kind::trimmed_mean;
            });
        if (candidate_it == candidates.end()) {
            std::cerr << "concept model smoke did not produce a trimmed mean candidate\n";
            return false;
        }
        common_flydelta_direction_candidate evaluation_direction;
        if (!common_flydelta_concept_candidate_to_direction(
                *candidate_it, evaluation_direction, error) ||
                !run_concept_model_evaluation(
                    value, loaded->model, loaded->chat_templates.get(), *inference,
                    layer, *candidate_it, evaluation_direction,
                    static_cast<size_t>(llama_model_n_embd(loaded->model)),
                    static_cast<size_t>(llama_model_n_layer(loaded->model)),
                    family.key,
                    "sha256:concept-qwen-model")) {
            std::cerr << "concept model candidate evaluation failed for " << family.key
                      << ": " << error << '\n';
            return false;
        }
        ++resolved_families;
    }
    std::cout << "flydelta_concept_model_smoke=completed"
              << " families=" << std::size(k_families)
              << " resolved_families=" << resolved_families
              << " captured_pairs=" << captured_pairs
              << " learning_credit=no\n";
    return true;
}

} // namespace

int main(int argc, char ** argv) {
    options value;
    if (!parse_args(argc, argv, value)) {
        std::cerr << "usage: " << argv[0]
                  << " [--model MODEL] [--n-predict N] [--threads N] [--n-gpu-layers N]\n";
        return 2;
    }
    if (!run_offline()) return 1;
    if (!run_model(value)) return 1;
    return 0;
}
