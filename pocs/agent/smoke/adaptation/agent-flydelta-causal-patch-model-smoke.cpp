#include "agent/agent-residual-patch.h"
#include "agent/tooling/catalog/tool-catalog.h"
#include "agent/tooling/schema/tool-schema-compact.h"
#include "agent/adaptation/flydelta/flydelta-hidden-state-hook.h"
#include "tools/agent/openapi/agent-openapi-catalog.h"
#include "tools/agent/runtime/agent-runtime-host.h"
#include "tools/agent/runtime/agent-runtime-assembly.h"
#include "tools/agent/runtime/agent-server-context-host.h"
#include "memory/memory-in-memory.h"
#include "plan/plan-in-memory.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace {
using json = nlohmann::ordered_json;
constexpr const char * kDataset = "dataset://local/sales";

struct causal_fixture {
    const char * id;
    const char * expected_tool;
    const char * failed_tool;
    const char * expected_arguments;
    const char * failed_arguments;
    const char * task;
    bool openapi = false;
    const char * repair_task = nullptr;
};

// The schema case is the primary candidate: the same field-oriented question
// that previously produced dataset.inspect is host-correctly admitted as a
// schema request.  It can therefore produce a natural
// dataset.inspect -> dataset.schema repair without contradictory prompting.
constexpr causal_fixture kSchemaFixture = {
    "schema-selection",
    "dataset.schema",
    "dataset.inspect",
    R"({"dataset":"dataset://local/sales"})",
    R"({"dataset":"dataset://local/sales"})",
    "For dataset://local/sales, describe the available fields and their types before analysis.",
};

// Keep the earlier experiment selectable so its historical result remains
// reproducible, but do not use it as the default counterfactual case.
constexpr causal_fixture kOverviewFixture = {
    "overview-selection",
    "dataset.inspect",
    "statistics.describe",
    R"({"dataset":"dataset://local/sales"})",
    R"({"dataset":"dataset://local/sales"})",
    "For dataset://local/sales, determine which fields are available before analysis.",
};

// Smoke-local counterfactual material.  The baseline and repair prompts are
// deliberately separate so the causal matrix can be exercised even when the
// model solves the ordinary production wording on its first planner pass.
// This does not change production prompting or admission semantics.
constexpr causal_fixture kExplicitRepairPairFixture = {
    "explicit-repair-pair",
    "dataset.schema",
    "dataset.inspect",
    R"({"dataset":"dataset://local/sales"})",
    R"({"dataset":"dataset://local/sales"})",
    "Inspect dataset://local/sales and return the available dataset fields.",
    false,
    "For dataset://local/sales, return its schema with each field and type before analysis.",
};

// Observed in the production-like Qwen/OpenAlex planner run: the registered
// collection operation was emitted without its required search argument and
// the bounded planner regeneration was responsible for repairing it. This is
// deliberately an argument-level counterfactual: it uses one canonical
// operation and distinguishes the rejected empty argument object from the
// host-valid repaired object.
constexpr causal_fixture kOpenAlexArgumentRepairFixture = {
    "openalex-listworks-argument-repair",
    "openalex.listWorks",
    "openalex.listWorks",
    R"({"search":"machine learning","per_page":1,"select":"id,display_name"})",
    R"({})",
    "Use only the openalex.listWorks tool. Call it with the search argument set to machine learning, per_page set to 1, and select set to id,display_name. After the tool succeeds, answer with the first work id and display name. Do not invent a result.",
    true,
};

constexpr causal_fixture kOpenAlexKnownIdFixture = {
    "openalex-known-id",
    "openalex.getWork",
    "openalex.listWorks",
    R"({"id":"W2101234009"})",
    R"({"search":"W2101234009","per_page":1})",
    "Retrieve the scholarly work whose OpenAlex id is W2101234009.",
    true,
};

struct options {
    std::string model;
    std::string fixture = kSchemaFixture.id;
    std::string openalex_spec;
    int threads = 3;
    int gpu_layers = 99;
    int n_predict = 32;
    bool capture_only = false;
    bool full_matrix = false;
    std::vector<uint32_t> layers = {21};
};

bool parse_options(int argc, char ** argv, options & out) {
    if (const char * value = std::getenv("LLAMA_AGENT_MODEL")) out.model = value;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&]() -> const char * { return i + 1 < argc ? argv[++i] : nullptr; };
        if (arg == "--model") { const char * v = next(); if (!v) return false; out.model = v; }
        else if (arg == "--fixture") { const char * v = next(); if (!v) return false; out.fixture = v; }
        else if (arg == "--openalex-spec") { const char * v = next(); if (!v) return false; out.openalex_spec = v; }
        else if (arg == "--threads") { const char * v = next(); if (!v) return false; out.threads = std::stoi(v); }
        else if (arg == "--n-gpu-layers") { const char * v = next(); if (!v) return false; out.gpu_layers = std::stoi(v); }
        else if (arg == "--n-predict") { const char * v = next(); if (!v) return false; out.n_predict = std::stoi(v); }
        else if (arg == "--capture-only") out.capture_only = true;
        else if (arg == "--full-matrix") out.full_matrix = true;
        else if (arg == "--all-layers") out.layers = {20, 21, 22};
        else if (arg == "--layer") {
            const char * v = next(); if (!v) return false;
            const int layer = std::stoi(v);
            if (layer < 0) return false;
            out.layers = {static_cast<uint32_t>(layer)};
        }
        else if (arg == "--help" || arg == "-h") return false;
        else return false;
    }
    return out.threads > 0 && out.threads <= 4 && out.n_predict > 0;
}

const causal_fixture * fixture_for(const options & value) {
    if (value.fixture == kSchemaFixture.id) return &kSchemaFixture;
    if (value.fixture == kOverviewFixture.id) return &kOverviewFixture;
    if (value.fixture == kExplicitRepairPairFixture.id) return &kExplicitRepairPairFixture;
    if (value.fixture == kOpenAlexArgumentRepairFixture.id) return &kOpenAlexArgumentRepairFixture;
    if (value.fixture == kOpenAlexKnownIdFixture.id) return &kOpenAlexKnownIdFixture;
    return nullptr;
}

std::string selected_tool(const common_agent_generation_result & result) {
    if (!common_agent_generation_succeeded(result)) return {};
    if (result.chat_params) {
        common_chat_parser_params params(*result.chat_params);
        params.parse_tool_calls = true;
        if (!result.chat_params->parser.empty()) params.parser.load(result.chat_params->parser);
        const auto parsed = common_chat_parse(result.content, false, params);
        if (!parsed.tool_calls.empty()) return parsed.tool_calls.front().name;
    }
    const auto parsed = json::parse(result.content, nullptr, false);
    return parsed.is_object() && parsed.contains("name") && parsed["name"].is_string()
        ? parsed["name"].get<std::string>() : std::string{};
}

std::string selected_arguments(const common_agent_generation_result & result) {
    if (!common_agent_generation_succeeded(result)) return {};
    if (result.chat_params) {
        common_chat_parser_params params(*result.chat_params);
        params.parse_tool_calls = true;
        if (!result.chat_params->parser.empty()) params.parser.load(result.chat_params->parser);
        const auto parsed = common_chat_parse(result.content, false, params);
        if (!parsed.tool_calls.empty()) return parsed.tool_calls.front().arguments;
    }
    const auto parsed = json::parse(result.content, nullptr, false);
    return parsed.is_object() && parsed.contains("arguments") && parsed["arguments"].is_object()
        ? parsed["arguments"].dump() : std::string{};
}

bool same_json(const std::string & lhs, const std::string & rhs) {
    const auto left = json::parse(lhs, nullptr, false);
    const auto right = json::parse(rhs, nullptr, false);
    return !left.is_discarded() && !right.is_discarded() && left == right;
}

std::string tool_continuation(const char * tool, const char * arguments) {
    // choice_prefix is {"name":".  Keep the continuation in the same
    // model-facing JSON form as the production FlyDelta scorer so the
    // positive and negative paths really encode different tool choices.
    return std::string(tool) + "\",\"arguments\":" + arguments + "}";
}

bool score_margin(common_agent_inference & inference,
        const causal_fixture & fixture,
        const common_agent_generation_request & base_context,
        const std::shared_ptr<const common_agent_residual_patch_request> & patch,
        common_agent_teacher_forced_choice_result & result) {
    common_agent_teacher_forced_choice_request score;
    score.sequence_id = patch && !patch->identity.empty() ? patch->identity : "baseline";
    // Teacher-forced scoring must use the same model-facing planner surface
    // as the captured causal state.  Rebuilding a shorter prompt here would
    // make the capture's absolute token position invalid and would compare
    // a different contract/context than the observed planner decision.
    score.context = base_context;
    score.context.flydelta_capture.reset();
    score.context.residual_patch = patch;
    score.choice_prefix = "{\"name\":\"";
    score.positive_choice = fixture.expected_tool;
    score.negative_choice = fixture.failed_tool;
    score.positive_continuation = tool_continuation(fixture.expected_tool, fixture.expected_arguments);
    score.negative_continuation = tool_continuation(fixture.failed_tool, fixture.failed_arguments);
    return inference.score_teacher_forced_choice(score, result) && result.available;
}

const char * outcome_name(bool baseline_passed, bool candidate_passed) {
    if (!baseline_passed && candidate_passed) return "HELPED";
    if (baseline_passed && !candidate_passed) return "HARMED";
    if (baseline_passed == candidate_passed) return "NEUTRAL";
    return "UNKNOWN";
}

std::shared_ptr<const common_flydelta_hidden_state_capture_request> capture_request(
        const std::string & profile, const std::vector<uint32_t> & layers) {
    auto result = std::make_shared<common_flydelta_hidden_state_capture_request>();
    result->enabled = true;
    result->layer_indices = layers;
    result->token_index = -1;
    result->position = common_flydelta_capture_position::generation_boundary;
    result->max_bytes = 4U * 1024U * 1024U;
    result->model_profile_fingerprint = profile;
    result->capture_layout_revision = "layer-input:generation-boundary:v1";
    return result;
}

bool capture_layer(const common_flydelta_hidden_state_capture & capture,
        uint32_t layer, std::vector<float> & values) {
    const auto it = std::find(capture.layer_indices.begin(), capture.layer_indices.end(), layer);
    if (it == capture.layer_indices.end() || !capture.captured || capture.n_embd == 0) return false;
    const size_t index = static_cast<size_t>(it - capture.layer_indices.begin());
    const size_t offset = index * capture.n_embd;
    if (offset + capture.n_embd > capture.values.size()) return false;
    values.assign(capture.values.begin() + static_cast<std::ptrdiff_t>(offset),
        capture.values.begin() + static_cast<std::ptrdiff_t>(offset + capture.n_embd));
    return true;
}

float norm(const std::vector<float> & values) {
    double sum = 0.0; for (float value : values) sum += double(value) * value;
    return static_cast<float>(std::sqrt(sum));
}

bool finite_nonzero_capture(const std::vector<float> & values) {
    return !values.empty() && std::all_of(values.begin(), values.end(), [](float value) {
        return std::isfinite(value);
    }) && norm(values) > 1.0e-6f;
}

bool load_openalex_tools(
        const options & value, const causal_fixture & fixture,
        std::vector<common_chat_tool> & tools, std::string & error) {
    if (value.openalex_spec.empty() || !std::filesystem::is_regular_file(value.openalex_spec)) {
        error = "openalex fixture requires --openalex-spec PATH";
        return false;
    }
    std::ifstream input(value.openalex_spec);
    json document;
    input >> document;
    agent_host_openapi_provider_config config;
    config.id = "openalex";
    config.prefix = "openalex";
    config.access = "read_only";
    config.exposure = "auto";
    agent_openapi_catalog catalog;
    if (!build_agent_openapi_catalog(document, config, catalog, error)) return false;
    for (const char * name : {fixture.expected_tool, fixture.failed_tool}) {
        const std::string full_name(name);
        const bool already_loaded = std::any_of(tools.begin(), tools.end(),
            [&](const common_chat_tool & tool) { return tool.name == full_name; });
        if (already_loaded) continue;
        const std::string prefix = catalog.prefix + ".";
        if (full_name.rfind(prefix, 0) != 0) { error = "OpenAlex fixture tool is outside catalog prefix"; return false; }
        const auto it = std::find_if(catalog.operations.begin(), catalog.operations.end(),
            [&](const agent_openapi_operation & operation) {
                return operation.operation_id == full_name.substr(prefix.size());
            });
        if (it == catalog.operations.end()) { error = "OpenAlex contract is missing " + full_name; return false; }
        tools.push_back({full_name, it->description.empty() ? it->summary : it->description,
            it->input_schema_json, it->result_schema_json});
    }
    return true;
}

std::vector<float> random_control(const std::vector<float> & direction, uint32_t layer) {
    std::vector<float> result(direction.size());
    uint32_t state = 0x9e3779b9U ^ (layer * 0x85ebca6bU);
    for (float & value : result) {
        state = state * 1664525U + 1013904223U;
        value = static_cast<float>((state & 0xffffU) / 32768.0 - 1.0);
    }
    const float projection_denominator = std::max(1.0e-12f, norm(direction) * norm(direction));
    double projection = 0.0;
    for (size_t i = 0; i < result.size(); ++i) projection += result[i] * direction[i];
    const float scale = static_cast<float>(projection / projection_denominator);
    for (size_t i = 0; i < result.size(); ++i) result[i] -= scale * direction[i];
    const float target = norm(direction);
    const float actual = std::max(1.0e-12f, norm(result));
    for (float & value : result) value *= target / actual;
    return result;
}

struct planner_generation_observation {
    common_agent_generation_request request;
    common_agent_generation_result result;
};

// The planner is production code. This decorator is smoke-local only: it
// records the generation requests/results that the planner actually sent and
// injects one experiment patch into those requests without changing planner,
// runtime, lifecycle, or promotion contracts.
class causal_observing_inference final : public common_agent_inference {
public:
    explicit causal_observing_inference(common_agent_inference & delegate,
            std::shared_ptr<const common_agent_residual_patch_request> patch = {})
        : delegate(delegate), patch(std::move(patch)) {}

    bool generate(const common_agent_generation_request & request,
            common_agent_generation_result & result) override {
        common_agent_generation_request observed = request;
        // The patch is a planner-local causal intervention.  In the full
        // runtime path family preflight and later execution generations are
        // separate model-facing requests and must not silently inherit a
        // planner capture's absolute layer/position.
        if (patch && request.purpose == common_agent_generation_purpose::planner) {
            observed.residual_patch = patch;
        }
        const bool ok = delegate.generate(observed, result);
        observations.push_back({observed, result});
        return ok;
    }

    bool score_teacher_forced_choice(
            const common_agent_teacher_forced_choice_request & request,
            common_agent_teacher_forced_choice_result & result) override {
        auto observed = request;
        if (patch) observed.context.residual_patch = patch;
        return delegate.score_teacher_forced_choice(observed, result);
    }

    bool score_teacher_forced_choice_batch(
            const common_agent_teacher_forced_choice_batch_request & request,
            common_agent_teacher_forced_choice_batch_result & result) override {
        auto observed = request;
        if (patch) {
            for (auto & choice : observed.choices) choice.context.residual_patch = patch;
        }
        return delegate.score_teacher_forced_choice_batch(observed, result);
    }

    common_agent_inference & delegate;
    std::shared_ptr<const common_agent_residual_patch_request> patch;
    std::vector<planner_generation_observation> observations;
};

std::string planner_tool(const common_agent_generation_result & result) {
    if (!common_agent_generation_succeeded(result)) return {};
    const auto parsed = json::parse(result.content, nullptr, false);
    if (parsed.is_discarded() || !parsed.is_object() || !parsed.contains("steps") ||
            !parsed["steps"].is_array()) return {};
    for (const auto & step : parsed["steps"]) {
        if (step.is_object() && step.contains("tool") && step["tool"].is_string()) {
            return step["tool"].get<std::string>();
        }
    }
    return {};
}

std::string planner_arguments(const common_agent_generation_result & result) {
    if (!common_agent_generation_succeeded(result)) return {};
    const auto parsed = json::parse(result.content, nullptr, false);
    if (parsed.is_discarded() || !parsed.is_object() || !parsed.contains("steps") ||
            !parsed["steps"].is_array()) return {};
    for (const auto & step : parsed["steps"]) {
        if (step.is_object() && step.contains("tool") && step["tool"].is_string() &&
                step.contains("args") && step["args"].is_object()) {
            return step["args"].dump();
        }
    }
    return {};
}

struct planner_counterfactual {
    planner_generation_observation baseline;
    planner_generation_observation repair;
    common_agent_generation_request diagnostic_context;
};

bool run_planner_observations(
        common_agent_inference & delegate,
        const options & value,
        const causal_fixture & fixture,
        const std::vector<common_chat_tool> & tools,
        const std::shared_ptr<const common_flydelta_hidden_state_capture_request> & capture,
        const std::shared_ptr<const common_agent_residual_patch_request> & patch,
        std::vector<planner_generation_observation> & observations,
        std::string & error) {
    causal_observing_inference observing(delegate, patch);
    common_agent_generation_config generation_config;
    generation_config.n_predict = std::max(value.n_predict, 512);
    generation_config.n_threads = value.threads;
    generation_config.generation_trace = true;
    generation_config.agent_trace = true;
    generation_config.context_size_tokens = 2048;

    common_memory_in_memory_store memories;
    common_plan_in_memory_store plans;
    if (!memories.open("", error) || !plans.open("", error)) {
        return false;
    }
    common_agent_runtime_turn_request turn_request;
    turn_request.request.session_id = "flydelta-causal-planner";
    turn_request.request.namespace_id = "local";
    turn_request.request.turn_id = "flydelta-causal-" + std::string(fixture.id);
    turn_request.request.prompt = fixture.task;
    turn_request.request.require_tool_execution = true;
    turn_request.request.enable_planning = true;
    turn_request.request.enable_reflection = true;
    turn_request.request.max_iterations = 2;
    turn_request.request.max_reflection_rounds = 1;
    turn_request.request.max_tool_batches = 1;
    turn_request.request.flydelta_capture = capture;
    turn_request.scope.session_id = turn_request.request.session_id;
    turn_request.scope.namespace_id = turn_request.request.namespace_id;
    turn_request.scope.turn_id = turn_request.request.turn_id;
    turn_request.policy.agent_inference_backend = "server-context";
    turn_request.policy.enable_reflection = true;
    turn_request.policy.max_iterations = 2;
    turn_request.policy.max_reflection_rounds = 1;
    turn_request.policy.max_tool_rounds = 1;
    turn_request.runtime_config.generation_config = generation_config;
    turn_request.orchestration_config = make_agent_orchestration_config({
        fixture.task, "auto", "auto", "none", {}, {}});
    turn_request.generation_options.n_predict = generation_config.n_predict;
    turn_request.generation_options.n_threads = generation_config.n_threads;

    common_agent_runtime_tooling tooling;
    tooling.tools = tools;
    const std::vector<common_blueprint_candidate> blueprints;
    const std::vector<common_memory_hit> memory_hits;
    std::string current_plan_id;
    common_agent_runtime_host_execution execution{
        common_agent_runtime_host_mode::agent,
        memories,
        &plans,
        observing,
        std::move(turn_request),
        &current_plan_id,
        &blueprints,
        &memory_hits,
        tooling,
    };
    common_agent_result runtime_result;
    const bool runtime_ok = run_agent_runtime_host(execution, runtime_result, error);
    // The causal smoke consumes planner observations. The subsequent real
    // action may fail because this smoke deliberately supplies no tool
    // handler; that is not a planner-capture failure once observations exist.
    observations = std::move(observing.observations);
    if (observations.empty() && !runtime_ok) return false;
    error.clear();
    return true;
}

bool run_planner_counterfactual(
        common_agent_inference & delegate,
        const options & value,
        const causal_fixture & fixture,
        const std::vector<common_chat_tool> & tools,
        const std::shared_ptr<const common_flydelta_hidden_state_capture_request> & capture,
        const std::shared_ptr<const common_agent_residual_patch_request> & patch,
        planner_counterfactual & out,
        std::string & error) {
    std::vector<planner_generation_observation> observations;
    if (!run_planner_observations(delegate, value, fixture, tools, capture, patch,
            observations, error)) return false;

    if (fixture.repair_task != nullptr) {
        const auto find_observation = [&](
                const std::vector<planner_generation_observation> & values,
                const char * expected_tool, const char * expected_arguments,
                planner_generation_observation & selected) {
            for (const auto & observation : values) {
                if (planner_tool(observation.result) == expected_tool &&
                        same_json(planner_arguments(observation.result), expected_arguments)) {
                    selected = observation;
                    return true;
                }
            }
            return false;
        };
        planner_generation_observation baseline;
        if (!find_observation(observations, fixture.failed_tool,
                fixture.failed_arguments, baseline)) {
            std::cerr << "planner explicit-pair baseline was not emitted\n";
            return false;
        }
        causal_fixture repair_fixture = fixture;
        repair_fixture.task = fixture.repair_task;
        std::vector<planner_generation_observation> repair_observations;
        if (!run_planner_observations(delegate, value, repair_fixture, tools, capture, patch,
                repair_observations, error)) return false;
        planner_generation_observation repair;
        if (!find_observation(repair_observations, fixture.expected_tool,
                fixture.expected_arguments, repair)) {
            std::cerr << "planner explicit-pair repair was not emitted\n";
            return false;
        }
        out.baseline = std::move(baseline);
        out.repair = std::move(repair);
        out.diagnostic_context = out.baseline.request;
        out.diagnostic_context.flydelta_capture.reset();
        out.diagnostic_context.residual_patch.reset();
        error.clear();
        return true;
    }

    size_t baseline_index = observations.size();
    size_t repair_index = observations.size();
    for (size_t index = 0; index < observations.size(); ++index) {
        const auto & observation = observations[index];
        const std::string tool = planner_tool(observation.result);
        const std::string arguments = planner_arguments(observation.result);
        const bool rejected_baseline = tool == fixture.failed_tool &&
            (std::string(fixture.failed_tool) != std::string(fixture.expected_tool) ||
                same_json(arguments, fixture.failed_arguments));
        if (baseline_index == observations.size() && rejected_baseline) {
            baseline_index = index;
            continue;
        }
        if (baseline_index != observations.size() &&
                tool == fixture.expected_tool && same_json(arguments, fixture.expected_arguments)) {
            repair_index = index;
            break;
        }
    }
    if (baseline_index == observations.size() || repair_index == observations.size()) {
        std::cerr << "planner observations=" << observations.size()
                  << " planner_error=" << (error.empty() ? "<none>" : error) << '\n';
        for (const auto & observation : observations) {
            std::cout << "planner_candidate tool=" << (planner_tool(observation.result).empty()
                ? "<none>" : planner_tool(observation.result))
                << " args=" << (planner_arguments(observation.result).empty()
                    ? "<none>" : planner_arguments(observation.result)) << '\n';
        }
        return false;
    }
    out.baseline = observations[baseline_index];
    out.repair = observations[repair_index];
    out.diagnostic_context = out.baseline.request;
    out.diagnostic_context.flydelta_capture.reset();
    out.diagnostic_context.residual_patch.reset();
    error.clear();
    return true;
}

} // namespace

int main(int argc, char ** argv) {
    options value;
    if (!parse_options(argc, argv, value)) {
        std::cerr << "usage: " << argv[0]
                  << " --model MODEL [--threads N] [--n-gpu-layers N]"
                  << " [--fixture schema-selection|overview-selection|explicit-repair-pair|openalex-known-id]"
                  << " [--openalex-spec PATH]"
                  << " [--layer N|--all-layers] [--full-matrix] [--capture-only]\n";
        return 2;
    }
    if (value.model.empty() || !std::filesystem::is_regular_file(value.model)) return 77;
    const causal_fixture * fixture = fixture_for(value);
    if (fixture == nullptr) {
        std::cerr << "unknown causal fixture: " << value.fixture << '\n';
        return 2;
    }

    std::string error;
    std::vector<common_chat_tool> tools;
    if (fixture->openapi) {
        if (!load_openalex_tools(value, *fixture, tools, error)) { std::cerr << error << '\n'; return 2; }
    } else {
        common_tool_catalog catalog;
        common_tool_bootstrap_result bootstrap;
        if (!catalog.bootstrap("analysis", bootstrap, error)) { std::cerr << error << '\n'; return 1; }
        for (const char * name : {fixture->expected_tool, fixture->failed_tool}) {
            const auto * definition = catalog.find_definition(name);
            if (!definition) { std::cerr << "missing tool " << name << '\n'; return 1; }
            const auto schema = common_tool_model_input_schema(*definition);
            std::string render_error;
            const auto description = common_render_compact_tool_description(
                definition->name, definition->description, schema,
                common_tool_model_result_schema(*definition), render_error);
            if (!render_error.empty()) { std::cerr << render_error << '\n'; return 1; }
            tools.push_back({definition->name, description, schema,
                common_tool_model_result_schema(*definition)});
        }
    }

    auto host = std::make_shared<common_agent_server_context_host>();
    common_agent_server_context_host_config config;
    config.context_key.load_key.model = value.model;
    config.context_key.load_key.n_gpu_layers = value.gpu_layers;
    config.context_key.load_key.fit_params = true;
    config.context_key.n_parallel = 1;
    config.context_key.n_sequences = 1;
    config.context_key.n_ctx = 2048;
    config.context_key.n_threads = value.threads;
    config.verbosity = LOG_LEVEL_WARN;
    if (!host->start(config, error)) { std::cerr << "server host: " << error << '\n'; return 1; }
    common_agent_inference_session session;
    if (!host->build_inference_session(session, error) || !session.inference) {
        std::cerr << "server inference session: " << error << '\n'; return 1;
    }
    auto inference = std::move(session.inference);
    auto capture = capture_request("sha256:flydelta-causal-patch-qwen", value.layers);
    planner_counterfactual counterfactual;
    if (!run_planner_counterfactual(*inference, value, *fixture, tools, capture, {},
            counterfactual, error)) {
        std::cout << "fixture_not_counterfactual"
                  << " fixture=" << fixture->id
                  << " expected_baseline_tool=" << fixture->failed_tool
                  << " expected_repair_tool=" << fixture->expected_tool
                  << " baseline_tool=<planner-did-not-emit-rejected-call>"
                  << " repair_tool=<not_run>"
                  << " learning_credit=no promotion=no\n";
        return 77;
    }
    const common_agent_generation_result & baseline = counterfactual.baseline.result;
    const common_agent_generation_result & repair = counterfactual.repair.result;
    const std::string baseline_selected_tool = planner_tool(baseline);
    const std::string repair_selected_tool = planner_tool(repair);
    const std::string baseline_selected_arguments = planner_arguments(baseline);
    const std::string repair_selected_arguments = planner_arguments(repair);
    if (!baseline.flydelta_capture || !repair.flydelta_capture ||
            !baseline.flydelta_capture->captured || !repair.flydelta_capture->captured) {
        std::cerr << "causal patch smoke could not capture baseline/repair states\n";
        return 1;
    }
    std::cout << "baseline_tool=" << baseline_selected_tool
              << " repair_tool=" << repair_selected_tool
              << " baseline_arguments=" << baseline_selected_arguments
              << " repair_arguments=" << repair_selected_arguments
              << " baseline_position=" << baseline.flydelta_capture->token_index
              << " repair_position=" << repair.flydelta_capture->token_index
              << " baseline_token_id=" << baseline.flydelta_capture->token_id
              << " repair_token_id=" << repair.flydelta_capture->token_id
              << " baseline_prompt_tokens=" << baseline.flydelta_capture->prompt_token_count
              << " repair_prompt_tokens=" << repair.flydelta_capture->prompt_token_count << '\n';

    const int32_t position = baseline.flydelta_capture->token_index;
    common_agent_teacher_forced_choice_result baseline_margin;
    common_agent_teacher_forced_choice_result repair_margin;
    const bool baseline_margin_available = score_margin(
        *inference, *fixture, counterfactual.diagnostic_context, {}, baseline_margin);
    const bool repair_margin_available = score_margin(
        *inference, *fixture, counterfactual.diagnostic_context, {}, repair_margin);
    std::cout << "baseline_teacher_margin_available="
              << (baseline_margin_available ? "yes" : "no")
              << " baseline_teacher_margin="
              << (baseline_margin_available ? baseline_margin.normalized_delta() : 0.0f)
              << " repair_teacher_margin_available="
              << (repair_margin_available ? "yes" : "no")
              << " repair_teacher_margin="
              << (repair_margin_available ? repair_margin.normalized_delta() : 0.0f) << '\n';
    struct arm_result {
        std::string family;
        uint32_t layer = 0;
        float alpha = 0.0f;
        float direction_norm = 0.0f;
        float construction_equivalence_norm = 0.0f;
        std::shared_ptr<const common_agent_residual_patch_request> patch;
        common_agent_teacher_forced_choice_result margin;
        bool margin_available = false;
    };
    std::vector<arm_result> diagnostic_arms;
    bool any_direction_signal = false;
    bool observed_nonzero_delta = false;
    for (const uint32_t layer : value.layers) {
        std::vector<float> base, repaired;
        if (!capture_layer(*baseline.flydelta_capture, layer, base) ||
                !capture_layer(*repair.flydelta_capture, layer, repaired) ||
                base.size() != baseline.flydelta_capture->n_embd ||
                repaired.size() != repair.flydelta_capture->n_embd ||
                base.size() != repaired.size() ||
                !finite_nonzero_capture(base) || !finite_nonzero_capture(repaired)) {
            std::cerr << "capture_integrity_failed layer=" << layer
                      << " expected_n_embd=" << baseline.flydelta_capture->n_embd
                      << " baseline_values=" << base.size()
                      << " repair_values=" << repaired.size() << '\n';
            return 1;
        }
        std::vector<float> exact_delta(base.size());
        for (size_t i = 0; i < base.size(); ++i) exact_delta[i] = repaired[i] - base[i];
        double base_sum = 0.0;
        double repair_sum = 0.0;
        double base_sq_sum = 0.0;
        double repair_sq_sum = 0.0;
        bool equal = base.size() == repaired.size();
        bool finite = true;
        for (size_t i = 0; i < base.size() && i < repaired.size(); ++i) {
            base_sum += base[i];
            repair_sum += repaired[i];
            base_sq_sum += double(base[i]) * base[i];
            repair_sq_sum += double(repaired[i]) * repaired[i];
            equal = equal && base[i] == repaired[i];
            finite = finite && std::isfinite(base[i]) && std::isfinite(repaired[i]);
        }
        std::cout << "capture layer=" << layer
                  << " base_norm=" << norm(base)
                  << " repair_norm=" << norm(repaired)
                  << " delta_norm=" << norm(exact_delta)
                  << " base_sum=" << base_sum
                  << " repair_sum=" << repair_sum
                  << " base_sq_sum=" << base_sq_sum
                  << " repair_sq_sum=" << repair_sq_sum
                  << " exactly_equal=" << (equal ? "yes" : "no")
                  << " finite=" << (finite ? "yes" : "no") << '\n';
        const float direction_norm = norm(exact_delta);
        observed_nonzero_delta = observed_nonzero_delta || direction_norm > 1.0e-6f;

        if (value.capture_only) continue;
        const auto add_diagnostic_arm = [&](const char * family,
                common_agent_residual_patch_operation operation,
                const std::vector<float> & source, float alpha) {
            auto patch = std::make_shared<common_agent_residual_patch_request>();
            patch->operation = operation;
            patch->site = common_agent_residual_patch_site::layer_input_residual;
            patch->layer = layer;
            patch->absolute_position = position;
            // The smoke configures one resident sequence.  Bind the patch to
            // that exact slot instead of accepting a server-selected fallback.
            patch->sequence_id = 0;
            patch->values = source;
            for (float & component : patch->values) component *= alpha;
            patch->identity = std::string(family) + ":layer=" + std::to_string(layer) +
                ":alpha=" + std::to_string(alpha);
            arm_result arm;
            arm.family = family;
            arm.layer = layer;
            arm.alpha = alpha;
            arm.direction_norm = direction_norm;
            if (std::string(family) == "scaled_exact_delta" &&
                    std::abs(alpha - 1.0f) <= 1.0e-6f) {
                std::vector<float> reconstructed = base;
                for (size_t i = 0; i < reconstructed.size(); ++i) {
                    reconstructed[i] += exact_delta[i] * alpha;
                }
                std::vector<float> reconstruction_error(reconstructed.size());
                for (size_t i = 0; i < reconstructed.size(); ++i) {
                    reconstruction_error[i] = reconstructed[i] - repaired[i];
                }
                arm.construction_equivalence_norm = norm(reconstruction_error);
            }
            arm.patch = patch;
            common_agent_teacher_forced_choice_result margin;
            arm.margin_available = score_margin(
                *inference, *fixture, counterfactual.diagnostic_context, patch, margin);
            if (arm.margin_available) {
                arm.margin = margin;
                any_direction_signal = any_direction_signal ||
                    (std::string(family) == "single_pair_exact_delta" &&
                        margin.total_delta() > (baseline_margin_available
                            ? baseline_margin.total_delta() : 0.0f));
            }
            std::cout << "diagnostic family=" << family << " layer=" << layer
                      << " position=" << position << " operation="
                      << (operation == common_agent_residual_patch_operation::replace ? "replace" : "add")
                      << " alpha=" << alpha
                      << " teacher_margin_available=" << (arm.margin_available ? "yes" : "no")
                      << " teacher_margin=" << (arm.margin_available ? arm.margin.normalized_delta() : 0.0f)
                      << " direction_norm=" << direction_norm;
            if (std::string(family) == "scaled_exact_delta" &&
                    std::abs(alpha - 1.0f) <= 1.0e-6f) {
                std::cout << " alpha1_state_equivalence_norm="
                          << arm.construction_equivalence_norm
                          << " alpha1_state_equivalence="
                          << (arm.construction_equivalence_norm <= 1.0e-5f
                              ? "within_tolerance" : "mismatch");
            }
            std::cout << '\n';
            diagnostic_arms.push_back(std::move(arm));
        };

        // Identity control: replacing the captured baseline state must retain
        // its teacher-forced decision within ordinary numerical noise.
        add_diagnostic_arm("self_replacement", common_agent_residual_patch_operation::replace,
            base, 1.0f);

        // This is intentionally named after what it is.  It is not the
        // aggregated production FlyDelta basis.  The small alpha values are
        // the existing-direction diagnostic, not a production search grid.
        const std::vector<float> direction_alphas = value.full_matrix
            ? std::vector<float>{0.01f, 0.02f} : std::vector<float>{0.02f};
        for (const float alpha : direction_alphas) {
            add_diagnostic_arm("single_pair_exact_delta", common_agent_residual_patch_operation::add,
                exact_delta, alpha);
            add_diagnostic_arm("norm_matched_random_control", common_agent_residual_patch_operation::add,
                random_control(exact_delta, layer), alpha);
        }
        const std::vector<float> scaled_alphas = value.full_matrix
            ? std::vector<float>{0.25f, 0.5f, 0.75f, 1.0f} : std::vector<float>{0.5f, 1.0f};
        for (const float alpha : scaled_alphas) {
            add_diagnostic_arm("scaled_exact_delta", common_agent_residual_patch_operation::add,
                exact_delta, alpha);
        }
        add_diagnostic_arm("exact_replacement", common_agent_residual_patch_operation::replace,
            repaired, 1.0f);
    }
    if (!observed_nonzero_delta) {
        std::cerr << "capture_integrity_failed no_tested_layer_has_nonzero_delta\n";
        return 1;
    }
    if (!value.capture_only) {
        // Diagnostics are teacher-forced first.  Full generation is deliberately
        // bounded: the exact replacement is the causal reference, while only
        // the strongest non-exact candidate per layer is allowed to reach the
        // expensive observable-behavior/Oracle phase.
        std::vector<size_t> frontier;
        for (const uint32_t layer : value.layers) {
            size_t exact_index = diagnostic_arms.size();
            size_t best_index = diagnostic_arms.size();
            float best_delta = -std::numeric_limits<float>::infinity();
            for (size_t index = 0; index < diagnostic_arms.size(); ++index) {
                const auto & arm = diagnostic_arms[index];
                if (arm.layer != layer || !arm.margin_available) continue;
                if (arm.family == "exact_replacement") exact_index = index;
                if (arm.family != "exact_replacement" && arm.margin.total_delta() > best_delta) {
                    best_delta = arm.margin.total_delta();
                    best_index = index;
                }
            }
            if (exact_index < diagnostic_arms.size() &&
                    (!baseline_margin_available ||
                        diagnostic_arms[exact_index].margin.total_delta() > baseline_margin.total_delta())) {
                frontier.push_back(exact_index);
            }
            if (best_index < diagnostic_arms.size() && best_index != exact_index &&
                    (!baseline_margin_available || best_delta > baseline_margin.total_delta())) {
                frontier.push_back(best_index);
            }
        }
        for (const size_t index : frontier) {
            auto & arm = diagnostic_arms[index];
            std::vector<planner_generation_observation> observations;
            std::string planner_error;
            const bool executed = run_planner_observations(
                *inference, value, *fixture, tools, capture, arm.patch, observations, planner_error);
            size_t valid_index = observations.size();
            for (size_t observation_index = 0; observation_index < observations.size(); ++observation_index) {
                if (planner_tool(observations[observation_index].result) == fixture->expected_tool &&
                        same_json(planner_arguments(observations[observation_index].result),
                            fixture->expected_arguments)) {
                    valid_index = observation_index;
                    break;
                }
            }
            const bool candidate_passed = valid_index < observations.size();
            const common_agent_generation_result * result = candidate_passed
                ? &observations[valid_index].result : nullptr;
            const std::string selected = result == nullptr ? std::string{} : planner_tool(*result);
            const std::string selected_args = result == nullptr ? std::string{} : planner_arguments(*result);
            bool patch_attempted = false;
            bool patch_applied = false;
            std::string patch_failure_reason;
            for (const auto & observation : observations) {
                if (!observation.result.residual_patch_observation) continue;
                patch_attempted = patch_attempted ||
                    observation.result.residual_patch_observation->attempted;
                patch_applied = patch_applied ||
                    observation.result.residual_patch_observation->applied;
                if (patch_failure_reason.empty()) {
                    patch_failure_reason =
                        observation.result.residual_patch_observation->failure_reason;
                }
            }
            std::cout << "frontier family=" << arm.family << " layer=" << arm.layer
                      << " position=" << position << " alpha=" << arm.alpha
                      << " executed=" << (executed ? "yes" : "no")
                      << " patch_attempted=" << (patch_attempted ? "yes" : "no")
                      << " patch_applied=" << (patch_applied ? "yes" : "no")
                      << " selected_tool=" << (selected.empty() ? "<none>" : selected)
                      << " selected_arguments=" << (selected_args.empty() ? "<none>" : selected_args)
                      << " teacher_margin=" << (arm.margin_available ? arm.margin.normalized_delta() : 0.0f)
                      << " outcome=" << (selected.empty() ? "UNKNOWN" : outcome_name(
                          baseline_selected_tool == fixture->expected_tool, candidate_passed))
                      << (patch_failure_reason.empty() ? "" :
                          " patch_failure=" + patch_failure_reason)
                      << '\n';
        }
        std::cout << "frontier_count=" << frontier.size() << " diagnostic_count="
                  << diagnostic_arms.size() << '\n';
    }
    if (value.capture_only) {
        std::cout << "capture_integrity_smoke learning_credit=no promotion=no\n";
    } else {
        std::cout << "causal_patch_smoke direction_signal=" << (any_direction_signal ? "yes" : "no")
                  << " learning_credit=no promotion=no\n";
    }
    return 0;
}
