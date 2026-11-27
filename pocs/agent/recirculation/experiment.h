#pragma once
#include "llama-backend.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <numeric>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace recirculation {
using experiment_json = nlohmann::ordered_json;

struct experiment_case {
    std::string id, task_type, system, user, update_text, expected, incorrect, control_case;
    std::string prompt;
    std::vector<llama_token> prompt_tokens;
    int32_t target_position = -1;
};

inline size_t experiment_context_bytes(llama_context * context);

inline std::string render_experiment_prompt(llama_model * model, const std::string & system,
        const std::string & user) {
    const char * tmpl = llama_model_chat_template(model, nullptr);
    if (!tmpl) throw std::runtime_error("model has no default chat template");
    llama_chat_message messages[] = {{"system", system.c_str()}, {"user", user.c_str()}};
    int32_t size = llama_chat_apply_template(tmpl, messages, 2, true, nullptr, 0);
    if (size <= 0) throw std::runtime_error("chat template rendering failed");
    std::vector<char> output(size + 1);
    size = llama_chat_apply_template(tmpl, messages, 2, true, output.data(), output.size());
    if (size <= 0 || size > int32_t(output.size())) throw std::runtime_error("chat template output invalid");
    return std::string(output.data(), size);
}

inline std::vector<llama_token> experiment_tokenize(llama_model * model, const std::string & text) {
    if (text.size() > size_t(INT32_MAX)) throw std::runtime_error("case text too large");
    auto * vocab = llama_model_get_vocab(model);
    int32_t size = llama_tokenize(vocab, text.data(), int32_t(text.size()), nullptr, 0, true, true);
    if (size >= 0) throw std::runtime_error("case tokenization returned no tokens");
    std::vector<llama_token> tokens(size_t(-size));
    size = llama_tokenize(vocab, text.data(), int32_t(text.size()), tokens.data(), int32_t(tokens.size()), true, true);
    if (size <= 0) throw std::runtime_error("case tokenization failed");
    tokens.resize(size);
    return tokens;
}

inline std::vector<size_t> experiment_token_byte_lengths(llama_model * model,
        const std::vector<llama_token> & tokens, std::string & reconstructed) {
    std::vector<size_t> lengths;
    reconstructed.clear();
    auto * vocab = llama_model_get_vocab(model);
    for (llama_token token : tokens) {
        if (lengths.empty() && llama_vocab_get_add_bos(vocab) && token == llama_vocab_bos(vocab)) {
            lengths.push_back(0);
            continue;
        }
        std::vector<char> piece(256);
        int32_t size = llama_token_to_piece(vocab, token, piece.data(), int32_t(piece.size()), 0, true);
        if (size < 0) {
            piece.resize(size_t(-size));
            size = llama_token_to_piece(vocab, token, piece.data(), int32_t(piece.size()), 0, true);
        }
        if (size < 0) throw std::runtime_error("token piece conversion failed");
        lengths.push_back(size_t(size));
        reconstructed.append(piece.data(), size_t(size));
    }
    return lengths;
}

inline std::vector<experiment_case> read_experiment_cases(llama_model * model, const std::string & path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open experiment cases: " + path);
    std::vector<experiment_case> cases;
    std::unordered_set<std::string> ids;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty()) continue;
        if (line.size() > 65536 || cases.size() >= 64) throw std::runtime_error("experiment case budget exceeded");
        const auto data = experiment_json::parse(line);
        experiment_case item;
        item.id = data.at("id").get<std::string>();
        item.task_type = data.at("task_type").get<std::string>();
        item.system = data.at("system").get<std::string>();
        item.user = data.at("user").get<std::string>();
        item.update_text = data.at("update_text").get<std::string>();
        item.expected = data.at("expected").get<std::string>();
        item.incorrect = data.at("incorrect").get<std::string>();
        item.control_case = data.value("control_case", std::string{});
        if (item.id.empty() || (item.task_type != "object_location" && item.task_type != "spatial_relation" &&
                    item.task_type != "contextual_meaning") || item.system.empty() || item.user.empty() || item.update_text.empty() ||
                    item.expected.empty() || item.incorrect.empty() || item.expected.size() > 1024 || item.incorrect.size() > 1024 ||
                    item.expected == item.incorrect || !ids.insert(item.id).second) {
            throw std::runtime_error("invalid or duplicate experiment case");
        }
        item.prompt = render_experiment_prompt(model, item.system, item.user);
        item.prompt_tokens = experiment_tokenize(model, item.prompt);
        if (item.prompt_tokens.empty() || item.prompt_tokens.size() > 512) throw std::runtime_error("case prompt token budget exceeded");
        std::string reconstructed;
        const auto lengths = experiment_token_byte_lengths(model, item.prompt_tokens, reconstructed);
        if (reconstructed != item.prompt) {
            const size_t mismatch = std::mismatch(reconstructed.begin(), reconstructed.end(), item.prompt.begin(), item.prompt.end()).first - reconstructed.begin();
            throw std::runtime_error("prompt tokenization does not round-trip exactly: " + item.id + " byte=" + std::to_string(mismatch) + " actual=" + reconstructed.substr(mismatch, 40) + " expected=" + item.prompt.substr(mismatch, 40));
        }
        const size_t start = item.prompt.find(item.update_text);
        if (start == std::string::npos || item.prompt.find(item.update_text, start + 1) != std::string::npos) {
            throw std::runtime_error("update text must occur exactly once in rendered prompt: " + item.id);
        }
        size_t target = 0;
        if (!recirculation_token_for_span(lengths, start, item.update_text.size(), target) || target >= item.prompt_tokens.size()) {
            throw std::runtime_error("update text token position is ambiguous: " + item.id);
        }
        item.target_position = int32_t(target);
        cases.push_back(std::move(item));
    }
    if (cases.empty()) throw std::runtime_error("experiment case file is empty");
    std::unordered_map<std::string, const experiment_case *> by_id;
    for (const auto & item : cases) by_id.emplace(item.id, &item);
    for (const auto & item : cases) {
        const auto donor = by_id.find(item.control_case);
        if (item.control_case.empty() || item.control_case == item.id || donor == by_id.end() || donor->second->task_type == item.task_type) {
            throw std::runtime_error("control_case must name a different task type in the same file: " + item.id);
        }
    }
    return cases;
}

struct donor_vector {
    std::vector<float> values;
    int64_t capture_us = 0;
    size_t context_memory_bytes = 0;
};

inline donor_vector experiment_donor_source(llama_model * model, const recirculation_config & config,
        const experiment_case & donor, std::ostream & trace) {
    const auto started = std::chrono::steady_clock::now();
    llama_backend host(model, config, uint32_t(donor.prompt_tokens.size() + 1));
    for (int32_t position = 0; position < donor.target_position; ++position) {
        if (!host.decode(donor.prompt_tokens[size_t(position)], position)) throw std::runtime_error("control donor prefix decode failed");
        trace << experiment_json{{"event", "control_donor_token"}, {"case_id", donor.id},
            {"position", position}, {"token", donor.prompt_tokens[size_t(position)]}, {"captured", false}}.dump() << '\n';
    }
    llama_set_embeddings_layer_inp(host.context(), config.source_layer_input, true);
    llama_set_embeddings_layer_inp(host.context(), config.destination_layer_input, true);
    capture captured;
    if (!host.first(donor.prompt_tokens[size_t(donor.target_position)], donor.target_position, true, captured) ||
            captured.source.empty()) throw std::runtime_error("control donor capture failed");
    trace << experiment_json{{"event", "control_donor_token"}, {"case_id", donor.id},
        {"position", donor.target_position}, {"token", donor.prompt_tokens[size_t(donor.target_position)]},
        {"captured", true}, {"source_norm", std::sqrt(std::inner_product(captured.source.begin(), captured.source.end(), captured.source.begin(), 0.0))}}.dump() << '\n';
    host.clear_hooks();
    const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - started).count();
    return {std::move(captured.source), elapsed, experiment_context_bytes(host.context())};
}

enum class experiment_arm { baseline, every_token, targeted, matched_control, host_window, activation_novelty, random_control };
inline const char * experiment_arm_name(experiment_arm arm) {
    switch (arm) {
        case experiment_arm::baseline: return "baseline";
        case experiment_arm::every_token: return "same_token_every_token";
        case experiment_arm::targeted: return "targeted_recirculation";
        case experiment_arm::matched_control: return "targeted_norm_matched_control";
        case experiment_arm::host_window: return "host_window";
        case experiment_arm::activation_novelty: return "activation_novelty";
        case experiment_arm::random_control: return "random_control";
    }
    return "unknown";
}

struct scored_continuation {
    // Entries predict target_position + 1, +2, ... on the fixed token path.
    std::vector<double> future_logprobs;
    double logprob = 0;
    int64_t elapsed_us = 0;
    size_t context_memory_bytes = 0;
    float source_norm = 0;
    float destination_norm = 0;
    float patch_delta_norm = 0;
    bool patch_applied = false;
    int32_t selected_position = -1;
    std::vector<double> token_path_logprobs;
};

inline size_t experiment_context_bytes(llama_context * context) {
    size_t bytes = 0;
    for (const auto & item : llama_get_memory_breakdown(context)) bytes += item.second.total();
    return bytes;
}

inline scored_continuation score_experiment_continuation(llama_model * model, const experiment_case & item,
        const std::string & continuation, const recirculation_config & config, experiment_arm arm,
        const std::vector<float> & donor_source, std::ostream & trace, float novelty_threshold = 0.15f) {
    if (arm == experiment_arm::matched_control && donor_source.empty()) {
        throw std::invalid_argument("matched control requires a donor vector");
    }
    const std::string complete = item.prompt + continuation;
    const auto tokens = experiment_tokenize(model, complete);
    if (tokens.size() <= item.prompt_tokens.size() ||
            !std::equal(item.prompt_tokens.begin(), item.prompt_tokens.end(), tokens.begin())) {
        throw std::runtime_error("continuation changes prompt tokenization: " + item.id);
    }
    if (tokens.size() > 640) throw std::runtime_error("case prompt and continuation token budget exceeded: " + item.id);
    const auto started = std::chrono::steady_clock::now();
    const uint32_t capacity = uint32_t(tokens.size() + 1);
    llama_backend host(model, config, capacity);
    const bool enabled = arm != experiment_arm::baseline;
    patch_policy policy = arm == experiment_arm::every_token ? patch_policy::every_token : patch_policy::selected_position;
    if (arm == experiment_arm::host_window) policy = patch_policy::host_window;
    if (arm == experiment_arm::activation_novelty) policy = patch_policy::activation_novelty;
    const int32_t window_start = std::max(0, item.target_position - 3);
    const target_policy target{policy, window_start, item.target_position, novelty_threshold};
    uint32_t hash = 2166136261u;
    for (unsigned char ch : item.id) hash = (hash ^ ch) * 16777619u;
    const int32_t random_position = window_start + int32_t(hash % uint32_t(item.target_position - window_start + 1));
    const std::vector<float> source = arm == experiment_arm::matched_control ? donor_source : std::vector<float>{};
    runner run(host, config, llama_model_n_layer(model), llama_model_n_embd(model), enabled,
        policy, arm == experiment_arm::random_control ? random_position : item.target_position, source, target);
    observation obs;
    std::vector<float> logits;
    float source_norm = 0, destination_norm = 0, patch_delta_norm = 0;
    bool patch_applied = false;
    int32_t selected_position = -1;
    std::vector<double> future_logprobs;
    std::vector<double> token_path_logprobs(1, 0.0);
    for (size_t position = 0; position < item.prompt_tokens.size(); ++position) {
        if (position > 0) {
            const double value = recirculation_token_log_probability(logits, item.prompt_tokens[position]);
            if (!std::isfinite(value)) throw std::runtime_error("invalid future prompt logprob");
            token_path_logprobs.push_back(value);
            if (int32_t(position) > item.target_position) future_logprobs.push_back(value);
        }
        const auto token_started = std::chrono::steady_clock::now();
        if (!run.step(item.prompt_tokens[position], int32_t(position), logits, obs)) {
            throw std::runtime_error(std::string(experiment_arm_name(arm)) + " prompt failed: " + obs.failure);
        }
        const auto token_us = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - token_started).count();
        trace << experiment_json{{"event", "experiment_token"}, {"case_id", item.id}, {"arm", experiment_arm_name(arm)},
            {"phase", "prompt"}, {"position", position}, {"token", item.prompt_tokens[position]},
            {"oracle_position", item.target_position}, {"target_mode", experiment_arm_name(arm)},
            {"patch_completed", obs.second_completed}, {"novelty", obs.novelty},
            {"source_norm", obs.source_norm}, {"destination_norm", obs.destination_norm},
            {"patch_norm", obs.delta_norm}, {"source_destination_cosine", obs.source_destination_cosine}, {"failure", obs.failure}, {"token_us", token_us}}.dump() << '\n';
        if (obs.second_completed && arm != experiment_arm::every_token) selected_position = int32_t(position);
        if (obs.second_completed) {
            source_norm = obs.source_norm;
            destination_norm = obs.destination_norm;
            patch_delta_norm = obs.delta_norm;
            patch_applied = patch_applied || (obs.second_completed && obs.patch_matched && obs.committed);
        }
    }
    std::vector<double> token_logprobs;
    for (size_t index = item.prompt_tokens.size(); index < tokens.size(); ++index) {
        const double value = recirculation_token_log_probability(logits, tokens[index]);
        if (!std::isfinite(value)) throw std::runtime_error("invalid continuation logprob");
        token_logprobs.push_back(value);
        future_logprobs.push_back(value);
        token_path_logprobs.push_back(value);
        int64_t token_us = 0;
        bool decoded = false;
        if (index + 1 < tokens.size()) {
            const auto token_started = std::chrono::steady_clock::now();
            if (!run.step(tokens[index], int32_t(index), logits, obs)) {
                throw std::runtime_error(std::string(experiment_arm_name(arm)) + " continuation failed: " + obs.failure);
            }
            token_us = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - token_started).count();
            decoded = true;
        }
        trace << experiment_json{{"event", "experiment_token"}, {"case_id", item.id}, {"arm", experiment_arm_name(arm)},
            {"phase", "continuation"}, {"position", index}, {"token", tokens[index]}, {"log_probability", value},
            {"decoded_for_next_token", decoded}, {"token_us", token_us}}.dump() << '\n';
    }
    double total_logprob = 0;
    if (!recirculation_sum_log_probabilities(token_logprobs, total_logprob)) throw std::runtime_error("invalid continuation score");
    const int64_t elapsed = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - started).count();
    return {std::move(future_logprobs), total_logprob, elapsed, experiment_context_bytes(host.context()), source_norm, destination_norm, patch_delta_norm, patch_applied, selected_position, std::move(token_path_logprobs)};
}

struct scored_case {
    std::vector<double> future_logprobs;
    std::vector<double> incorrect_future_logprobs;
    double margin = 0;
    double positive = 0;
    double negative = 0;
    int64_t elapsed_us = 0;
    size_t memory_bytes = 0;
    float source_norm = 0;
    float destination_norm = 0;
    float patch_delta_norm = 0;
    bool patch_applied = false;
    int32_t selected_position = -1;
    std::vector<double> token_path_logprobs, incorrect_token_path_logprobs;
};

inline scored_case score_experiment_case(llama_model * model, const experiment_case & item,
        const recirculation_config & config, experiment_arm arm, std::ostream & trace, const std::vector<float> & donor_source = {}, float novelty_threshold = 0.15f) {
    const auto positive = score_experiment_continuation(model, item, item.expected, config, arm, donor_source, trace, novelty_threshold);
    const auto negative = score_experiment_continuation(model, item, item.incorrect, config, arm, donor_source, trace, novelty_threshold);
    if (positive.selected_position != negative.selected_position) throw std::runtime_error("target decision changed across continuations");
    return {positive.future_logprobs, negative.future_logprobs, recirculation_answer_margin(positive.logprob, negative.logprob), positive.logprob, negative.logprob,
        positive.elapsed_us + negative.elapsed_us, std::max(positive.context_memory_bytes, negative.context_memory_bytes),
        positive.source_norm, positive.destination_norm, positive.patch_delta_norm, positive.patch_applied, positive.selected_position,
        positive.token_path_logprobs, negative.token_path_logprobs};
}

inline experiment_json score_record(const scored_case & value) {
    return {{"expected_path_future_logprobs", value.future_logprobs},
        {"incorrect_path_future_logprobs", value.incorrect_future_logprobs},
        {"relative_patch_norm", value.destination_norm > 0 ? experiment_json(value.patch_delta_norm / value.destination_norm) : experiment_json(nullptr)},
        {"positive_total_logprob", value.positive}, {"negative_total_logprob", value.negative},
        {"answer_margin", value.margin}, {"prefers_expected", value.margin > 0},
        {"patch_applied", value.patch_applied}, {"source_norm", value.source_norm},
        {"destination_norm", value.destination_norm}, {"patch_delta_norm", value.patch_delta_norm},
        {"scoring_us", value.elapsed_us}, {"context_memory_bytes", value.memory_bytes}};
}

inline recirculation_config experiment_config(const std::string & model_id, size_t source, size_t destination, float alpha) {
    return {model_id, source, destination, alpha, 1.0f, 1};
}

inline experiment_json tune_experiment(llama_model * model, const std::string & model_id,
        const std::vector<experiment_case> & cases, std::ostream & trace) {
    const size_t layers = llama_model_n_layer(model);
    std::vector<std::pair<size_t, size_t>> pairs;
    for (const auto & fractions : {std::pair<size_t, size_t>{1, 4}, {2, 4}, {3, 4}}) {
        const size_t source = std::min(layers - 1, layers * fractions.first / fractions.second);
        const size_t destination = std::max(size_t(1), source / 3);
        if (source > destination && std::find(pairs.begin(), pairs.end(), std::make_pair(source, destination)) == pairs.end()) {
            pairs.emplace_back(source, destination);
        }
    }
    if (pairs.size() < 2) throw std::runtime_error("model needs at least two distinct candidate layer pairs");
    experiment_json geometry = experiment_json::array();
    double best_geometry = -1;
    std::pair<size_t, size_t> geometry_pair;
    for (const auto & pair : pairs) {
        double novelty = 0;
        const auto config = experiment_config(model_id, pair.first, pair.second, 0.1f);
        for (const auto & item : cases) {
            llama_backend host(model, config, uint32_t(item.prompt_tokens.size() + 1));
            for (int32_t pos = 0; pos < item.target_position; ++pos) {
                if (!host.decode(item.prompt_tokens[size_t(pos)], pos)) throw std::runtime_error("geometry prefix failed");
            }
            capture values;
            if (!host.first(item.prompt_tokens[size_t(item.target_position)], item.target_position, true, values)) {
                throw std::runtime_error("geometry capture failed");
            }
            double dot = 0, source_sq = 0, destination_sq = 0;
            if (values.source.empty() || values.source.size() != values.destination.size()) throw std::runtime_error("geometry capture shape mismatch");
            for (size_t i = 0; i < values.source.size(); ++i) {
                dot += double(values.source[i]) * values.destination[i];
                source_sq += double(values.source[i]) * values.source[i];
                destination_sq += double(values.destination[i]) * values.destination[i];
            }
            if (!(source_sq > 0 && destination_sq > 0) || !std::isfinite(dot)) throw std::runtime_error("invalid geometry norms");
            const double cosine = std::clamp(dot / std::sqrt(source_sq * destination_sq), -1.0, 1.0);
            // Prefer compatible but nonidentical representations; this is an experimental ranking heuristic.
            const double score = std::max(0.0, cosine) * (1.0 - cosine);
            novelty += score;
            trace << experiment_json{{"event", "pair_geometry"}, {"case_id", item.id},
                {"source_layer_input", pair.first}, {"destination_layer_input", pair.second},
                {"source_destination_cosine", cosine}, {"ranking_score", score}}.dump() << '\n';
        }
        novelty /= cases.size();
        geometry.push_back({{"source_layer_input", pair.first}, {"destination_layer_input", pair.second}, {"score", novelty}});
        if (novelty > best_geometry) { best_geometry = novelty; geometry_pair = pair; }
    }
    const float doses[] = {0.01f, 0.03f, 0.1f, 0.2f};
    experiment_json candidates = experiment_json::array();
    double best_objective = -std::numeric_limits<double>::infinity();
    recirculation_config best;
    for (const auto & pair : pairs) for (float alpha : doses) {
        if (pair.first >= layers) continue;
        const auto config = experiment_config(model_id, pair.first, pair.second, alpha);
        std::vector<double> deltas;
        experiment_json rows = experiment_json::array();
        for (const auto & item : cases) {
            const auto base = score_experiment_case(model, item, config, experiment_arm::baseline, trace);
            const auto treatment = score_experiment_case(model, item, config, experiment_arm::targeted, trace);
            const double delta = treatment.margin - base.margin;
            deltas.push_back(delta);
            rows.push_back({{"case_id", item.id}, {"baseline", score_record(base)},
                {"targeted", score_record(treatment)}, {"margin_change_vs_baseline", delta}});
        }
        const double objective = std::accumulate(deltas.begin(), deltas.end(), 0.0) / deltas.size();
        candidates.push_back({{"source_layer_input", pair.first}, {"destination_layer_input", pair.second},
            {"alpha", alpha}, {"beta", 1.0}, {"selection_objective", "mean(targeted answer_margin - baseline answer_margin)"},
            {"mean_margin_change", objective}, {"cases", std::move(rows)}});
        if (objective > best_objective) { best_objective = objective; best = config; }
    }
    if (!std::isfinite(best_objective)) throw std::runtime_error("no valid tuning configuration for model layer count");
    experiment_json development_ids = experiment_json::array();
    for (const auto & item : cases) development_ids.push_back(item.id);
    experiment_json dose_response = experiment_json::array();
    for (const auto & pair : pairs) {
        bool have_previous = false;
        double previous_alpha = 0, previous_margin = 0;
        for (const auto & candidate : candidates) {
            if (candidate.at("source_layer_input") != pair.first || candidate.at("destination_layer_input") != pair.second) continue;
            const double alpha = candidate.at("alpha").get<double>();
            const double margin = candidate.at("mean_margin_change").get<double>();
            if (have_previous) dose_response.push_back({{"source_layer_input", pair.first}, {"destination_layer_input", pair.second},
                {"alpha_low", previous_alpha}, {"alpha_high", alpha},
                {"margin_slope", (margin - previous_margin) / (alpha - previous_alpha)}});
            previous_alpha = alpha; previous_margin = margin; have_previous = true;
        }
    }
    return {{"schema", "recirculation-selection.v1"}, {"mechanism", "targeted_same_token_replay"},
        {"model", model_id}, {"selection_dataset", "development"}, {"cases_used", cases.size()},
        {"selection_objective", "mean(targeted answer_margin - baseline answer_margin)"},
        {"selected", {{"source_layer_input", best.source_layer_input}, {"destination_layer_input", best.destination_layer_input},
            {"alpha", best.alpha}, {"beta", best.beta}, {"mean_margin_change", best_objective}}},
        {"development_case_ids", std::move(development_ids)},
        {"dose_response_secants", std::move(dose_response)},
        {"geometry_ranking", std::move(geometry)},
        {"geometry_ranked_pair", {{"source_layer_input", geometry_pair.first}, {"destination_layer_input", geometry_pair.second}}},
        {"ranking_definition", "mean(max(0,cos(source,destination))*(1-cos(source,destination))); experimental, no concept reference"},
        {"candidates", std::move(candidates)}};
}

inline experiment_json evaluate_experiment(llama_model * model, const std::string & model_id,
        const std::vector<experiment_case> & cases, const recirculation_config & config, std::ostream & trace) {
    std::unordered_map<std::string, const experiment_case *> by_id;
    std::unordered_map<std::string, donor_vector> donors;
    for (const auto & item : cases) by_id.emplace(item.id, &item);
    experiment_json rows = experiment_json::array();
    std::unordered_map<std::string, std::vector<double>> margins;
    std::unordered_map<std::string, size_t> preferred;
    std::unordered_map<std::string, int64_t> times;
    std::unordered_map<std::string, size_t> memory;
    for (const auto & item : cases) {
        const auto donor_it = by_id.find(item.control_case);
        if (donor_it == by_id.end()) throw std::runtime_error("evaluation case has no predetermined control donor: " + item.id);
        auto donor_it_cached = donors.find(item.control_case);
        int64_t donor_capture_charge = 0;
        if (donor_it_cached == donors.end()) {
            donor_it_cached = donors.emplace(item.control_case,
                experiment_donor_source(model, config, *donor_it->second, trace)).first;
            donor_capture_charge = donor_it_cached->second.capture_us;
        }
        const auto & donor = donor_it_cached->second;
        experiment_json arms = experiment_json::object();
        for (experiment_arm arm : {experiment_arm::baseline, experiment_arm::every_token,
                experiment_arm::targeted, experiment_arm::matched_control}) {
            const std::string name = experiment_arm_name(arm);
            auto result = score_experiment_case(model, item, config, arm, trace,
                arm == experiment_arm::matched_control ? donor.values : std::vector<float>{});
            if (arm == experiment_arm::matched_control) {
                result.elapsed_us += donor_capture_charge;
                result.memory_bytes = std::max(result.memory_bytes, donor.context_memory_bytes);
            }
            arms[name] = score_record(result);
            margins[name].push_back(result.margin);
            preferred[name] += result.margin > 0;
            times[name] += result.elapsed_us;
            memory[name] = std::max(memory[name], result.memory_bytes);
        }
        experiment_json lag_effects = experiment_json::array();
        const auto baseline_path = arms.at("baseline").at("expected_path_future_logprobs").get<std::vector<double>>();
        const auto treatment_path = arms.at("targeted_recirculation").at("expected_path_future_logprobs").get<std::vector<double>>();
        const auto control_path = arms.at("targeted_norm_matched_control").at("expected_path_future_logprobs").get<std::vector<double>>();
        if (baseline_path.size() != treatment_path.size() || baseline_path.size() != control_path.size()) {
            throw std::runtime_error("future token paths are not aligned");
        }
        double cumulative = 0;
        for (size_t i = 0; i < baseline_path.size(); ++i) {
            const double gain = treatment_path[i] - baseline_path[i];
            cumulative += gain;
            lag_effects.push_back({{"lag", i + 1}, {"token_position", item.target_position + i + 1},
                {"logprob_gain_vs_baseline", gain}, {"cumulative_logprob_gain_vs_baseline", cumulative},
                {"logprob_gain_vs_control", treatment_path[i] - control_path[i]}});
        }
        const double targeted_margin = arms.at("targeted_recirculation").at("answer_margin").get<double>();
        const double baseline_margin = arms.at("baseline").at("answer_margin").get<double>();
        const double control_margin = arms.at("targeted_norm_matched_control").at("answer_margin").get<double>();
        rows.push_back({{"case_id", item.id}, {"task_type", item.task_type}, {"update_text", item.update_text},
            {"update_token_position", item.target_position}, {"prompt_token_count", item.prompt_tokens.size()},
            {"control_case", item.control_case}, {"control_vector_capture_us", donor_capture_charge},
            {"control_context_memory_bytes", donor.context_memory_bytes},
            {"targeted_margin_change_vs_baseline", targeted_margin - baseline_margin},
            {"targeted_change_vs_baseline", targeted_margin > baseline_margin ? "improved" :
                targeted_margin < baseline_margin ? "regressed" : "tied"},
            {"targeted_margin_change_vs_control", targeted_margin - control_margin},
            {"targeted_change_vs_control", targeted_margin > control_margin ? "improved" :
                targeted_margin < control_margin ? "regressed" : "tied"},
            {"expected_path_lag_effects", std::move(lag_effects)}, {"arms", std::move(arms)}});
    }
    experiment_json summary = experiment_json::object();
    for (const std::string name : {"baseline", "same_token_every_token", "targeted_recirculation", "targeted_norm_matched_control"}) {
        const auto & values = margins[name];
        const double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
        summary[name] = {{"cases_preferring_expected", preferred[name]}, {"mean_answer_margin", mean},
            {"mean_scoring_us", double(times[name]) / values.size()}, {"max_context_memory_bytes", memory[name]}};
    }
    size_t improved_base = 0, regressed_base = 0, improved_control = 0, regressed_control = 0;
    for (size_t i = 0; i < cases.size(); ++i) {
        if (margins["targeted_recirculation"][i] > margins["baseline"][i]) ++improved_base;
        if (margins["targeted_recirculation"][i] < margins["baseline"][i]) ++regressed_base;
        if (margins["targeted_recirculation"][i] > margins["targeted_norm_matched_control"][i]) ++improved_control;
        if (margins["targeted_recirculation"][i] < margins["targeted_norm_matched_control"][i]) ++regressed_control;
    }
    return {{"schema", "recirculation-evaluation.v1"}, {"mechanism", "targeted_same_token_replay"},
        {"model", model_id}, {"weights", "frozen"}, {"chat_template", "model default; system and user messages, assistant generation prompt"},
        {"score_definition", "sum of token log probabilities for the full continuation; margin = expected total - incorrect total"},
        {"configuration", {{"source_layer_input", config.source_layer_input}, {"destination_layer_input", config.destination_layer_input},
            {"alpha", config.alpha}, {"beta", config.beta}}}, {"summary", std::move(summary)},
        {"targeted_improved_cases_vs_baseline", improved_base}, {"targeted_regressed_cases_vs_baseline", regressed_base},
        {"targeted_improved_cases_vs_matched_control", improved_control}, {"targeted_regressed_cases_vs_matched_control", regressed_control},
        {"cases", std::move(rows)}, {"claim_limit", "Small held-out set for this model and configuration; no general improvement claim."}};
}
inline experiment_json evaluate_repair_scenario(llama_model * model, const std::string & model_id,
        const std::vector<experiment_case> & cases, const recirculation_config & config, std::ostream & trace) {
    experiment_json rows = experiment_json::array();
    size_t failures = 0, rescued = 0, reflected = 0, persistent_passes = 0, persistent_regressions = 0;
    int64_t total_us = 0;
    for (const auto & item : cases) {
        const auto base = score_experiment_case(model, item, config, experiment_arm::baseline, trace);
        const auto persistent = score_experiment_case(model, item, config, experiment_arm::every_token, trace);
        persistent_passes += persistent.margin > 0;
        persistent_regressions += base.margin > 0 && persistent.margin <= 0;
        experiment_json row = {{"case_id", item.id}, {"baseline", score_record(base)},
            {"persistent_context_recirculation", score_record(persistent)}};
        total_us += base.elapsed_us + persistent.elapsed_us;
        if (base.margin <= 0) {
            ++failures;
            const auto retry = score_experiment_case(model, item, config, experiment_arm::baseline, trace);
            const auto recirculated = score_experiment_case(model, item, config, experiment_arm::targeted, trace);
            auto reflection = item;
            reflection.user += "\nPrevious candidate: " + item.incorrect +
                "\nVerification: this candidate failed. Reconsider the update and check the answer carefully.";
            reflection.prompt = render_experiment_prompt(model, reflection.system, reflection.user);
            reflection.prompt_tokens = experiment_tokenize(model, reflection.prompt);
            if (reflection.prompt_tokens.size() > 512) throw std::runtime_error("reflection prompt budget exceeded");
            const auto repaired = score_experiment_case(model, reflection, config, experiment_arm::baseline, trace);
            rescued += recirculated.margin > 0;
            reflected += repaired.margin > 0;
            total_us += retry.elapsed_us + recirculated.elapsed_us + repaired.elapsed_us;
            row["plain_retry"] = score_record(retry);
            row["targeted_retry"] = score_record(recirculated);
            row["reflection_with_failure_feedback"] = score_record(repaired);
            row["reflection_avoided_proxy"] = recirculated.margin > 0;
            row["retry_chain_us"] = base.elapsed_us + recirculated.elapsed_us + (recirculated.margin > 0 ? 0 : repaired.elapsed_us);
        }
        rows.push_back(std::move(row));
    }
    return {{"schema", "recirculation-repair-scenario.v1"}, {"model", model_id},
        {"configuration", {{"source", config.source_layer_input}, {"destination", config.destination_layer_input}, {"alpha", config.alpha}, {"beta", config.beta}}},
        {"cases", std::move(rows)}, {"baseline_failures", failures}, {"targeted_rescues", rescued},
        {"reflection_rescues", reflected}, {"persistent_passes", persistent_passes},
        {"persistent_regressions", persistent_regressions}, {"total_scoring_us", total_us},
        {"claim_limit", "Forced-choice repair proxy, not generated code or an external verifier. Reflection receives the predetermined incorrect candidate and failure feedback, never the expected answer. Fresh contexts replay the prefix; no filesystem rollback is performed. Fixed-path margins do not establish saved real reflection rounds."}};
}

}
