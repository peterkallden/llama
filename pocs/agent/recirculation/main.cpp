#include "llama-backend.h"
#include "experiment.h"
#include "window.h"
#include "historical-window.h"
#include "alternative-direction.h"
#include "selective-mask.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <filesystem>

using json = nlohmann::ordered_json;
using namespace recirculation;
struct options {
    std::string model, text_file, trace, cases_file, experiment_mode, dev_cases_file, eval_cases_file, selection_file;
    uint32_t steps = 128, prompt_tokens = 32;
    float novelty_threshold = 0.15f;
    unsigned window_rounds = 2;
    std::vector<std::string> case_ids;
    recirculation_config config;
};
static json metrics(const observation & o) {
    return {{"position", o.position}, {"first_completed", o.first_completed}, {"rewound", o.rewound}, {"patch_matched", o.patch_matched},
        {"second_completed", o.second_completed}, {"committed", o.committed}, {"source_norm", o.source_norm},
        {"destination_norm", o.destination_norm}, {"delta_norm", o.delta_norm}, {"first_us", o.first_us},
        {"second_us", o.second_us}, {"failure", o.failure}};
}
static double max_difference(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size() || a.empty()) throw std::runtime_error("logit size mismatch");
    double difference = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) throw std::runtime_error("non-finite logits");
        difference = std::max(difference, std::abs(double(a[i]) - b[i]));
        if (std::abs(double(a[i]) - b[i]) > 1e-4 + 1e-4 * std::max(std::abs(a[i]), std::abs(b[i])))
            throw std::runtime_error("logit tolerance exceeded");
    }
    return difference;
}
static std::vector<float> logits_of(llama_backend & b) {
    const auto count = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(b.context())));
    auto * data = llama_get_logits_ith(b.context(), 0);
    if (!data) throw std::runtime_error("missing reference logits");
    return {data, data + count};
}
static double loss(const std::vector<float> & logits, int32_t token) {
    if (token < 0 || size_t(token) >= logits.size()) throw std::runtime_error("invalid target token");
    const double top = *std::max_element(logits.begin(), logits.end());
    double total = 0;
    for (float v : logits) total += std::exp(v - top);
    return std::log(total) + top - logits[token];
}
static std::vector<llama_token> tokenize(llama_model * model, const std::string & text) {
    auto * vocab = llama_model_get_vocab(model);
    int count = llama_tokenize(vocab, text.data(), int32_t(text.size()), nullptr, 0, true, false);
    if (count >= 0) throw std::runtime_error("empty tokenization");
    std::vector<llama_token> tokens(-count);
    count = llama_tokenize(vocab, text.data(), int32_t(text.size()), tokens.data(), int32_t(tokens.size()), true, false);
    if (count <= 0) throw std::runtime_error("tokenization failed");
    tokens.resize(count); return tokens;
}
static void check_step(runner & run, int32_t token, int32_t pos, std::vector<float> & logits,
        observation & obs, std::ostream & trace, const char * arm) {
    const auto start = std::chrono::steady_clock::now();
    const bool success = run.step(token, pos, logits, obs);
    const auto total_us = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count();
    trace << json{{"event", "token"}, {"arm", arm}, {"total_us", total_us}, {"token", token}, {"observation", metrics(obs)}}.dump() << '\n';
    if (!success) throw std::runtime_error(std::string(arm) + ": " + obs.failure);
}
// Fault injection uses the real CPU abort callback only in this smoke adapter.
struct abort_backend final : backend {
    llama_backend host;
    abort_backend(llama_model * model, const recirculation_config & config, uint32_t capacity) : host(model, config, capacity) {}
    bool ready(int32_t p) override { return host.ready(p); }
    bool first(int32_t t, int32_t p, bool c, capture & out) override { return host.first(t, p, c, out); }
    bool rewind(int32_t p) override { return host.rewind(p); }
    bool second(int32_t t, int32_t p, const std::vector<float> & patch, bool & matched) override {
        llama_set_abort_callback(host.context(), [](void *) { return true; }, nullptr);
        return host.second(t, p, patch, matched);
    }
    void clear_hooks() noexcept override { host.clear_hooks(); }
    void discard() noexcept override { host.discard(); }
};
static void model_smoke(llama_model * model, const options & opt, const std::vector<llama_token> & data, std::ostream & trace) {
    auto config = opt.config;
    auto identity = config; identity.alpha = 0; identity.beta = 1;
    const size_t width = llama_model_n_embd(model), layers = llama_model_n_layer(model);
    const uint32_t capacity = opt.steps + 1;
    llama_backend base(model, config, capacity), disabled(model, config, capacity), id(model, identity, capacity),
        targeted_id(model, identity, capacity);
    llama_backend active(model, config, capacity), replay(model, config, capacity);
    runner disabled_run(disabled, config, layers, width, false), identity_run(id, identity, layers, width);
    runner targeted_identity_run(targeted_id, identity, layers, width, true, patch_policy::selected_position, 31);
    runner active_run(active, config, layers, width);
    observation obs;
    std::vector<float> normal, off, same, selected_identity, changed;
    double base_loss = 0, active_loss = 0;
    int loss_count = 0;
    std::vector<int32_t> history;
    std::vector<std::vector<float>> patches;
    auto started = std::chrono::steady_clock::now();
    for (uint32_t i = 0; i < opt.steps; ++i) {
        const int32_t token = i < opt.prompt_tokens || opt.text_file.size() ? data.at(i) :
            int32_t(std::max_element(changed.begin(), changed.end()) - changed.begin());
        history.push_back(token);
        capture baseline;
        if (!base.ready(i) || !base.first(token, i, false, baseline)) throw std::runtime_error("baseline decode failed");
        normal = std::move(baseline.logits);
        check_step(disabled_run, token, i, off, obs, trace, "disabled");
        const auto disabled_error = max_difference(normal, off);
        check_step(identity_run, token, i, same, obs, trace, "identity");
        const auto identity_error = max_difference(normal, same);
        check_step(targeted_identity_run, token, i, selected_identity, obs, trace, "targeted_identity");
        const auto targeted_identity_error = max_difference(normal, selected_identity);
        check_step(active_run, token, i, changed, obs, trace, "recirculation");
        // Reference A reads independently replayed, patch-only history. It is
        // disposable; the persistent replay context is never decoded twice.
        double first_error = 0;
        if (i == 0 || i == opt.steps - 1 || i == 31) {
            llama_backend probe(model, config, capacity);
            for (uint32_t j = 0; j < i; ++j) {
                bool matched = false;
                if (!probe.second(history[j], j, patches[j], matched) || !matched) throw std::runtime_error("history replay failed");
            }
            capture first;
            if (!probe.first(token, i, false, first)) throw std::runtime_error("reference A failed");
            first_error = max_difference(changed, first.logits);
        }
        bool matched = false;
        if (!replay.ready(i) || !replay.second(token, i, active.last_patch, matched) || !matched)
            throw std::runtime_error("reference B failed");
        const auto second_error = max_difference(logits_of(active), logits_of(replay));
        patches.push_back(active.last_patch);
        trace << json{{"event", "comparison"}, {"position", i}, {"disabled_max_error", disabled_error},
            {"identity_max_error", identity_error}, {"targeted_identity_max_error", targeted_identity_error},
            {"targeted_identity_patch_expected", i == 31}, {"reference_A_checked", i == 0 || i == opt.steps - 1 || i == 31},
            {"reference_A_max_error", first_error}, {"reference_B_max_error", second_error},
            {"patch", active.last_patch}}.dump() << '\n';
        if (i + 1 < data.size() && (!opt.text_file.empty() || i + 1 < opt.prompt_tokens)) {
            base_loss += loss(normal, data[i + 1]); active_loss += loss(changed, data[i + 1]); ++loss_count;
        }
        // At the same step, identity B must also preserve the baseline readout.
        max_difference(normal, logits_of(id));
    }
    // A separate fresh sequence must not inherit hooks or KV from any arm.
    llama_backend fresh(model, config, capacity);
    runner fresh_run(fresh, config, layers, width);
    std::vector<float> first;
    check_step(fresh_run, history[0], 0, first, obs, trace, "fresh_sequence");
    llama_backend fresh_base(model, config, capacity);
    capture baseline;
    if (!fresh_base.first(history[0], 0, false, baseline)) throw std::runtime_error("fresh baseline failed");
    max_difference(first, baseline.logits);
    abort_backend aborted(model, config, capacity);
    runner abort_run(aborted, config, layers, width);
    std::vector<float> failed_logits;
    if (abort_run.step(history[0], 0, failed_logits, obs) || !failed_logits.empty() ||
            !obs.first_completed || !obs.rewound || obs.committed || aborted.host.context())
        throw std::runtime_error("abort did not discard provisional context");
    trace << json{{"event", "controlled_abort"}, {"observation", metrics(obs)}, {"context_discarded", true}}.dump() << '\n';
    if (abort_run.step(history[0], 0, failed_logits, obs)) throw std::runtime_error("discarded context reused");
    for (const bool wrong_sequence : {false, true}) {
        llama_backend wrong(model, config, capacity);
        llama_residual_patch_ref ref;
        ref.operation = llama_residual_patch_operation::replace;
        ref.layer = config.destination_layer_input;
        ref.absolute_position = wrong_sequence ? 0 : 1;
        ref.sequence_id = wrong_sequence ? 1 : 0;
        ref.values = patches[0].data(); ref.n_embd = patches[0].size();
        bool matched_wrong = false; ref.applied = &matched_wrong;
        llama_set_residual_patch(wrong.context(), &ref);
        if (!wrong.decode(history[0], 0)) throw std::runtime_error("wrong-row probe decode failed");
        llama_set_residual_patch(wrong.context(), nullptr);
        if (matched_wrong) throw std::runtime_error("wrong patch row matched");
        max_difference(logits_of(wrong), baseline.logits);
        trace << json{{"event", "wrong_patch_row"}, {"wrong_sequence", wrong_sequence}, {"matched", matched_wrong}}.dump() << '\n';
    }
    size_t memory = 0;
    for (const auto & entry : llama_get_memory_breakdown(active.context())) memory += entry.second.total();
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - started).count();
    trace << json{{"event", "summary"}, {"status", "passed"}, {"tokens", opt.steps}, {"elapsed_ms_including_references", elapsed},
        {"reported_active_model_context_compute_bytes", memory}, {"loss_samples", loss_count},
        {"baseline_mean_loss", loss_count ? json(base_loss / loss_count) : json(nullptr)},
        {"recirculation_mean_loss", loss_count ? json(active_loss / loss_count) : json(nullptr)},
        {"quality_interpretation", opt.text_file.empty() ? "generation-path diagnostic; not held-out quality evidence" : "teacher-forced comparison on supplied evaluation text"}}.dump() << '\n';
}
static std::string normalized(std::string value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    return value.substr(first, value.find_last_not_of(" \t\r\n") - first + 1);
}
static void evaluate_cases(llama_model * model, const options & opt, std::ostream & trace) {
    if (opt.cases_file.empty()) return;
    std::ifstream input(opt.cases_file);
    if (!input) throw std::runtime_error("cannot open state-tracking cases");
    std::string line;
    size_t count = 0;
    while (std::getline(input, line)) {
        if (line.empty()) continue;
        if (++count > 64 || line.size() > 65536) throw std::runtime_error("cases budget exceeded");
        const auto task = json::parse(line);
        const auto prompt = task.at("prompt").get<std::string>();
        const auto expected = normalized(task.at("expected").get<std::string>());
        const auto tokens = tokenize(model, prompt);
        if (tokens.size() > 512 || expected.empty()) throw std::runtime_error("invalid case");
        for (const bool enabled : {false, true}) {
            llama_backend host(model, opt.config, uint32_t(tokens.size() + 65));
            runner run(host, opt.config, llama_model_n_layer(model), llama_model_n_embd(model), enabled);
            std::vector<float> logits;
            observation obs;
            int32_t position = 0;
            auto started = std::chrono::steady_clock::now();
            for (const auto token : tokens) check_step(run, token, position++, logits, obs, trace, enabled ? "case_recirc" : "case_baseline");
            const auto ttft = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - started).count();
            std::string answer;
            unsigned generated = 0;
            for (; generated < 64; ++generated) {
                const int32_t token = int32_t(std::max_element(logits.begin(), logits.end()) - logits.begin());
                if (llama_vocab_is_eog(llama_model_get_vocab(model), token)) break;
                std::vector<char> piece(256);
                int size = llama_token_to_piece(llama_model_get_vocab(model), token, piece.data(), int(piece.size()), 0, false);
                if (size < 0) { piece.resize(-size); size = llama_token_to_piece(llama_model_get_vocab(model), token, piece.data(), int(piece.size()), 0, false); }
                if (size < 0) throw std::runtime_error("token piece failed");
                answer.append(piece.data(), size);
                check_step(run, token, position++, logits, obs, trace, enabled ? "case_recirc" : "case_baseline");
            }
            const auto us = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - started).count();
            trace << json{{"event", "state_tracking_case"}, {"case", count}, {"enabled", enabled}, {"prompt", prompt},
                {"expected", expected}, {"answer", answer}, {"exact_match_after_trim", normalized(answer) == expected},
                {"ttft_us", ttft}, {"total_us", us}, {"generated_tokens", generated}}.dump() << '\n';
        }
    }
    if (count == 0) throw std::runtime_error("empty cases file");
}
int main(int argc, char ** argv) {
    try {
        options opt;
        opt.config.source_layer_input = 8; opt.config.destination_layer_input = 3;
        for (int i = 1; i < argc; ++i) {
            const std::string key = argv[i];
            if (i + 1 >= argc) throw std::runtime_error("expected option value");
            const std::string value = argv[++i];
            if (key == "--model") opt.model = value;
            else if (key == "--trace") opt.trace = value;
            else if (key == "--cases-file") opt.cases_file = value;
            else if (key == "--experiment-mode") opt.experiment_mode = value;
            else if (key == "--dev-cases-file") opt.dev_cases_file = value;
            else if (key == "--eval-cases-file") opt.eval_cases_file = value;
            else if (key == "--case-id") opt.case_ids.push_back(value);
            else if (key == "--window-rounds") opt.window_rounds = std::stoul(value);
            else if (key == "--selection-file") opt.selection_file = value;
            else if (key == "--novelty-threshold") opt.novelty_threshold = std::stof(value);
            else if (key == "--text-file") opt.text_file = value;
            else if (key == "--steps") {
                size_t consumed = 0;
                const auto n = std::stoul(value, &consumed);
                if (consumed != value.size() || n < 128 || n > 512) throw std::runtime_error("invalid steps");
                opt.steps = uint32_t(n);
            }
            else if (key == "--source") opt.config.source_layer_input = std::stoul(value);
            else if (key == "--destination") opt.config.destination_layer_input = std::stoul(value);
            else if (key == "--alpha") opt.config.alpha = std::stof(value);
            else if (key == "--beta") opt.config.beta = std::stof(value);
            else throw std::runtime_error("unknown option: " + key);
        }
        if (opt.model.empty()) { std::cout << "SKIP: supply --model for model smoke; run CTests for model-free contracts\n"; return 77; }
        if (opt.trace.empty() || (opt.experiment_mode.empty() && (opt.steps < 128 || opt.steps > 512))) throw std::runtime_error("require --trace and 128 <= --steps <= 512 outside experiment mode");
        if (!opt.experiment_mode.empty() && opt.experiment_mode != "tune" && opt.experiment_mode != "evaluate" && opt.experiment_mode != "repair" && opt.experiment_mode != "target-evaluate" && opt.experiment_mode != "window-smoke" && opt.experiment_mode != "historical-window" && opt.experiment_mode != "alternative-direction" && opt.experiment_mode != "selective-mask") throw std::runtime_error("invalid experiment mode");
        if (!std::isfinite(opt.novelty_threshold) || opt.novelty_threshold < 0 || opt.novelty_threshold > 2) throw std::runtime_error("invalid novelty threshold");
        if (opt.experiment_mode == "tune" && (opt.dev_cases_file.empty() || opt.selection_file.empty())) throw std::runtime_error("tune mode requires --dev-cases-file and --selection-file");
        if ((opt.experiment_mode == "evaluate" || opt.experiment_mode == "repair" || opt.experiment_mode == "target-evaluate") && (opt.eval_cases_file.empty() || opt.selection_file.empty())) throw std::runtime_error("evaluation requires --eval-cases-file and --selection-file");
        if (opt.experiment_mode == "historical-window" &&
                (opt.eval_cases_file.empty() || opt.case_ids.empty() || opt.window_rounds < 1 ||
                 opt.window_rounds > 3 || opt.config.beta != 1.0f))
            throw std::runtime_error("historical window requires explicit cases, 1-3 rounds and beta=1");
        if (opt.experiment_mode == "alternative-direction" &&
                (opt.dev_cases_file.empty() || opt.eval_cases_file.empty() || opt.case_ids.empty() ||
                 opt.config.beta != 1.0f))
            throw std::runtime_error("alternative direction requires development, evaluation and explicit cases");
        if (opt.experiment_mode == "selective-mask" &&
                (opt.dev_cases_file.empty() || opt.eval_cases_file.empty() || opt.case_ids.empty() ||
                 opt.config.beta != 1.0f || opt.config.alpha <= 0))
            throw std::runtime_error("selective mask requires development, evaluation, explicit cases, beta=1 and positive alpha");
        opt.config.model_identity = std::filesystem::canonical(opt.model).string();
        std::ofstream trace(opt.trace);
        if (!trace) throw std::runtime_error("cannot open trace");
        trace.exceptions(std::ios::badbit | std::ios::failbit);
        const auto size = std::filesystem::file_size(opt.model);
        if (size == 0) throw std::runtime_error("empty model");
        const auto mtime = std::filesystem::last_write_time(opt.model).time_since_epoch().count();
        trace << json{{"event", "configuration"}, {"schema", "recirculation-poc.v1"}, {"experiment_mode", opt.experiment_mode}, {"source_revision", RECIRCULATION_SOURCE_REVISION},
            {"worktree_status", "not_embedded; archive git diff with trace"}, {"backend", "CPU"}, {"threads", 1},
            {"evaluation_text", opt.text_file}, {"evaluation_cases", opt.cases_file}, {"development_cases", opt.dev_cases_file},
            {"held_out_cases", opt.eval_cases_file}, {"selection_file", opt.selection_file},
            {"source_snapshot", "HEAD plus local worktree; archive diff with trace"},
            {"model", opt.config.model_identity}, {"model_bytes", size}, {"model_mtime", mtime},
            {"source_layer_input", opt.config.source_layer_input}, {"destination_layer_input", opt.config.destination_layer_input},
            {"alpha", opt.config.alpha}, {"beta", opt.config.beta}, {"extra_passes", 1},
            {"window_rounds", opt.window_rounds}, {"case_ids", opt.case_ids},
            {"sampling", "greedy; one selection per generated token"},
            {"paper_schedule", "same-token A/B serial dependency schedule; paper setup and results not reproduced"}}.dump() << '\n';
        llama_backend_init();
        struct release { ~release() { llama_backend_free(); } } cleanup;
        auto params = llama_model_default_params(); params.n_gpu_layers = 0;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(opt.model.c_str(), params), llama_model_free);
        if (!model) throw std::runtime_error("model load failed");
        if (opt.experiment_mode == "selective-mask") {
            const auto development = read_experiment_cases(model.get(), opt.dev_cases_file);
            const auto evaluation = read_experiment_cases(model.get(), opt.eval_cases_file);
            const std::unordered_set<std::string> requested(opt.case_ids.begin(), opt.case_ids.end());
            if (requested.size() != opt.case_ids.size()) throw std::runtime_error("duplicate selective mask case ID");
            const size_t width = size_t(llama_model_n_embd(model.get()));
            const size_t count = std::max<size_t>(1, width / 4);
            std::vector<double> magnitudes(width, 0);
            std::vector<const experiment_case *> dev_cases;
            std::unordered_set<std::string> dev_ids;
            for (const auto & item : development) {
                dev_ids.insert(item.id);
                if (item.task_type != "object_location" || item.prompt_tokens.size() < 6) continue;
                const auto captured = alternative_capture_layers(model.get(), opt.config, item.prompt_tokens);
                for (size_t i = 0; i < width; ++i) magnitudes[i] += std::abs(double(captured.downstream[i]));
                dev_cases.push_back(&item);
            }
            if (dev_cases.size() < 2) throw std::runtime_error("insufficient development mask cases");
            for (const auto & item : evaluation) if (dev_ids.count(item.id))
                throw std::runtime_error("development and evaluation case IDs overlap");
            std::vector<std::vector<uint8_t>> candidates{selective_magnitude_mask(magnitudes, count)};
            for (uint32_t seed : {11u, 23u, 37u}) candidates.push_back(selective_random_mask(width, count, seed));
            std::vector<double> dev_mean(candidates.size(), 0);
            for (const auto * item : dev_cases) {
                for (size_t i = 0; i < candidates.size(); ++i) {
                    const auto result = selective_mask_case(model.get(), *item, opt.config, candidates[i], {});
                    dev_mean[i] += result.at("selected_mask").at("margin").get<double>() -
                        result.at("baseline").at("margin").get<double>();
                }
            }
            for (double & value : dev_mean) value /= dev_cases.size();
            const size_t chosen = size_t(std::max_element(dev_mean.begin(), dev_mean.end()) - dev_mean.begin());
            std::vector<size_t> selected_indices;
            for (size_t i = 0; i < width; ++i) if (candidates[chosen][i]) selected_indices.push_back(i);
            trace << json{{"event", "selective_mask_selection"}, {"candidate_names", {"source_magnitude", "random_11", "random_23", "random_37"}},
                {"development_mean_margin_changes", dev_mean}, {"selected_candidate", chosen},
                {"selected_indices", selected_indices}, {"development_cases", dev_cases.size()},
                {"mask_density", count}, {"selection_rule", "maximum_mean_development_margin_change"},
                {"control_seeds", {101, 103, 107}}, {"locked_before_evaluation", true}}.dump() << '\n';
            const std::vector<std::vector<uint8_t>> controls{
                selective_random_mask(width, count, 101), selective_random_mask(width, count, 103),
                selective_random_mask(width, count, 107)};
            size_t completed = 0;
            for (const auto & item : evaluation) {
                if (!requested.count(item.id)) continue;
                if (item.task_type != "object_location") throw std::runtime_error("selective mask requires object-location cases");
                const auto result = selective_mask_case(model.get(), item, opt.config, candidates[chosen], controls);
                trace << json{{"event", "selective_mask_case"}, {"result", result}}.dump() << '\n';
                ++completed;
            }
            if (completed != requested.size()) throw std::runtime_error("selective mask case ID not found");
            std::cout << "PASS: selective mask cases=" << completed << '\n';
            return 0;
        }
        if (opt.experiment_mode == "alternative-direction") {
            const auto development = read_experiment_cases(model.get(), opt.dev_cases_file);
            const auto evaluation = read_experiment_cases(model.get(), opt.eval_cases_file);
            const std::unordered_set<std::string> requested(opt.case_ids.begin(), opt.case_ids.end());
            if (requested.size() != opt.case_ids.size()) throw std::runtime_error("duplicate alternative case ID");
            for (const auto & item : development)
                if (requested.count(item.id)) throw std::runtime_error("development and evaluation cases overlap");
            const auto fit = alternative_fit_direction(model.get(), opt.config, development);
            trace << json{{"event", "alternative_direction_fit"}, {"training_ids", fit.training_ids},
                {"excluded_ids", fit.excluded_ids}, {"training_alignments", fit.training_alignments},
                {"leave_one_out_alignments", fit.leave_one_out_alignments},
                {"downstream_leave_one_out_alignments", fit.downstream_leave_one_out_alignments},
                {"capture_layer", opt.config.destination_layer_input},
                {"downstream_reference_layer", opt.config.source_layer_input},
                {"capture_site", "second_to_last_prompt_token"}, {"direction_semantics", "development_update_minus_counterfactual_old_location"},
                {"experimental_only", true}}.dump() << '\n';
            size_t completed = 0;
            for (const auto & item : evaluation) {
                if (!requested.count(item.id)) continue;
                const auto result = alternative_direction_case(model.get(), item, opt.config,
                    fit.values, fit.downstream_reference);
                trace << json{{"event", "alternative_direction_case"}, {"result", result}}.dump() << '\n';
                std::cout << item.id << " baseline=" << result.at("baseline").at("margin")
                    << " alternative_two=" << result.at("alternative_two").at("margin") << '\n';
                ++completed;
            }
            if (completed != requested.size()) throw std::runtime_error("alternative case ID not found");
            std::cout << "PASS: alternative direction cases=" << completed << '\n';
            return 0;
        }
        if (opt.experiment_mode == "historical-window") {
            const auto cases = read_experiment_cases(model.get(), opt.eval_cases_file);
            const std::unordered_set<std::string> selected(opt.case_ids.begin(), opt.case_ids.end());
            if (selected.size() != opt.case_ids.size()) throw std::runtime_error("duplicate historical case ID");
            size_t completed = 0;
            for (const auto & item : cases) {
                if (!selected.count(item.id)) continue;
                const auto result = historical_window_case(model.get(), item, opt.config, opt.window_rounds, trace);
                trace << json{{"event", "historical_window_case"}, {"result", result}}.dump() << '\n';
                std::cout << item.id << " old_gain=" << result.at("old_targeted_gain")
                    << " window_forced_gain=" << result.at("window_forced_gain")
                    << " window_policy_gain=" << result.at("window_policy_gain")
                    << " committed=" << result.at("committed") << '\n';
                ++completed;
            }
            if (completed != selected.size()) throw std::runtime_error("historical case ID not found");
            std::cout << "PASS: historical window cases=" << completed << '\n';
            return 0;
        }
        if (opt.experiment_mode == "window-smoke") {
            const auto tokens = tokenize(model.get(), "A red key is in a drawer. Then Mira moves the key into a pouch. The key is now in the pouch.");
            if (tokens.size() < 24) throw std::runtime_error("window smoke prompt too short");
            const std::vector<llama_token> prefix(tokens.begin(), tokens.begin() + 16);
            const std::vector<llama_token> window(tokens.begin() + 16, tokens.begin() + 22);
            auto identity = opt.config; identity.alpha = 0;
            const auto identity_window = run_window(model.get(), identity, prefix, window, 2, 1);
            const double identity_error = max_difference(identity_window.baseline_logits, identity_window.refined_logits);
            llama_backend targeted_host(model.get(), opt.config, 23);
            runner targeted(targeted_host, opt.config, llama_model_n_layer(model.get()), llama_model_n_embd(model.get()),
                true, patch_policy::selected_position, 18);
            std::vector<float> targeted_logits;
            observation targeted_obs;
            for (int32_t pos = 0; pos < 22; ++pos) {
                if (!targeted.step(tokens[size_t(pos)], pos, targeted_logits, targeted_obs))
                    throw std::runtime_error("targeted window reference failed: " + targeted_obs.failure);
            }
            for (unsigned rounds = 1; rounds <= 3; ++rounds) {
                auto result = run_window(model.get(), opt.config, prefix, window, 2, rounds);
                const double targeted_error = rounds == 1 ? max_difference(targeted_logits, result.refined_logits) : 0;
                llama_backend imported_window(model.get(), opt.config, 64);
                if (llama_state_seq_set_data(imported_window.context(), result.refined_sequence_state.data(),
                        result.refined_sequence_state.size(), 0) != result.refined_sequence_state.size() ||
                        !imported_window.ready(22)) throw std::runtime_error("refined window sequence import failed");
                llama_backend continuation(model.get(), opt.config, 23);
                window_restore(continuation.context(), result.committed ? result.refined_state : result.baseline_state);
                if (!continuation.ready(22)) throw std::runtime_error("window continuation state invalid");
                capture next;
                if (!continuation.first(tokens[22], 22, false, next)) throw std::runtime_error("window continuation failed");
                capture imported_next;
                if (!imported_window.first(tokens[22], 22, false, imported_next))
                    throw std::runtime_error("imported window continuation failed");
                llama_backend refined_direct(model.get(), opt.config, 23);
                window_restore(refined_direct.context(), result.refined_state);
                capture refined_next;
                if (!refined_direct.first(tokens[22], 22, false, refined_next))
                    throw std::runtime_error("refined direct continuation failed");
                const double refined_import_error = max_difference(imported_next.logits, refined_next.logits);
                if (refined_import_error > 1e-5)
                    throw std::runtime_error("refined window sequence continuation diverged");
                llama_backend larger(model.get(), opt.config, 64);
                auto * source_ctx = continuation.context();
                const size_t seq_size = llama_state_seq_get_size(source_ctx, 0);
                std::vector<uint8_t> seq_state(seq_size);
                if (!seq_size || llama_state_seq_get_data(source_ctx, seq_state.data(), seq_size, 0) != seq_size ||
                        llama_state_seq_set_data(larger.context(), seq_state.data(), seq_size, 0) != seq_size ||
                        !larger.ready(23)) throw std::runtime_error("cross-capacity sequence restore failed");
                capture copied_next;
                if (!larger.first(tokens[23], 23, false, copied_next)) throw std::runtime_error("cross-capacity continuation failed");
                capture direct_next;
                if (!continuation.first(tokens[23], 23, false, direct_next)) throw std::runtime_error("direct continuation failed");
                const double sequence_restore_error = max_difference(direct_next.logits, copied_next.logits);
                if (sequence_restore_error > 1e-5)
                    throw std::runtime_error("cross-capacity sequence continuation diverged");
                trace << json{{"event", "window_smoke"}, {"rounds", rounds}, {"completed_rounds", result.completed_rounds},
                    {"committed", result.committed}, {"baseline_entropy", result.baseline_entropy},
                    {"refined_entropy", result.refined_entropy}, {"patch_changes", result.patch_changes},
                    {"identity_max_error", identity_error}, {"targeted_max_error", targeted_error},
                    {"continuation_position", 22}, {"continuation_logits", next.logits.size()},
                    {"refined_sequence_import_error", refined_import_error},
                    {"cross_capacity_sequence_restore_error", sequence_restore_error}}.dump() << '\n';
            }
            for (const auto mode : {window_target_mode::uncertainty, window_target_mode::curvature}) {
                const auto selected = run_window(model.get(), opt.config, prefix, window, 0, 2, 0.5f, mode, 0);
                const auto absent = run_window(model.get(), opt.config, prefix, window, 0, 2, 0.5f, mode, 3);
                if (absent.target_position != -1 || absent.completed_rounds != 0)
                    throw std::runtime_error("window no-trigger contract failed");
                trace << json{{"event", "window_target_smoke"},
                    {"mode", mode == window_target_mode::uncertainty ? "uncertainty" : "curvature"},
                    {"selected_position", selected.target_position}, {"trigger_signal", selected.trigger_signal},
                    {"rounds", selected.completed_rounds}, {"committed", selected.committed},
                    {"no_trigger_verified", true}}.dump() << '\n';
            }
            std::cout << "PASS: window smoke; trace " << opt.trace << '\n';
            return 0;
        }
        if (!opt.experiment_mode.empty()) {
            try {
                if (opt.experiment_mode == "tune") {
                    const auto cases = read_experiment_cases(model.get(), opt.dev_cases_file);
                    auto selection = tune_experiment(model.get(), opt.config.model_identity, cases, trace);
                    selection["source_revision"] = RECIRCULATION_SOURCE_REVISION;
                    selection["mechanism_detail"] = "single-token patch during same-token A/B replay; not paper-equivalence";
                    std::ofstream selected(opt.selection_file);
                    if (!selected) throw std::runtime_error("cannot write selection file");
                    selected << selection.dump(2) << '\n';
                    trace << experiment_json{{"event", "development_selection"}, {"selection", selection}}.dump() << '\n';
                } else {
                    std::ifstream selected(opt.selection_file);
                    if (!selected) throw std::runtime_error("cannot open locked selection file");
                    const auto selection = experiment_json::parse(selected);
                    if (selection.at("schema") != "recirculation-selection.v1" ||
                            selection.at("mechanism") != "targeted_same_token_replay" ||
                            selection.at("model") != opt.config.model_identity ||
                            selection.at("selection_dataset") != "development") throw std::runtime_error("selection file does not match model, mechanism or schema");
                    const auto chosen = selection.at("selected");
                    const auto config = experiment_config(opt.config.model_identity,
                        chosen.at("source_layer_input").get<size_t>(), chosen.at("destination_layer_input").get<size_t>(),
                        chosen.at("alpha").get<float>());
                    if (chosen.at("beta").get<float>() != 1.0f || !recirculation_config_valid(config,
                            llama_model_n_layer(model.get()), llama_model_n_embd(model.get()))) throw std::runtime_error("locked selection is invalid for this model");
                    const auto cases = read_experiment_cases(model.get(), opt.eval_cases_file);
                    for (const auto & item : cases) {
                        for (const auto & id : selection.at("development_case_ids")) {
                            if (id.get<std::string>() == item.id) throw std::runtime_error("development and evaluation case IDs overlap");
                        }
                    }
                    if (opt.experiment_mode == "target-evaluate") {
                        for (const auto & item : cases) {
                            const auto base = score_experiment_case(model.get(), item, config, experiment_arm::baseline, trace);
                            json arms = json::object();
                            for (const auto arm : {experiment_arm::targeted, experiment_arm::host_window,
                                    experiment_arm::activation_novelty, experiment_arm::random_control}) {
                                const auto score = score_experiment_case(model.get(), item, config, arm, trace, {}, opt.novelty_threshold);
                                json lags = json::array();
                                if (score.selected_position >= 0) for (size_t position = size_t(score.selected_position + 1);
                                        position < score.token_path_logprobs.size() && position < score.incorrect_token_path_logprobs.size(); ++position) {
                                    lags.push_back({{"lag", position - size_t(score.selected_position)}, {"token_position", position},
                                        {"expected_delta", score.token_path_logprobs[position] - base.token_path_logprobs[position]},
                                        {"incorrect_delta", score.incorrect_token_path_logprobs[position] - base.incorrect_token_path_logprobs[position]}});
                                }
                                arms[experiment_arm_name(arm)] = {{"selected_position", score.selected_position},
                                    {"triggered", score.patch_applied}, {"margin_change", score.margin - base.margin},
                                    {"elapsed_us", score.elapsed_us}, {"token_lags", std::move(lags)}};
                            }
                            trace << json{{"event", "target_evaluation"}, {"schema", "recirculation-target-v1"},
                                {"case_id", item.id}, {"task_type", item.task_type}, {"oracle_position", item.target_position},
                                {"novelty_threshold", opt.novelty_threshold}, {"baseline_margin", base.margin},
                                {"baseline_elapsed_us", base.elapsed_us}, {"arms", std::move(arms)}}.dump() << '\n';
                        }
                    } else if (opt.experiment_mode == "repair") {
                        const auto report = evaluate_repair_scenario(model.get(), opt.config.model_identity, cases, config, trace);
                        trace << experiment_json{{"event", "repair_scenario"}, {"report", report}}.dump() << '\n';
                    } else {
                    if (selection.contains("locked_comparisons")) {
                        std::unordered_map<std::string, experiment_json> cached;
                        for (const auto & comparison : selection.at("locked_comparisons")) {
                            const auto chosen_comparison = comparison.at("configuration");
                            const auto comparison_config = experiment_config(opt.config.model_identity,
                                chosen_comparison.at("source_layer_input").get<size_t>(),
                                chosen_comparison.at("destination_layer_input").get<size_t>(),
                                chosen_comparison.at("alpha").get<float>());
                            if (!recirculation_config_valid(comparison_config, llama_model_n_layer(model.get()),
                                    llama_model_n_embd(model.get())) || chosen_comparison.at("beta") != 1.0) {
                                throw std::runtime_error("invalid locked comparison");
                            }
                            const auto key = chosen_comparison.dump();
                            if (!cached.count(key)) cached.emplace(key,
                                evaluate_experiment(model.get(), opt.config.model_identity, cases, comparison_config, trace));
                            trace << experiment_json{{"event", "locked_comparison"}, {"method", comparison.at("method")},
                                {"comparison", comparison.at("comparison")}, {"report", cached.at(key)}}.dump() << '\n';
                        }
                    } else {
                        for (const auto & role : {std::string("geometry_ranked_pair"), std::string("fixed_control_pair")}) {
                            auto pair = selection.at("geometry_ranked_pair");
                            if (role == "fixed_control_pair") {
                                for (const auto & candidate_pair : selection.at("geometry_ranking")) {
                                    if (candidate_pair.at("source_layer_input") != pair.at("source_layer_input") ||
                                        candidate_pair.at("destination_layer_input") != pair.at("destination_layer_input")) {
                                        pair = candidate_pair;
                                        break;
                                    }
                                }
                            }
                            double best_margin = -std::numeric_limits<double>::infinity();
                            recirculation_config pair_config;
                            for (const auto & candidate : selection.at("candidates")) {
                                if (candidate.at("source_layer_input") != pair.at("source_layer_input") ||
                                    candidate.at("destination_layer_input") != pair.at("destination_layer_input")) continue;
                                const double margin = candidate.at("mean_margin_change").get<double>();
                                if (margin > best_margin) {
                                    best_margin = margin;
                                    pair_config = experiment_config(opt.config.model_identity,
                                        pair.at("source_layer_input").get<size_t>(), pair.at("destination_layer_input").get<size_t>(),
                                        candidate.at("alpha").get<float>());
                                }
                            }
                            if (!std::isfinite(best_margin)) throw std::runtime_error("missing development dose selection");
                            const auto pair_report = evaluate_experiment(model.get(), opt.config.model_identity, cases, pair_config, trace);
                            trace << experiment_json{{"event", "pair_comparison"}, {"role", role}, {"report", pair_report}}.dump() << '\n';
                        }
                        const auto report = evaluate_experiment(model.get(), opt.config.model_identity, cases, config, trace);
                        trace << experiment_json{{"event", "held_out_evaluation"}, {"selection_file", opt.selection_file}, {"report", report}}.dump() << '\n';
                    }
                    }
                }
            } catch (const std::exception & e) {
                trace << experiment_json{{"event", "experiment_failure"}, {"mode", opt.experiment_mode}, {"reason", e.what()}}.dump() << '\n';
                throw;
            }
            std::cout << "PASS: experiment; trace " << opt.trace << '\n';
            return 0;
        }
        std::string text;
        if (opt.text_file.empty()) {
            for (int i = 0; i < 12; ++i) text += "Mira has a red key. She gives it to Leo. Leo puts the key in a box. Who has the key now? ";
        } else {
            std::ifstream input(opt.text_file);
            if (!input) throw std::runtime_error("cannot open evaluation text");
            std::ostringstream buffer; buffer << input.rdbuf(); text = buffer.str();
        }
        if (text.size() > 1024 * 1024) throw std::runtime_error("evaluation text too large");
        const auto data = tokenize(model.get(), text);
        if (data.size() <= (opt.text_file.empty() ? opt.prompt_tokens : opt.steps)) throw std::runtime_error("insufficient evaluation tokens");
        try { model_smoke(model.get(), opt, data, trace); evaluate_cases(model.get(), opt, trace); }
        catch (const std::exception & e) { trace << json{{"event", "failure"}, {"reason", e.what()}}.dump() << '\n'; throw; }
        std::cout << "PASS: model smoke; trace " << opt.trace << '\n';
        return 0;
    } catch (const std::exception & e) { std::cerr << "recirculation: " << e.what() << '\n'; return 1; }
}
