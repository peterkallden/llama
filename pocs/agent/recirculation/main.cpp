#include "llama-backend.h"
#include "experiment.h"
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
            else if (key == "--selection-file") opt.selection_file = value;
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
        if (!opt.experiment_mode.empty() && opt.experiment_mode != "tune" && opt.experiment_mode != "evaluate" && opt.experiment_mode != "repair") throw std::runtime_error("invalid experiment mode");
        if (opt.experiment_mode == "tune" && (opt.dev_cases_file.empty() || opt.selection_file.empty())) throw std::runtime_error("tune mode requires --dev-cases-file and --selection-file");
        if ((opt.experiment_mode == "evaluate" || opt.experiment_mode == "repair") && (opt.eval_cases_file.empty() || opt.selection_file.empty())) throw std::runtime_error("evaluate mode requires --eval-cases-file and --selection-file");
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
            {"alpha", opt.config.alpha}, {"beta", opt.config.beta}, {"extra_passes", 1}, {"sampling", "greedy; one selection per generated token"},
            {"paper_schedule", "same-token A/B serial dependency schedule; paper setup and results not reproduced"}}.dump() << '\n';
        llama_backend_init();
        struct release { ~release() { llama_backend_free(); } } cleanup;
        auto params = llama_model_default_params(); params.n_gpu_layers = 0;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(opt.model.c_str(), params), llama_model_free);
        if (!model) throw std::runtime_error("model load failed");
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
                    if (opt.experiment_mode == "repair") {
                        const auto report = evaluate_repair_scenario(model.get(), opt.config.model_identity, cases, config, trace);
                        trace << experiment_json{{"event", "repair_scenario"}, {"report", report}}.dump() << '\n';
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
