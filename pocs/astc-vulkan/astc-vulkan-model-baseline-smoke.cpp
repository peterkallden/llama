#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace {

bool tokenize(const llama_vocab * vocab, const std::string & prompt, std::vector<llama_token> & tokens) {
    const int32_t count = llama_tokenize(vocab, prompt.c_str(), static_cast<int32_t>(prompt.size()),
                                         nullptr, 0, true, true);
    if (count >= 0) return false;
    tokens.resize(static_cast<size_t>(-count));
    return llama_tokenize(vocab, prompt.c_str(), static_cast<int32_t>(prompt.size()),
                          tokens.data(), static_cast<int32_t>(tokens.size()), true, true) >= 0;
}

llama_batch make_batch(const std::vector<llama_token> & tokens) {
    llama_batch batch = llama_batch_init(static_cast<int32_t>(tokens.size()), 0, 1);
    batch.n_tokens = static_cast<int32_t>(tokens.size());
    for (size_t i = 0; i < tokens.size(); ++i) {
        batch.token[i] = tokens[i];
        batch.pos[i] = static_cast<llama_pos>(i);
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = 1;
    }
    return batch;
}

struct logits_result {
    std::vector<float> values;
    size_t n_tokens = 0;
    size_t n_vocab = 0;
    int32_t status = 1;
};

logits_result run_model(llama_model * model, const std::vector<llama_token> & tokens) {
    llama_context_params params = llama_context_default_params();
    params.n_ctx = std::max<uint32_t>(512, static_cast<uint32_t>(tokens.size()));
    params.n_batch = static_cast<uint32_t>(tokens.size());
    params.n_ubatch = static_cast<uint32_t>(tokens.size());
    llama_context * context = llama_init_from_model(model, params);
    if (context == nullptr) return {};

    llama_batch batch = make_batch(tokens);
    logits_result result;
    result.status = llama_decode(context, batch);
    if (result.status == 0) {
        result.n_tokens = tokens.size();
        result.n_vocab = static_cast<size_t>(llama_vocab_n_tokens(llama_model_get_vocab(model)));
        result.values.resize(result.n_tokens * result.n_vocab);
        for (size_t token = 0; token < result.n_tokens; ++token) {
            const float * logits = llama_get_logits_ith(context, static_cast<int32_t>(token));
            if (logits == nullptr) {
                result.values.clear();
                break;
            }
            std::copy(logits, logits + result.n_vocab, result.values.begin() + token * result.n_vocab);
        }
    }
    llama_batch_free(batch);
    llama_free(context);
    return result;
}

double cross_entropy(const float * logits, size_t n_vocab, llama_token target) {
    float max_logit = logits[0];
    for (size_t i = 1; i < n_vocab; ++i) max_logit = std::max(max_logit, logits[i]);
    double sum_exp = 0.0;
    for (size_t i = 0; i < n_vocab; ++i) {
        sum_exp += std::exp(static_cast<double>(logits[i] - max_logit));
    }
    return std::log(sum_exp) + max_logit - logits[target];
}

} // namespace

int main(int argc, char ** argv) {
    std::string reference_path, candidate_path, prompt;
    for (int i = 1; i < argc; ++i) {
        const std::string option = argv[i];
        if (i + 1 >= argc) break;
        if (option == "--reference-model") reference_path = argv[++i];
        else if (option == "--model") candidate_path = argv[++i];
        else if (option == "--prompt") prompt = argv[++i];
        else { std::fprintf(stderr, "unknown option: %s\n", option.c_str()); return 2; }
    }
    if (reference_path.empty() || candidate_path.empty() || prompt.empty()) {
        std::fprintf(stderr, "usage: %s --reference-model fp16.gguf --model candidate.gguf --prompt text\n", argv[0]);
        return 2;
    }

    llama_backend_init();
    llama_model * reference_model = llama_model_load_from_file(reference_path.c_str(), llama_model_default_params());
    llama_model * candidate_model = llama_model_load_from_file(candidate_path.c_str(), llama_model_default_params());
    if (reference_model == nullptr || candidate_model == nullptr) {
        std::fprintf(stderr, "failed to load reference or candidate model\n");
        if (reference_model) llama_model_free(reference_model);
        if (candidate_model) llama_model_free(candidate_model);
        llama_backend_free();
        return 1;
    }

    std::vector<llama_token> tokens;
    if (!tokenize(llama_model_get_vocab(reference_model), prompt, tokens) || tokens.empty()) {
        std::fprintf(stderr, "failed to tokenize prompt\n");
        llama_model_free(candidate_model);
        llama_model_free(reference_model);
        llama_backend_free();
        return 1;
    }
    const logits_result reference = run_model(reference_model, tokens);
    const logits_result candidate = run_model(candidate_model, tokens);
    if (reference.status != 0 || candidate.status != 0 || reference.values.size() != candidate.values.size() ||
        reference.n_vocab != candidate.n_vocab) {
        std::fprintf(stderr, "baseline replay failed: reference=%d candidate=%d\n", reference.status, candidate.status);
        llama_model_free(candidate_model);
        llama_model_free(reference_model);
        llama_backend_free();
        return 1;
    }

    double mse = 0.0, reference_energy = 0.0, reference_loss = 0.0, candidate_loss = 0.0;
    size_t top1_matches = 0, loss_tokens = 0;
    for (size_t i = 0; i < reference.values.size(); ++i) {
        const double delta = static_cast<double>(reference.values[i]) - candidate.values[i];
        mse += delta * delta;
        reference_energy += static_cast<double>(reference.values[i]) * reference.values[i];
    }
    for (size_t token = 0; token < reference.n_tokens; ++token) {
        const float * ref = reference.values.data() + token * reference.n_vocab;
        const float * got = candidate.values.data() + token * candidate.n_vocab;
        top1_matches += static_cast<size_t>(std::max_element(ref, ref + reference.n_vocab) - ref ==
                                            std::max_element(got, got + candidate.n_vocab) - got);
        if (token + 1 < tokens.size()) {
            reference_loss += cross_entropy(ref, reference.n_vocab, tokens[token + 1]);
            candidate_loss += cross_entropy(got, candidate.n_vocab, tokens[token + 1]);
            ++loss_tokens;
        }
    }
    const double loss_scale = static_cast<double>(std::max(loss_tokens, size_t(1)));
    std::printf("baseline candidate=%s tokens=%zu vocab=%zu logits-mse=%.8g logits-relative-mse=%.8g "
                "top1-agreement=%.8g reference-loss=%.8g candidate-loss=%.8g loss-delta=%.8g\n",
                candidate_path.c_str(), reference.n_tokens, reference.n_vocab,
                mse / std::max(reference.values.size(), size_t(1)),
                mse / std::max(reference_energy, 1e-12),
                static_cast<double>(top1_matches) / std::max(reference.n_tokens, size_t(1)),
                reference_loss / loss_scale, candidate_loss / loss_scale,
                (candidate_loss - reference_loss) / loss_scale);

    llama_model_free(candidate_model);
    llama_model_free(reference_model);
    llama_backend_free();
    return 0;
}
