#include "astc-vulkan-input.h"

#include "llama-ext.h"
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

template<typename T>
std::vector<T> read_binary(const std::string & path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) return {};
    const std::streamsize size = file.tellg();
    if (size <= 0 || size % static_cast<std::streamsize>(sizeof(T)) != 0) return {};
    std::vector<T> result(static_cast<size_t>(size) / sizeof(T));
    file.seekg(0);
    file.read(reinterpret_cast<char *>(result.data()), size);
    return file ? result : std::vector<T>();
}

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
    int32_t status = 1;
};

logits_result run_model(llama_model * model, const std::vector<llama_token> & tokens,
                        uint32_t layer, const std::vector<float> * override_output) {
    llama_context_params params = llama_context_default_params();
    params.n_ctx = std::max<uint32_t>(512, static_cast<uint32_t>(tokens.size()));
    params.n_batch = static_cast<uint32_t>(tokens.size());
    params.n_ubatch = static_cast<uint32_t>(tokens.size());
    llama_context * context = llama_init_from_model(model, params);
    if (context == nullptr) return {};

    if (override_output != nullptr &&
        !llama_set_ffn_down_output_override(context, layer, override_output->data(),
                                            static_cast<uint32_t>(tokens.size()),
                                            static_cast<uint32_t>(llama_model_n_embd(model)))) {
        llama_free(context);
        return {};
    }

    llama_batch batch = make_batch(tokens);
    const int32_t status = llama_decode(context, batch);
    logits_result result;
    result.status = status;
    if (status == 0) {
        const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
        const float * logits = llama_get_logits_ith(context, static_cast<int32_t>(tokens.size() - 1));
        if (logits != nullptr) result.values.assign(logits, logits + n_vocab);
    }
    llama_batch_free(batch);
    llama_free(context);
    return result;
}

} // namespace

int main(int argc, char ** argv) {
    std::string model_path, rgba_path, weights_path, activation_path, prompt;
    uint32_t layer = 0, width = 0, height = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string option = argv[i];
        if (i + 1 >= argc) break;
        if (option == "--model") model_path = argv[++i];
        else if (option == "--rgba") rgba_path = argv[++i];
        else if (option == "--weights") weights_path = argv[++i];
        else if (option == "--activations") activation_path = argv[++i];
        else if (option == "--prompt") prompt = argv[++i];
        else if (option == "--layer") layer = static_cast<uint32_t>(std::stoul(argv[++i]));
        else if (option == "--width") width = static_cast<uint32_t>(std::stoul(argv[++i]));
        else if (option == "--height") height = static_cast<uint32_t>(std::stoul(argv[++i]));
        else { std::fprintf(stderr, "unknown option: %s\n", option.c_str()); return 2; }
    }
    if (model_path.empty() || rgba_path.empty() || weights_path.empty() || activation_path.empty() || prompt.empty() ||
        width == 0 || height == 0) {
        std::fprintf(stderr, "usage: %s --model model.gguf --rgba decoded.rgba --weights weights.f32 --activations trace "
                            "--layer N --width columns --height rows --prompt text\n", argv[0]);
        return 2;
    }

    const std::vector<float> rgba = read_binary<float>(rgba_path);
    const std::vector<float> weights = read_binary<float>(weights_path);
    ggml_vk_astc_activation_trace activations;
    std::string trace_error;
    if (!ggml_vk_astc_load_activation_trace(activation_path, activations, trace_error)) {
        std::fprintf(stderr, "invalid activation trace: %s\n", trace_error.c_str());
        return 2;
    }
    if (rgba.size() != static_cast<size_t>(width) * height * 4 ||
        weights.size() != static_cast<size_t>(width) * height) {
        std::fprintf(stderr, "decoded RGBA and source weight sizes do not match the requested matrix\n");
        return 2;
    }
    const auto weight_minmax = std::minmax_element(weights.begin(), weights.end());
    const float scale = *weight_minmax.second - *weight_minmax.first;
    const float offset = *weight_minmax.first;

    llama_backend_init();
    llama_model * model = llama_model_load_from_file(model_path.c_str(), llama_model_default_params());
    if (model == nullptr || layer >= static_cast<uint32_t>(llama_model_n_layer(model)) ||
        width != static_cast<uint32_t>(llama_model_n_ff(model, layer)) ||
        height != static_cast<uint32_t>(llama_model_n_embd(model))) {
        std::fprintf(stderr, "model or matrix shape is incompatible with the selected FFN-down layer\n");
        if (model) llama_model_free(model);
        llama_backend_free();
        return 2;
    }

    std::vector<llama_token> tokens;
    if (!tokenize(llama_model_get_vocab(model), prompt, tokens) || tokens.empty()) {
        std::fprintf(stderr, "failed to tokenize prompt\n");
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    if (activations.columns != width || activations.samples < tokens.size()) {
        std::fprintf(stderr, "activation trace must contain at least one row per prompt token and %u columns\n", width);
        llama_model_free(model);
        llama_backend_free();
        return 2;
    }
    std::vector<float> override_output(static_cast<size_t>(tokens.size()) * height);
    for (size_t sample = 0; sample < tokens.size(); ++sample) {
        for (uint32_t row = 0; row < height; ++row) {
            float value = 0.0f;
            for (uint32_t column = 0; column < width; ++column) {
                const size_t index = (static_cast<size_t>(row) * width + column) * 4;
                const float latent = (rgba[index] + rgba[index + 1] + rgba[index + 2]) / 3.0f;
                const float activation = activations.values[sample * activations.columns + column];
                value += (scale * latent + offset) * activation;
            }
            override_output[sample * height + row] = value;
        }
    }

    const logits_result reference = run_model(model, tokens, layer, nullptr);
    const logits_result replay = run_model(model, tokens, layer, &override_output);
    if (reference.status != 0 || replay.status != 0 || reference.values.size() != replay.values.size()) {
        std::fprintf(stderr, "model replay failed: reference=%d replay=%d\n", reference.status, replay.status);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    double mse = 0.0, max_abs = 0.0, ref_energy = 0.0;
    for (size_t i = 0; i < reference.values.size(); ++i) {
        const double delta = static_cast<double>(reference.values[i]) - replay.values[i];
        mse += delta * delta;
        max_abs = std::max(max_abs, std::abs(delta));
        ref_energy += static_cast<double>(reference.values[i]) * reference.values[i];
    }
    std::printf("model-replay tokens=%zu vocab=%zu logits-mse=%.8g logits-relative-mse=%.8g max-abs=%.8g\n",
                tokens.size(), reference.values.size(), mse / reference.values.size(),
                mse / std::max(ref_energy, 1e-12), max_abs);

    llama_model_free(model);
    llama_backend_free();
    return 0;
}
