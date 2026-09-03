#include "astc-vulkan-input.h"

#include "llama-ext.h"
#include "llama.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

struct trace_capture_params {
    std::string model_path;
    std::string output_path;
    std::string prompt;
    uint32_t layer = 0;
    bool ffn_down_input = false;
    bool ffn_down_output = false;
};

void print_usage(const char * program) {
    std::fprintf(stderr,
                 "usage: %s --model model.gguf --output trace.astc --layer N --prompt text [--ffn-down-input|--ffn-down-output]\n",
                 program);
}

bool parse_u32(const char * text, uint32_t & value) {
    if (text == nullptr || *text == '\0') {
        return false;
    }
    char * end = nullptr;
    const unsigned long parsed = std::strtoul(text, &end, 10);
    if (*end != '\0' || parsed > UINT32_MAX) {
        return false;
    }
    value = static_cast<uint32_t>(parsed);
    return true;
}

bool parse_args(int argc, char ** argv, trace_capture_params & params) {
    for (int i = 1; i < argc; ++i) {
        const char * option = argv[i];
        if (std::strcmp(option, "--help") == 0 || std::strcmp(option, "-h") == 0) {
            return false;
        }
        if (std::strcmp(option, "--ffn-down-input") == 0) {
            params.ffn_down_input = true;
            continue;
        }
        if (std::strcmp(option, "--ffn-down-output") == 0) {
            params.ffn_down_output = true;
            continue;
        }
        if (i + 1 == argc) {
            return false;
        }
        const char * value = argv[++i];
        if (std::strcmp(option, "--model") == 0 || std::strcmp(option, "-m") == 0) {
            params.model_path = value;
        } else if (std::strcmp(option, "--output") == 0 || std::strcmp(option, "-o") == 0) {
            params.output_path = value;
        } else if (std::strcmp(option, "--prompt") == 0 || std::strcmp(option, "-p") == 0) {
            params.prompt = value;
        } else if (std::strcmp(option, "--layer") == 0) {
            if (!parse_u32(value, params.layer)) {
                return false;
            }
        } else {
            return false;
        }
    }
    return !params.model_path.empty() && !params.output_path.empty() && !params.prompt.empty();
}

bool tokenize(const llama_vocab * vocab, const std::string & prompt, std::vector<llama_token> & tokens) {
    const int32_t count = llama_tokenize(vocab, prompt.c_str(), static_cast<int32_t>(prompt.size()),
                                         nullptr, 0, true, true);
    if (count >= 0) {
        return false;
    }
    tokens.resize(static_cast<size_t>(-count));
    return llama_tokenize(vocab, prompt.c_str(), static_cast<int32_t>(prompt.size()),
                          tokens.data(), static_cast<int32_t>(tokens.size()), true, true) >= 0;
}

} // namespace

int main(int argc, char ** argv) {
    trace_capture_params params;
    if (!parse_args(argc, argv, params)) {
        print_usage(argv[0]);
        return 2;
    }

    llama_backend_init();
    // Trace capture is an offline oracle input. Keep the full model on CPU so
    // a limited Vulkan device cannot fail before the requested activation is
    // copied to the host; the isolated ASTC Vulkan replay remains a separate
    // device test.
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(params.model_path.c_str(), model_params);
    if (model == nullptr) {
        std::fprintf(stderr, "failed to load model: %s\n", params.model_path.c_str());
        llama_backend_free();
        return 1;
    }
    if (params.layer > static_cast<uint32_t>(llama_model_n_layer(model))) {
        std::fprintf(stderr, "layer must be in [0, %d]\n", llama_model_n_layer(model));
        llama_model_free(model);
        llama_backend_free();
        return 2;
    }

    std::vector<llama_token> tokens;
    if (!tokenize(llama_model_get_vocab(model), params.prompt, tokens) || tokens.empty()) {
        std::fprintf(stderr, "failed to tokenize prompt\n");
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = std::max<uint32_t>(512, static_cast<uint32_t>(tokens.size()));
    context_params.n_batch = static_cast<uint32_t>(tokens.size());
    context_params.n_ubatch = static_cast<uint32_t>(tokens.size());
    llama_context * context = llama_init_from_model(model, context_params);
    if (context == nullptr) {
        std::fprintf(stderr, "failed to create context\n");
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    if (params.ffn_down_input && params.ffn_down_output) {
        std::fprintf(stderr, "choose only one FFN capture mode\n");
        llama_free(context);
        llama_model_free(model);
        llama_backend_free();
        return 2;
    }
    if (params.ffn_down_input) {
        llama_set_embeddings_ffn_down_inp(context, params.layer, true);
    } else if (params.ffn_down_output) {
        llama_set_embeddings_ffn_down_out(context, params.layer, true);
    } else {
        llama_set_embeddings_layer_inp(context, params.layer, true);
    }
    llama_batch batch = llama_batch_init(static_cast<int32_t>(tokens.size()), 0, 1);
    batch.n_tokens = static_cast<int32_t>(tokens.size());
    for (size_t i = 0; i < tokens.size(); ++i) {
        batch.token[i] = tokens[i];
        batch.pos[i] = static_cast<llama_pos>(i);
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = 0;
    }
    const int32_t decode_status = llama_decode(context, batch);
    if (decode_status != 0) {
        std::fprintf(stderr, "llama_decode failed: %d\n", decode_status);
        llama_batch_free(batch);
        llama_free(context);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    const int32_t columns = params.ffn_down_input ? llama_model_n_ff(model, params.layer) : llama_model_n_embd(model);
    const float * layer_input = nullptr;
    if (params.ffn_down_input) {
        layer_input = llama_get_embeddings_ffn_down_inp(context, params.layer);
    } else if (params.ffn_down_output) {
        layer_input = llama_get_embeddings_ffn_down_out(context, params.layer);
    } else {
        layer_input = llama_get_embeddings_layer_inp(context, params.layer);
    }
    if (columns <= 0 || layer_input == nullptr) {
        std::fprintf(stderr, "failed to obtain requested activation capture\n");
        llama_batch_free(batch);
        llama_free(context);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }
    ggml_vk_astc_activation_trace trace;
    trace.samples = static_cast<uint32_t>(tokens.size());
    trace.columns = static_cast<uint32_t>(columns);
    trace.values.assign(layer_input, layer_input + static_cast<size_t>(trace.samples) * trace.columns);
    std::string error;
    const bool write_ok = ggml_vk_astc_write_activation_trace(params.output_path, trace, error);
    if (!write_ok) {
        std::fprintf(stderr, "%s\n", error.c_str());
    } else {
        std::printf("captured %u %s activations for layer-%u with %u columns to %s\n",
                    trace.samples,
                    params.ffn_down_input ? "ffn-down-input" : (params.ffn_down_output ? "ffn-down-output" : "layer-input"),
                    params.layer, trace.columns, params.output_path.c_str());
    }

    llama_batch_free(batch);
    llama_free(context);
    llama_model_free(model);
    llama_backend_free();
    return write_ok ? 0 : 1;
}
