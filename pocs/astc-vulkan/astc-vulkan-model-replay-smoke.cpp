#include "astc-vulkan-input.h"
#include "astc-vulkan-paired.h"
#include "astc-vulkan-paired-layout.h"
#if defined(ASTC_VULKAN_MODEL_REPLAY_GPU)
#include "astc-vulkan-scheduler-adapter.h"
#endif

#include "ggml-backend.h"
#include "llama-ext.h"
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>
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

bool read_decoder_metadata(const std::string & path, float & scale_l, float & scale_a, float & offset) {
    std::ifstream file(path);
    if (!file) return false;
    bool have_l = false, have_a = false, have_offset = false;
    std::string line;
    while (std::getline(file, line)) {
        const size_t separator = line.find('=');
        if (separator == std::string::npos) continue;
        const std::string key = line.substr(0, separator);
        const std::string value = line.substr(separator + 1);
        try {
            if (key == "scale_l") { scale_l = std::stof(value); have_l = true; }
            else if (key == "scale_a") { scale_a = std::stof(value); have_a = true; }
            else if (key == "offset") { offset = std::stof(value); have_offset = true; }
        } catch (...) { return false; }
    }
    return have_l && have_a && have_offset;
}

bool tokenize(const llama_vocab * vocab, const std::string & prompt, std::vector<llama_token> & tokens) {
    const int32_t count = llama_tokenize(vocab, prompt.c_str(), static_cast<int32_t>(prompt.size()),
                                         nullptr, 0, true, true);
    if (count >= 0) return false;
    tokens.resize(static_cast<size_t>(-count));
    return llama_tokenize(vocab, prompt.c_str(), static_cast<int32_t>(prompt.size()),
                          tokens.data(), static_cast<int32_t>(tokens.size()), true, true) >= 0;
}

double cross_entropy(const float * logits, size_t n_vocab, llama_token target) {
    float max_logit = logits[0];
    for (size_t i = 1; i < n_vocab; ++i) max_logit = std::max(max_logit, logits[i]);
    double sum_exp = 0.0;
    for (size_t i = 0; i < n_vocab; ++i) {
        sum_exp += std::exp(static_cast<double>(logits[i] - max_logit));
    }
    return static_cast<double>(std::log(sum_exp) + max_logit - logits[target]);
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
        result.n_tokens = tokens.size();
        result.n_vocab = static_cast<size_t>(n_vocab);
        result.values.resize(result.n_tokens * result.n_vocab);
        for (size_t token = 0; token < result.n_tokens; ++token) {
            const float * logits = llama_get_logits_ith(context, static_cast<int32_t>(token));
            if (logits == nullptr) {
                result.values.clear();
                break;
            }
            std::copy(logits, logits + n_vocab, result.values.begin() + token * result.n_vocab);
        }
    }
    llama_batch_free(batch);
    llama_free(context);
    return result;
}

} // namespace

int main(int argc, char ** argv) {
    std::string model_path, rgba_path, weights_path, activation_path, metadata_path, prompt;
    std::string layout_map_path, row_scales_path, representation = "d1", footprint_name = "8x5";
    std::string gpu_shader_path, cache_path, tensor_name;
    uint32_t layer = 0, width = 0, height = 0;
    uint64_t stream_band_bytes = 0;
    bool cpu_only = false, streamed = false;
    for (int i = 1; i < argc; ++i) {
        const std::string option = argv[i];
        if (option == "--cpu-only") {
            cpu_only = true;
            continue;
        }
        if (option == "--streamed") {
            streamed = true;
            continue;
        }
        if (i + 1 >= argc) break;
        if (option == "--model") model_path = argv[++i];
        else if (option == "--rgba") rgba_path = argv[++i];
        else if (option == "--weights") weights_path = argv[++i];
        else if (option == "--activations") activation_path = argv[++i];
        else if (option == "--metadata") metadata_path = argv[++i];
        else if (option == "--layout-map") layout_map_path = argv[++i];
        else if (option == "--row-scales") row_scales_path = argv[++i];
        else if (option == "--gpu-shader") gpu_shader_path = argv[++i];
        else if (option == "--cache") cache_path = argv[++i];
        else if (option == "--tensor") tensor_name = argv[++i];
        else if (option == "--stream-band-bytes") stream_band_bytes = std::stoull(argv[++i]);
        else if (option == "--representation") representation = argv[++i];
        else if (option == "--footprint") footprint_name = argv[++i];
        else if (option == "--prompt") prompt = argv[++i];
        else if (option == "--layer") layer = static_cast<uint32_t>(std::stoul(argv[++i]));
        else if (option == "--width") width = static_cast<uint32_t>(std::stoul(argv[++i]));
        else if (option == "--height") height = static_cast<uint32_t>(std::stoul(argv[++i]));
        else { std::fprintf(stderr, "unknown option: %s\n", option.c_str()); return 2; }
    }
    const bool gpu_requested = !gpu_shader_path.empty() || !cache_path.empty() || !tensor_name.empty();
#if !defined(ASTC_VULKAN_MODEL_REPLAY_GPU)
    if (gpu_requested) {
        std::fprintf(stderr, "GPU model replay is unavailable in this build; enable the experimental scheduler adapter\n");
        return 2;
    }
#endif
    bool paired_d2 = representation == "paired-d2" || representation == "paired-d2-la";
    astc_vulkan_paired_semantic paired_semantic = representation == "paired-d2-la" ?
        astc_vulkan_paired_semantic::luminance_alpha : astc_vulkan_paired_semantic::direct_rgb;
    if ((!gpu_requested && (!paired_d2 && representation != "d1")) || model_path.empty() ||
        (!gpu_requested && (rgba_path.empty() || weights_path.empty())) ||
        (gpu_requested && (gpu_shader_path.empty() || cache_path.empty() || tensor_name.empty())) ||
        activation_path.empty() || prompt.empty() ||
        width == 0 || height == 0) {
        std::fprintf(stderr, "usage: %s --model model.gguf --rgba decoded.rgba --weights weights.f32 --activations trace "
                            "--layer N --width columns --height rows --metadata export.meta --prompt text [--cpu-only]\n"
                            "       %s --model model.gguf --cache cache-dir --gpu-shader paired-matvec.spv "
                            "--tensor name --activations trace --layer N --width columns --height rows --prompt text "
                            "[--streamed --stream-band-bytes N]\n",
                     argv[0], argv[0]);
        return 2;
    }

    const std::vector<float> rgba = gpu_requested ? std::vector<float>() : read_binary<float>(rgba_path);
    const std::vector<float> weights = gpu_requested ? std::vector<float>() : read_binary<float>(weights_path);
    std::vector<uint32_t> layout_map;
    std::vector<float> row_scales;
    if (!gpu_requested && paired_d2) layout_map = read_binary<uint32_t>(layout_map_path);
    if (!gpu_requested && !row_scales_path.empty()) row_scales = read_binary<float>(row_scales_path);
    ggml_vk_astc_activation_trace activations;
    std::string trace_error;
    if (!ggml_vk_astc_load_activation_trace(activation_path, activations, trace_error)) {
        std::fprintf(stderr, "invalid activation trace: %s\n", trace_error.c_str());
        return 2;
    }
    uint32_t physical_width = width, physical_height = height;
    const uint32_t paired_block_width = footprint_name == "6x5" ? 6u :
        footprint_name == "10x5" ? 10u : 8u;
    astc_vulkan_footprint paired_footprint = footprint_name == "6x5" ?
        astc_vulkan_footprint::k6x5 : footprint_name == "10x5" ?
        astc_vulkan_footprint::k10x5 : astc_vulkan_footprint::k8x5;
    if (!gpu_requested && paired_d2) {
        if (layout_map_path.empty()) {
            std::fprintf(stderr, "paired-d2 replay requires --layout-map\n");
            return 2;
        }
        if (!row_scales.empty() && row_scales.size() != height) {
            std::fprintf(stderr, "row-scale sidecar must contain one F32 scale per logical output row\n");
            return 2;
        }
        if (footprint_name != "6x5" && footprint_name != "8x5" && footprint_name != "10x5") {
            std::fprintf(stderr, "paired-d2 replay supports only 6x5, 8x5, and 10x5\n");
            return 2;
        }
        physical_width = ((width + paired_block_width - 1u) / paired_block_width) * paired_block_width;
        physical_height = ((height + 9u) / 10u) * 5u;
        if (rgba.size() != static_cast<size_t>(physical_width) * physical_height * 4 ||
            layout_map.size() != astc_vulkan_paired_layout_word_count(paired_footprint, width, height)) {
            std::fprintf(stderr, "paired-d2 decoded texture/layout-map sizes do not match the logical matrix\n");
            return 2;
        }
    }
    if (!gpu_requested && ((!paired_d2 && rgba.size() != static_cast<size_t>(width) * height * 4) ||
        weights.size() != static_cast<size_t>(width) * height)) {
        std::fprintf(stderr, "decoded RGBA and source weight sizes do not match the requested matrix\n");
        return 2;
    }
    const auto weight_minmax = weights.empty() ? std::pair<std::vector<float>::const_iterator,
                                                            std::vector<float>::const_iterator>(weights.end(), weights.end()) :
                                                 std::minmax_element(weights.begin(), weights.end());
    float scale_l = weights.empty() ? 1.0f : *weight_minmax.second - *weight_minmax.first;
    float scale_a = 0.0f;
    float offset = weights.empty() ? 0.0f : *weight_minmax.first;
    if (!gpu_requested && !metadata_path.empty() && !read_decoder_metadata(metadata_path, scale_l, scale_a, offset)) {
        std::fprintf(stderr, "invalid decoder metadata: %s\n", metadata_path.c_str());
        return 2;
    }

    llama_backend_init();
    llama_model_params model_params = llama_model_default_params();
    ggml_backend_dev_t cpu_devices[2] = { nullptr, nullptr };
    if (cpu_only || gpu_requested) {
        model_params.n_gpu_layers = 0;
        cpu_devices[0] = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (cpu_devices[0] == nullptr) {
            std::fprintf(stderr, "CPU backend is unavailable\n");
            llama_backend_free();
            return 2;
        }
        model_params.devices = cpu_devices;
    }
    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (model == nullptr || layer >= static_cast<uint32_t>(llama_model_n_layer(model)) ||
        width != static_cast<uint32_t>(llama_model_n_ff(model, layer)) ||
        height != static_cast<uint32_t>(llama_model_n_embd(model))) {
        std::fprintf(stderr, "model or matrix shape is incompatible with the selected FFN-down layer\n");
        if (model) llama_model_free(model);
        llama_backend_free();
        return 2;
    }

#if defined(ASTC_VULKAN_MODEL_REPLAY_GPU)
    astc_vulkan_scheduler_adapter gpu_adapter;
    std::vector<uint32_t> gpu_spirv;
    if (gpu_requested) {
        gpu_spirv = read_binary<uint32_t>(gpu_shader_path);
        if (gpu_spirv.empty()) {
            std::fprintf(stderr, "GPU model replay shader is missing or invalid\n");
            llama_model_free(model);
            llama_backend_free();
            return 2;
        }
        std::string gpu_error;
        if (!gpu_adapter.prepare_from_cache(model_path, cache_path, tensor_name, paired_footprint,
                                            gpu_error, true) || !gpu_adapter.ready()) {
            std::fprintf(stderr, "GPU model replay cache prepare failed: %s\n", gpu_error.c_str());
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }
        paired_d2 = gpu_adapter.dispatch_kind() == astc_vulkan_scheduler_dispatch_kind::kD2Paired;
        if (gpu_adapter.binding().record.width != width || gpu_adapter.binding().record.height != height) {
            std::fprintf(stderr, "GPU model replay cache shape does not match requested matrix\n");
            llama_model_free(model);
            llama_backend_free();
            return 2;
        }
        if (paired_d2) paired_semantic = gpu_adapter.paired_semantic();
    }
#endif

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
#if defined(ASTC_VULKAN_MODEL_REPLAY_GPU)
    if (gpu_requested) {
        std::vector<float> gpu_activations(static_cast<size_t>(tokens.size()) * width);
        for (size_t sample = 0; sample < tokens.size(); ++sample) {
            std::copy_n(activations.values.begin() + sample * activations.columns, width,
                        gpu_activations.begin() + sample * width);
        }
        std::string gpu_error;
        const bool dispatched = streamed ?
            gpu_adapter.run_streamed(stream_band_bytes == 0 ? (1u << 20) : stream_band_bytes,
                                     gpu_spirv, gpu_activations, override_output, gpu_error) :
            gpu_adapter.run(gpu_spirv, gpu_activations, override_output, gpu_error);
        if (!dispatched ||
            override_output.size() != static_cast<size_t>(tokens.size()) * height) {
            std::fprintf(stderr, "GPU model replay dispatch failed: %s\n", gpu_error.c_str());
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }
    } else
#endif
    for (size_t sample = 0; sample < tokens.size(); ++sample) {
        for (uint32_t row = 0; row < height; ++row) {
            float value = 0.0f;
            for (uint32_t column = 0; column < width; ++column) {
                size_t index = (static_cast<size_t>(row) * width + column) * 4;
                float latent = 0.0f;
                if (paired_d2) {
                    const uint64_t block_x = column / paired_block_width;
                    const uint64_t block_y = (row / 2u) / 5u;
                    astc_vulkan_paired_layout layout;
                    const uint64_t physical_blocks_x =
                        (static_cast<uint64_t>(width) + paired_block_width - 1u) /
                        paired_block_width;
                    if (!astc_vulkan_paired_layout_get(layout_map,
                            block_y * physical_blocks_x + block_x, layout)) {
                        std::fprintf(stderr, "paired-d2 layout-map lookup failed\n");
                        llama_model_free(model);
                        llama_backend_free();
                        return 2;
                    }
                    const uint32_t texel_y = row / 2u;
                    index = (static_cast<size_t>(texel_y) * physical_width + column) * 4;
                    const astc_vulkan_rgba_texel texel{rgba[index], rgba[index + 1], rgba[index + 2], rgba[index + 3]};
                    latent = astc_vulkan_paired_weight(texel, row & 1u, layout,
                                                        astc_vulkan_paired_basis::direct,
                                                        paired_semantic);
                } else {
                    latent = (rgba[index] + rgba[index + 1] + rgba[index + 2]) / 3.0f;
                }
                const float activation = activations.values[sample * activations.columns + column];
                float weight = scale_l * latent + scale_a * rgba[index + 3] + offset;
                if (!row_scales.empty()) weight *= row_scales[row];
                value += weight * activation;
            }
            override_output[sample * height + row] = value;
        }
    }

    const logits_result reference = run_model(model, tokens, layer, nullptr);
    const logits_result replay = run_model(model, tokens, layer, &override_output);
    if (reference.status != 0 || replay.status != 0 || reference.values.size() != replay.values.size() ||
        reference.n_tokens != replay.n_tokens || reference.n_vocab != replay.n_vocab ||
        reference.n_tokens != tokens.size()) {
        std::fprintf(stderr, "model replay failed: reference=%d replay=%d\n", reference.status, replay.status);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    double mse = 0.0, max_abs = 0.0, ref_energy = 0.0;
    size_t top1_matches = 0;
    for (size_t i = 0; i < reference.values.size(); ++i) {
        const double delta = static_cast<double>(reference.values[i]) - replay.values[i];
        mse += delta * delta;
        max_abs = std::max(max_abs, std::abs(delta));
        ref_energy += static_cast<double>(reference.values[i]) * reference.values[i];
    }
    double reference_loss = 0.0, replay_loss = 0.0;
    size_t loss_tokens = 0;
    for (size_t token = 0; token + 1 < tokens.size(); ++token) {
        const llama_token target = tokens[token + 1];
        if (target < 0 || static_cast<size_t>(target) >= reference.n_vocab) continue;
        const float * ref = reference.values.data() + token * reference.n_vocab;
        const float * got = replay.values.data() + token * replay.n_vocab;
        reference_loss += cross_entropy(ref, reference.n_vocab, target);
        replay_loss += cross_entropy(got, replay.n_vocab, target);
        ++loss_tokens;
    }
    for (size_t token = 0; token < reference.n_tokens; ++token) {
        const float * ref = reference.values.data() + token * reference.n_vocab;
        const float * got = replay.values.data() + token * replay.n_vocab;
        const auto ref_it = std::max_element(ref, ref + reference.n_vocab);
        const auto got_it = std::max_element(got, got + replay.n_vocab);
        top1_matches += static_cast<size_t>((ref_it - ref) == (got_it - got));
    }
    const double normalized_mse = mse / std::max(reference.values.size(), size_t(1));
    const double normalized_relative_mse = mse / std::max(ref_energy, 1e-12);
    std::printf("model-replay tokens=%zu vocab=%zu logits-mse=%.8g logits-relative-mse=%.8g max-abs=%.8g "
                "top1-agreement=%.8g reference-loss=%.8g replay-loss=%.8g loss-delta=%.8g\n",
                reference.n_tokens, reference.n_vocab, normalized_mse, normalized_relative_mse, max_abs,
                static_cast<double>(top1_matches) / std::max(reference.n_tokens, size_t(1)),
                reference_loss / std::max(loss_tokens, size_t(1)), replay_loss / std::max(loss_tokens, size_t(1)),
                (replay_loss - reference_loss) / std::max(loss_tokens, size_t(1)));

    llama_model_free(model);
    llama_backend_free();
    return 0;
}
