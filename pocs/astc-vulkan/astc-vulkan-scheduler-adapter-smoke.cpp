#include "astc-vulkan-input.h"
#include "astc-vulkan-scheduler-adapter.h"
#include "astc-vulkan-paired-layout.h"

#include <astcenc.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

namespace {
bool cpu_reference(const astc_vulkan_scheduler_adapter & adapter,
                   astc_vulkan_footprint footprint,
                   const std::vector<float> & activations,
                   std::vector<float> & output, std::string & error) {
    const auto & record = adapter.binding().record;
    if (record.representation != astc_vulkan_representation::kPairedD2 ||
        adapter.payload().empty() || adapter.layout().empty() ||
        adapter.layout().size() % sizeof(uint32_t) != 0 ||
        activations.size() % record.width != 0) {
        error = "CPU ASTC reference requires a ready paired-D2 artifact";
        return false;
    }
    const auto format = astc_vulkan_format(footprint);
    const uint32_t texture_height = astc_vulkan_paired_storage_height(record.height);
    const uint32_t blocks_x = (record.width + format.block_width - 1u) / format.block_width;
    const uint32_t blocks_y = (texture_height + format.block_height - 1u) / format.block_height;
    const size_t expected_payload = static_cast<size_t>(blocks_x) * blocks_y * 16u;
    if (adapter.payload().size() != expected_payload ||
        adapter.layout().size() < astc_vulkan_paired_layout_bytes(footprint, record.width, record.height)) {
        error = "CPU ASTC reference artifact geometry is invalid";
        return false;
    }
    astcenc_config config{};
    if (astcenc_config_init(ASTCENC_PRF_LDR, format.block_width, format.block_height, 1,
                            ASTCENC_PRE_FASTEST, 0, &config) != ASTCENC_SUCCESS) {
        error = "CPU ASTC reference configuration failed";
        return false;
    }
    astcenc_context * context = nullptr;
    if (astcenc_context_alloc(&config, 1, &context) != ASTCENC_SUCCESS) {
        error = "CPU ASTC reference context allocation failed";
        return false;
    }
    std::vector<float> rgba(static_cast<size_t>(record.width) * texture_height * 4u);
    void * slice = rgba.data();
    astcenc_image image{record.width, texture_height, 1, ASTCENC_TYPE_F32, &slice};
    const astcenc_swizzle swizzle{ASTCENC_SWZ_R, ASTCENC_SWZ_G,
                                  ASTCENC_SWZ_B, ASTCENC_SWZ_A};
    const astcenc_error status = astcenc_decompress_image(
        context, adapter.payload().data(), adapter.payload().size(), &image, &swizzle, 0);
    if (status != ASTCENC_SUCCESS) {
        astcenc_context_free(context);
        error = "CPU ASTC reference decode failed";
        return false;
    }
    std::vector<uint32_t> layout(adapter.layout().size() / sizeof(uint32_t));
    std::memcpy(layout.data(), adapter.layout().data(), adapter.layout().size());
    const uint32_t samples = static_cast<uint32_t>(activations.size() / record.width);
    output.assign(static_cast<size_t>(samples) * record.height, 0.0f);
    const auto semantic = adapter.paired_semantic();
    const auto & scales = adapter.row_scales();
    const float scale = adapter.binding().reconstruction.scale_l;
    const float offset = adapter.binding().reconstruction.offset;
    for (uint32_t sample = 0; sample < samples; ++sample) {
        for (uint32_t row = 0; row < record.height; ++row) {
            float sum = 0.0f;
            for (uint32_t column = 0; column < record.width; ++column) {
                const uint32_t texture_row = row / 2u;
                const uint64_t block_index =
                    static_cast<uint64_t>(texture_row / format.block_height) * blocks_x +
                    column / format.block_width;
                astc_vulkan_paired_layout pairing;
                if (!astc_vulkan_paired_layout_get(layout, block_index, pairing)) {
                    astcenc_context_free(context);
                    error = "CPU ASTC reference layout lookup failed";
                    return false;
                }
                const size_t texel =
                    (static_cast<size_t>(texture_row) * record.width + column) * 4u;
                const astc_vulkan_rgba_texel value{
                    rgba[texel], rgba[texel + 1], rgba[texel + 2], rgba[texel + 3]};
                const float latent = astc_vulkan_paired_weight(
                    value, row & 1u, pairing, astc_vulkan_paired_basis::direct, semantic);
                const float row_scale = scales.empty() ? 1.0f : scales[row];
                const float weight = row_scale * (scale * latent + offset);
                sum += weight * activations[static_cast<size_t>(sample) * record.width + column];
            }
            output[static_cast<size_t>(sample) * record.height + row] = sum;
        }
    }
    astcenc_context_free(context);
    return true;
}
}

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

int main(int argc, char ** argv) {
    std::string shader, manifest, payload, model, cache, trace_path, tensor = "blk.0.ffn_down.weight";
    astc_vulkan_footprint footprint = astc_vulkan_footprint::k6x6;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string option = argv[i];
        const std::string value = argv[i + 1];
        if (option == "--shader") shader = value;
        else if (option == "--manifest") manifest = value;
        else if (option == "--payload-blob") payload = value;
        else if (option == "--model") model = value;
        else if (option == "--cache") cache = value;
        else if (option == "--trace") trace_path = value;
        else if (option == "--tensor") tensor = value;
        else if (option == "--footprint") {
            if (value == "4x4") footprint = astc_vulkan_footprint::k4x4;
            else if (value == "5x5") footprint = astc_vulkan_footprint::k5x5;
            else if (value == "6x6") footprint = astc_vulkan_footprint::k6x6;
            else if (value == "6x5") footprint = astc_vulkan_footprint::k6x5;
            else if (value == "8x5") footprint = astc_vulkan_footprint::k8x5;
            else if (value == "10x5") footprint = astc_vulkan_footprint::k10x5;
            else if (value == "8x6") footprint = astc_vulkan_footprint::k8x6;
            else if (value == "10x6") footprint = astc_vulkan_footprint::k10x6;
            else if (value == "8x8") footprint = astc_vulkan_footprint::k8x8;
            else if (value == "10x8") footprint = astc_vulkan_footprint::k10x8;
            else return 2;
        }
        else return 2;
    }
    const std::vector<uint32_t> spirv = read_binary<uint32_t>(shader);
    ggml_vk_astc_activation_trace trace;
    std::string error;
    const bool direct_artifact = !manifest.empty() || !payload.empty();
    const bool cached_artifact = !model.empty();
    if (spirv.empty() || trace_path.empty() || direct_artifact == cached_artifact ||
        (direct_artifact && (manifest.empty() || payload.empty())) ||
        !ggml_vk_astc_load_activation_trace(trace_path, trace, error) || trace.samples == 0) return 2;
    astc_vulkan_scheduler_adapter adapter;
    const bool prepared = cached_artifact ?
        adapter.prepare_from_cache(model, cache.empty() ? "auto" : cache, tensor, footprint, error, true) :
        adapter.prepare(manifest, payload, tensor, footprint, error, true);
    if (!prepared || !adapter.ready()) {
        std::fprintf(stderr, "scheduler adapter prepare failed: %s\n", error.c_str());
        return 1;
    }
    const uint32_t columns = adapter.binding().record.width;
    std::vector<float> activations(static_cast<size_t>(trace.samples) * columns);
    if (trace.columns < columns) return 2;
    for (uint32_t sample = 0; sample < trace.samples; ++sample) {
        std::copy_n(trace.values.begin() + static_cast<size_t>(sample) * trace.columns,
                    columns, activations.begin() + static_cast<size_t>(sample) * columns);
    }
    std::vector<float> output;
    if (!adapter.run(spirv, activations, output, error)) {
        std::fprintf(stderr, "scheduler adapter run failed: %s\n", error.c_str());
        return 1;
    }
    std::vector<float> cpu_output;
    if (!cpu_reference(adapter, footprint, activations, cpu_output, error) ||
        cpu_output.size() != output.size()) {
        std::fprintf(stderr, "CPU ASTC reference failed: %s\n", error.c_str());
        return 1;
    }
    double cpu_gpu_mse = 0.0;
    double cpu_gpu_max_abs = 0.0;
    for (size_t i = 0; i < output.size(); ++i) {
        const double delta = static_cast<double>(output[i]) - cpu_output[i];
        cpu_gpu_mse += delta * delta;
        cpu_gpu_max_abs = std::max(cpu_gpu_max_abs, std::abs(delta));
    }
    cpu_gpu_mse /= std::max(output.size(), size_t(1));
    // CPU and GPU use the same decoded ASTC texels, but the GPU reduces 64
    // products as a tree while the CPU reference accumulates serially. Keep
    // the gate tight enough to catch layout/semantic/lifetime bugs without
    // treating harmless floating-point association differences as failures.
    constexpr double kCpuGpuMseTolerance = 1.0e-8;
    constexpr double kCpuGpuMaxAbsTolerance = 1.0e-3;
    if (cpu_gpu_mse > kCpuGpuMseTolerance || cpu_gpu_max_abs > kCpuGpuMaxAbsTolerance) {
        std::fprintf(stderr, "CPU/Vulkan ASTC reference mismatch: mse=%.8g max-abs=%.8g\n",
                     cpu_gpu_mse, cpu_gpu_max_abs);
        return 1;
    }
    double energy = 0.0;
    for (float value : output) energy += static_cast<double>(value) * value;
    const auto format = astc_vulkan_format(footprint);
    std::printf("scheduler-adapter format=ASTC-%ux%u tensor=%s samples=%u rows=%u columns=%u "
                "output-values=%zu output-l2=%.8g cpu-vs-gpu-mse=%.8g cpu-vs-gpu-max-abs=%.8g\n",
                format.block_width, format.block_height, tensor.c_str(), trace.samples,
                adapter.binding().record.height, columns, output.size(), std::sqrt(energy),
                cpu_gpu_mse, cpu_gpu_max_abs);
    return 0;
}
