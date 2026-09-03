#include "astc-vulkan-input.h"
#include "astc-vulkan-scheduler-adapter.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
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
}

int main(int argc, char ** argv) {
    std::string shader, manifest, payload, trace_path, tensor = "blk.0.ffn_down.weight";
    astc_vulkan_footprint footprint = astc_vulkan_footprint::k6x6;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string option = argv[i];
        const std::string value = argv[i + 1];
        if (option == "--shader") shader = value;
        else if (option == "--manifest") manifest = value;
        else if (option == "--payload-blob") payload = value;
        else if (option == "--trace") trace_path = value;
        else if (option == "--tensor") tensor = value;
        else if (option == "--footprint") {
            if (value == "4x4") footprint = astc_vulkan_footprint::k4x4;
            else if (value == "5x5") footprint = astc_vulkan_footprint::k5x5;
            else if (value == "6x6") footprint = astc_vulkan_footprint::k6x6;
            else if (value == "8x5") footprint = astc_vulkan_footprint::k8x5;
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
    if (spirv.empty() || manifest.empty() || payload.empty() || trace_path.empty() ||
        !ggml_vk_astc_load_activation_trace(trace_path, trace, error) || trace.samples == 0) return 2;
    astc_vulkan_scheduler_adapter adapter;
    if (!adapter.prepare(manifest, payload, tensor, footprint, error)) {
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
    double energy = 0.0;
    for (float value : output) energy += static_cast<double>(value) * value;
    const auto format = astc_vulkan_format(footprint);
    std::printf("scheduler-adapter format=ASTC-%ux%u tensor=%s samples=%u rows=%u columns=%u "
                "output-values=%zu output-l2=%.8g\n",
                format.block_width, format.block_height, tensor.c_str(), trace.samples,
                adapter.binding().record.height, columns, output.size(), std::sqrt(energy));
    return 0;
}
