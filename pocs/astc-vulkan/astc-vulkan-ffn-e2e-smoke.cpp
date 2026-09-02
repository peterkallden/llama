#include "astc-vulkan-driver.h"
#include "astc-vulkan-ffn-adapter.h"
#include "astc-vulkan-input.h"
#include "astc-vulkan-resource.h"
#include "astc-vulkan-dispatch.h"

#include <vulkan/vulkan.h>

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
    std::vector<T> values(static_cast<size_t>(size) / sizeof(T));
    file.seekg(0);
    file.read(reinterpret_cast<char *>(values.data()), size);
    return file ? values : std::vector<T>();
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

std::vector<uint32_t> read_spirv(const std::string & path) { return read_binary<uint32_t>(path); }

} // namespace

int main(int argc, char ** argv) {
    std::string shader, payload_path, activation_path, decoded_path, weights_path, metadata_path, reference_output_path;
    uint32_t width = 0, height = 0, max_samples = 0;
    uint8_t footprint = 2;
    for (int i = 1; i < argc; ++i) {
        const std::string option = argv[i];
        if (i + 1 >= argc) break;
        if (option == "--shader") shader = argv[++i];
        else if (option == "--payload") payload_path = argv[++i];
        else if (option == "--activations") activation_path = argv[++i];
        else if (option == "--decoded") decoded_path = argv[++i];
        else if (option == "--weights") weights_path = argv[++i];
        else if (option == "--metadata") metadata_path = argv[++i];
        else if (option == "--reference-output") reference_output_path = argv[++i];
        else if (option == "--width") width = static_cast<uint32_t>(std::stoul(argv[++i]));
        else if (option == "--height") height = static_cast<uint32_t>(std::stoul(argv[++i]));
        else if (option == "--footprint") footprint = static_cast<uint8_t>(std::stoul(argv[++i]));
        else if (option == "--max-samples") max_samples = static_cast<uint32_t>(std::stoul(argv[++i]));
        else { std::fprintf(stderr, "unknown option: %s\n", option.c_str()); return 2; }
    }
    if (shader.empty() || payload_path.empty() || activation_path.empty() || decoded_path.empty() ||
        weights_path.empty() || width == 0 || height == 0 || footprint > 2) return 2;
    const std::vector<uint32_t> spirv = read_spirv(shader);
    const std::vector<uint8_t> payload = read_binary<uint8_t>(payload_path);
    const std::vector<float> decoded = read_binary<float>(decoded_path);
    const std::vector<float> weights = read_binary<float>(weights_path);
    ggml_vk_astc_activation_trace trace;
    ggml_vk_astc_activation_trace reference_output;
    std::string error;
    if (spirv.empty() || payload.empty() || decoded.size() != static_cast<size_t>(width) * height * 4 ||
        weights.size() != static_cast<size_t>(width) * height ||
        !ggml_vk_astc_load_activation_trace(activation_path, trace, error) || trace.columns < width) {
        std::fprintf(stderr, "invalid FFN end-to-end inputs: %s\n", error.c_str()); return 2;
    }
    if (!reference_output_path.empty() &&
        (!ggml_vk_astc_load_activation_trace(reference_output_path, reference_output, error) ||
         reference_output.columns != height || reference_output.samples < trace.samples)) {
        std::fprintf(stderr, "invalid FFN reference output: %s\n", error.c_str()); return 2;
    }
    if (max_samples != 0) trace.samples = std::min(trace.samples, max_samples);
    if (trace.samples == 0) return 2;
    const auto weight_minmax = std::minmax_element(weights.begin(), weights.end());
    float reconstruction_scale = *weight_minmax.second - *weight_minmax.first;
    float reconstruction_scale_a = 0.0f;
    float reconstruction_offset = *weight_minmax.first;
    if (!metadata_path.empty() && !read_decoder_metadata(metadata_path, reconstruction_scale,
                                                         reconstruction_scale_a, reconstruction_offset)) {
        std::fprintf(stderr, "invalid decoder metadata: %s\n", metadata_path.c_str());
        return 2;
    }

    VkInstance instance = VK_NULL_HANDLE;
    const VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO, nullptr, "astc-vulkan-ffn-e2e", 1,
                                "llama.cpp ASTC Vulkan PoC", 1, VK_API_VERSION_1_0};
    const VkInstanceCreateInfo instance_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, nullptr, 0,
                                             &app, 0, nullptr, 0, nullptr};
    if (vkCreateInstance(&instance_info, nullptr, &instance) != VK_SUCCESS) return 77;
    uint32_t device_count = 0;
    vkEnumeratePhysicalDevices(instance, &device_count, nullptr);
    std::vector<VkPhysicalDevice> devices(device_count);
    if (device_count == 0 || vkEnumeratePhysicalDevices(instance, &device_count, devices.data()) != VK_SUCCESS) {
        vkDestroyInstance(instance, nullptr); return 77;
    }
    const VkFormat format = astc_vulkan_vk_format(footprint);
    VkPhysicalDevice physical_device = VK_NULL_HANDLE;
    uint32_t queue_family = UINT32_MAX;
    for (VkPhysicalDevice candidate : devices) {
        if (!astc_vulkan_supports_sampled_transfer(candidate, format)) continue;
        uint32_t count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(candidate, &count, nullptr);
        std::vector<VkQueueFamilyProperties> queues(count);
        vkGetPhysicalDeviceQueueFamilyProperties(candidate, &count, queues.data());
        for (uint32_t q = 0; q < count; ++q) if (queues[q].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            physical_device = candidate; queue_family = q; break;
        }
        if (physical_device != VK_NULL_HANDLE) break;
    }
    if (physical_device == VK_NULL_HANDLE) { vkDestroyInstance(instance, nullptr); return 77; }
    constexpr float priority = 1.0f;
    const VkDeviceQueueCreateInfo queue_info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, nullptr, 0,
                                             queue_family, 1, &priority};
    const VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, nullptr, 0, 1,
                                         &queue_info, 0, nullptr, 0, nullptr, nullptr};
    VkDevice device = VK_NULL_HANDLE;
    if (vkCreateDevice(physical_device, &device_info, nullptr, &device) != VK_SUCCESS) {
        vkDestroyInstance(instance, nullptr); return 77;
    }
    VkQueue queue = VK_NULL_HANDLE; vkGetDeviceQueue(device, queue_family, 0, &queue);
    astc_vulkan_tensor_record record{"blk.0.ffn_down.weight", width, height,
        static_cast<astc_vulkan_footprint>(footprint), 0,
        astc_vulkan_image_bytes(static_cast<astc_vulkan_footprint>(footprint), width, height)};
    record.representation = astc_vulkan_representation::kScalar;
    record.scale_l = reconstruction_scale;
    record.scale_a = reconstruction_scale_a;
    record.offset = reconstruction_offset;
    record.payload_hash64 = astc_vulkan_payload_hash64(payload.data(), payload.size());
    astc_vulkan_manifest manifest;
    manifest.tensors.push_back(record);
    astc_vulkan_ffn_adapter adapter;
    astc_vulkan_ffn_binding binding;
    if (!adapter.prepare(manifest, record.name, width, true, height, binding, error) ||
        !adapter.upload(physical_device, device, queue, queue_family, binding, payload, error)) {
        std::fprintf(stderr, "%s\n", error.c_str()); vkDestroyDevice(device, nullptr); vkDestroyInstance(instance, nullptr); return 1;
    }
    const astc_vulkan_tensor_session & tensor = adapter.session();
    (void) tensor;
    std::vector<float> dispatch_activations(static_cast<size_t>(trace.samples) * width);
    for (uint32_t sample = 0; sample < trace.samples; ++sample) {
        std::copy_n(trace.values.begin() + static_cast<size_t>(sample) * trace.columns,
                    width, dispatch_activations.begin() + static_cast<size_t>(sample) * width);
    }
    astc_vulkan_matvec_session dispatch;
    std::vector<float> actual;
    bool success = dispatch.init(physical_device, device, queue, queue_family, tensor,
                                 spirv, width, height, trace.samples, error) &&
                   dispatch.run(dispatch_activations, binding.reconstruction, actual, error);
    if (!success) std::fprintf(stderr, "%s\n", error.c_str());
    double astc_error = 0.0, source_error = 0.0, source_energy = 0.0;
    double reference_error = 0.0, reference_energy = 0.0;
    for (uint32_t sample = 0; success && sample < trace.samples; ++sample) for (uint32_t row = 0; row < height; ++row) {
        double astc_dot = 0.0, source_dot = 0.0;
        for (uint32_t column = 0; column < width; ++column) {
            const size_t index = (static_cast<size_t>(row) * width + column) * 4;
            const float latent = (decoded[index] + decoded[index + 1] + decoded[index + 2]) / 3.0f;
            const float astc_weight = reconstruction_scale * latent +
                                      reconstruction_scale_a * decoded[index + 3] +
                                      reconstruction_offset;
            const float activation = trace.values[static_cast<size_t>(sample) * trace.columns + column];
            astc_dot += astc_weight * activation;
            source_dot += weights[static_cast<size_t>(row) * width + column] * activation;
        }
        const float gpu = actual[static_cast<size_t>(sample) * height + row];
        astc_error += (gpu - astc_dot) * (gpu - astc_dot);
        source_error += (astc_dot - source_dot) * (astc_dot - source_dot);
        source_energy += source_dot * source_dot;
        if (!reference_output_path.empty()) {
            const float reference = reference_output.values[static_cast<size_t>(sample) * reference_output.columns + row];
            reference_error += (astc_dot - reference) * (astc_dot - reference);
            reference_energy += reference * reference;
        }
    }
    const std::string reference_metric = reference_output_path.empty() ? std::string() :
        (" astc-vs-reference-relative-mse=" + std::to_string(reference_error / std::max(reference_energy, 1e-12)));
    if (success) std::printf("ffn-e2e footprint=%ux%u samples=%u rows=%u columns=%u gpu-vs-cpu-mse=%.8g astc-vs-source-relative-mse=%.8g%s\n",
                             astc_vulkan_format(static_cast<astc_vulkan_footprint>(footprint)).block_width,
                             astc_vulkan_format(static_cast<astc_vulkan_footprint>(footprint)).block_height,
                             trace.samples, height, width, astc_error / (trace.samples * height),
                             source_error / std::max(source_energy, 1e-12),
                             reference_metric.c_str());
    dispatch.reset();
    adapter.reset();
    vkDestroyDevice(device, nullptr); vkDestroyInstance(instance, nullptr);
    return success ? 0 : 1;
}
