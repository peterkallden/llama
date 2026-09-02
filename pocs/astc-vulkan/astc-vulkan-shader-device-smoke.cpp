#include <vulkan/vulkan.h>

#include "astc-vulkan-contract.h"
#include "astc-vulkan-resource.h"

#include <cmath>
#include <cstring>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>
#include <cstdint>

namespace {

constexpr VkDeviceSize kAstcBlockBytes = ggml_vk_astc_4x4_unorm_rgba.block_size_bytes;
constexpr float kExpectedChannel = 0.5f;
constexpr uint32_t kBenchmarkTexelExtent = 192;
constexpr uint32_t kBenchmarkDispatchRepeats = 20;

// A valid 2D LDR ASTC void-extent block.  Bits 10 and 11 are set as required;
// all-ones s/t extents make this a constant-color block.  Each RGBA component
// is UNORM16(0x8000), so texelFetch should return approximately 0.5.
constexpr unsigned char kConstantHalfBlock[kAstcBlockBytes] = {
    0xfc, 0xfd, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80,
};

struct push_constants {
    uint32_t width;
    uint32_t height;
    uint32_t access_pattern;
    float scale_l;
    float scale_a;
    float offset;
};

float half_to_float(uint16_t bits) {
    const uint32_t sign = (bits >> 15u) & 1u;
    const uint32_t exponent = (bits >> 10u) & 0x1fu;
    const uint32_t mantissa = bits & 0x3ffu;
    uint32_t result = sign << 31u;
    if (exponent == 0) {
        if (mantissa != 0) {
            float value = std::ldexp(static_cast<float>(mantissa), -24);
            return sign != 0 ? -value : value;
        }
    } else if (exponent == 31u) {
        result |= 0x7f800000u | (mantissa << 13u);
        float value = 0.0f;
        std::memcpy(&value, &result, sizeof(value));
        return value;
    } else {
        result |= (exponent + (127u - 15u)) << 23u | (mantissa << 13u);
        float value = 0.0f;
        std::memcpy(&value, &result, sizeof(value));
        return value;
    }
    return sign != 0 ? -0.0f : 0.0f;
}

float packed_weight(const std::vector<uint8_t> & payload, uint32_t width,
                    uint32_t row, uint32_t column, const std::string & kind) {
    const bool q4 = kind == "q4";
    const bool q3 = kind == "q3";
    const bool tq1 = kind == "tq1";
    const bool tq2 = kind == "tq2";
    const uint32_t elements = q4 ? 32u : 256u;
    const uint32_t bytes = q4 ? 18u : q3 ? 110u : tq1 ? 54u : 66u;
    const size_t block = (static_cast<size_t>(row) * (width / elements) + column / elements) * bytes;
    if (q4) {
        const uint8_t byte = payload[block + 2u + (column % 16u)];
        const uint32_t code = (column % 32u) < 16u ? byte & 0xfu : byte >> 4u;
        const uint16_t scale_bits = static_cast<uint16_t>(payload[block]) |
                                     static_cast<uint16_t>(payload[block + 1u]) << 8u;
        return static_cast<float>(static_cast<int>(code) - 8) * half_to_float(scale_bits);
    }
    if (tq2) {
        const uint8_t byte = payload[block + column % elements / 4u];
        const uint32_t code = (byte >> ((column % 4u) * 2u)) & 3u;
        const uint16_t scale_bits = static_cast<uint16_t>(payload[block + 64u]) |
                                     static_cast<uint16_t>(payload[block + 65u]) << 8u;
        return static_cast<float>(static_cast<int>(code) - 1) * half_to_float(scale_bits);
    }
    if (tq1) {
        const uint32_t within = column % 256u;
        uint32_t byte_offset = 0, digit = 0;
        if (within < 160u) {
            digit = within / 32u;
            byte_offset = within % 32u;
        } else if (within < 240u) {
            const uint32_t residual = within - 160u;
            digit = residual / 16u;
            byte_offset = 32u + residual % 16u;
        } else {
            const uint32_t residual = within - 240u;
            digit = residual / 4u;
            byte_offset = 48u + residual % 4u;
        }
        const uint32_t trit = (static_cast<uint32_t>(payload[block + byte_offset]) *
                               (digit == 0u ? 1u : digit == 1u ? 3u : digit == 2u ? 9u : digit == 3u ? 27u : 81u)) & 0xffu;
        const uint32_t code = (trit * 3u) >> 8u;
        const uint16_t scale_bits = static_cast<uint16_t>(payload[block + 52u]) |
                                     static_cast<uint16_t>(payload[block + 53u]) << 8u;
        return static_cast<float>(static_cast<int>(code) - 1) * half_to_float(scale_bits);
    }
    const uint32_t within = column % 256u;
    const uint32_t group = within / 128u;
    const uint32_t group_offset = within % 128u;
    const uint32_t pair = group_offset / 32u;
    const uint32_t half = (group_offset % 32u) / 16u;
    const uint32_t lane = group_offset % 16u;
    const uint32_t q_index = group * 32u + half * 16u + lane;
    const uint32_t shift = pair * 2u;
    const uint32_t code = (payload[block + 32u + q_index] >> shift) & 3u;
    const uint32_t high = (payload[block + half * 16u + lane] >> pair) & 1u;
    const uint32_t scale_index = group * 8u + pair * 2u + half;
    const uint32_t scale_group = scale_index / 4u;
    const uint32_t scale_lane = scale_index % 4u;
    const uint32_t source_index = scale_group < 2u ? scale_group * 4u + scale_lane :
                                   scale_group == 2u ? scale_lane : 4u + scale_lane;
    const uint8_t source = payload[block + 96u + source_index];
    const uint32_t nibble = scale_group < 2u ? source & 0xfu : source >> 4u;
    const uint32_t extra = (payload[block + 104u + scale_lane] >> (scale_group * 2u)) & 3u;
    const int scale = static_cast<int>(nibble | (extra << 4u));
    const uint16_t scale_bits = static_cast<uint16_t>(payload[block + 108u]) |
                                 static_cast<uint16_t>(payload[block + 109u]) << 8u;
    return half_to_float(scale_bits) * static_cast<float>(scale - 32) *
           static_cast<float>(static_cast<int>(code) - (high != 0u ? 0 : 4));
}

std::vector<uint32_t> read_spirv(const char * path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        return {};
    }
    const std::streamsize size = input.tellg();
    if (size <= 0 || size % sizeof(uint32_t) != 0) {
        return {};
    }
    std::vector<uint32_t> code(static_cast<size_t>(size) / sizeof(uint32_t));
    input.seekg(0);
    input.read(reinterpret_cast<char *>(code.data()), size);
    return input ? code : std::vector<uint32_t>();
}

template<typename T>
std::vector<T> read_binary(const std::string & path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) return {};
    const std::streamsize size = input.tellg();
    if (size <= 0 || size % static_cast<std::streamsize>(sizeof(T)) != 0) return {};
    std::vector<T> values(static_cast<size_t>(size) / sizeof(T));
    input.seekg(0);
    input.read(reinterpret_cast<char *>(values.data()), size);
    return input ? values : std::vector<T>();
}

} // namespace

int main(int argc, char ** argv) {
    const std::string format_name = argc >= 3 ? argv[2] : "";
    std::string pattern_name = "sequential";
    bool benchmark = false;
    bool matvec = false;
    bool buffer_matvec = false;
    bool q4_matvec = false;
    bool q3_matvec = false;
    bool tq1_matvec = false;
    bool tq2_matvec = false;
    bool sampled_f32 = false;
    std::string payload_path;
    std::string reference_path;
    std::string weights_path;
    uint32_t supplied_width = 0;
    uint32_t supplied_height = 0;
    float scale_l = 1.0f;
    float scale_a = 0.0f;
    float offset = 0.0f;
    uint32_t requested_repeats = 0;
    int index = 3;
    if (index < argc && argv[index][0] != '-') pattern_name = argv[index++];
    while (index < argc) {
        const std::string option = argv[index++];
        if (option == "--benchmark") benchmark = true;
        else if (option == "--matvec") matvec = true;
        else if (option == "--buffer-matvec") { matvec = true; buffer_matvec = true; }
        else if (option == "--q4-matvec") { matvec = true; q4_matvec = true; }
        else if (option == "--q3-matvec") { matvec = true; q3_matvec = true; }
        else if (option == "--tq1-matvec") { matvec = true; tq1_matvec = true; }
        else if (option == "--tq2-matvec") { matvec = true; tq2_matvec = true; }
        else if (option == "--sampled-f32") { matvec = true; sampled_f32 = true; }
        else if ((option == "--payload" || option == "--reference" || option == "--weights" ||
                  option == "--width" || option == "--height" || option == "--scale-l" ||
                  option == "--scale-a" || option == "--offset" || option == "--repeats") && index < argc) {
            const std::string value = argv[index++];
            if (option == "--payload") payload_path = value;
            else if (option == "--reference") reference_path = value;
            else if (option == "--weights") weights_path = value;
            else if (option == "--width") supplied_width = static_cast<uint32_t>(std::stoul(value));
            else if (option == "--height") supplied_height = static_cast<uint32_t>(std::stoul(value));
            else if (option == "--scale-l") scale_l = std::stof(value);
            else if (option == "--scale-a") scale_a = std::stof(value);
            else if (option == "--offset") offset = std::stof(value);
            else requested_repeats = static_cast<uint32_t>(std::stoul(value));
        } else {
            std::fprintf(stderr, "unknown or incomplete option: %s\n", option.c_str());
            return 2;
        }
    }
    if (argc < 3 || (format_name != "4x4" && format_name != "5x5" &&
                     format_name != "6x6" && format_name != "8x6" &&
                     format_name != "8x8") ||
        (pattern_name != "sequential" && pattern_name != "nonlocal") ||
        ((payload_path.empty() != reference_path.empty()) && !buffer_matvec && !sampled_f32) ||
        (buffer_matvec && reference_path.empty()) ||
        (static_cast<int>(q4_matvec) + static_cast<int>(q3_matvec) +
         static_cast<int>(tq1_matvec) + static_cast<int>(tq2_matvec) > 1) ||
        (sampled_f32 && (buffer_matvec || q4_matvec || q3_matvec || tq1_matvec || tq2_matvec)) ||
        ((!payload_path.empty() || buffer_matvec || sampled_f32) && (supplied_width == 0 || supplied_height == 0)) ||
        (matvec && ((!buffer_matvec && !sampled_f32 && payload_path.empty()) || pattern_name != "sequential"))) {
        std::fprintf(stderr,
                     "usage: %s <validation.spv> <4x4|5x5|6x6|8x6|8x8> "
                     "[sequential|nonlocal] [--benchmark] "
                     "[--payload astc.bin --reference decoded-rgba-f32.bin --width N --height N] "
                     "[--matvec|--buffer-matvec|--q4-matvec|--q3-matvec|--tq1-matvec|--tq2-matvec|--sampled-f32 --weights weights-f32.bin "
                     "--scale-l S --scale-a S --offset B --repeats N]\n",
                     argv[0]);
        return 2;
    }
    const uint32_t access_pattern = pattern_name == "nonlocal" ? 1u : 0u;
    const std::vector<uint32_t> spirv = read_spirv(argv[1]);
    if (spirv.empty()) {
        std::fprintf(stderr, "ASTC shader smoke skipped: SPIR-V file unavailable\n");
        return 77;
    }

    const VkApplicationInfo application_info{
        VK_STRUCTURE_TYPE_APPLICATION_INFO, nullptr, "astc-vulkan-shader-device-smoke", 1,
        "llama.cpp ASTC Vulkan PoC", 1, VK_API_VERSION_1_0,
    };
    const VkInstanceCreateInfo instance_info{
        VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, nullptr, 0, &application_info,
        0, nullptr, 0, nullptr,
    };
    VkInstance instance = VK_NULL_HANDLE;
    if (vkCreateInstance(&instance_info, nullptr, &instance) != VK_SUCCESS) {
        return 77;
    }
    uint32_t device_count = 0;
    if (vkEnumeratePhysicalDevices(instance, &device_count, nullptr) != VK_SUCCESS || device_count == 0) {
        vkDestroyInstance(instance, nullptr);
        return 77;
    }
    std::vector<VkPhysicalDevice> devices(device_count);
    vkEnumeratePhysicalDevices(instance, &device_count, devices.data());
    const VkFormat format = format_name == "4x4" ? VK_FORMAT_ASTC_4x4_UNORM_BLOCK :
                            format_name == "5x5" ? VK_FORMAT_ASTC_5x5_UNORM_BLOCK :
                            format_name == "6x6" ? VK_FORMAT_ASTC_6x6_UNORM_BLOCK :
                            format_name == "8x6" ? VK_FORMAT_ASTC_8x6_UNORM_BLOCK :
                                                   VK_FORMAT_ASTC_8x8_UNORM_BLOCK;
    const VkFormat image_format = sampled_f32 ? VK_FORMAT_R32_SFLOAT : format;
    VkPhysicalDevice physical_device = VK_NULL_HANDLE;
    uint32_t queue_family = UINT32_MAX;
    uint32_t timestamp_valid_bits = 0;
    for (VkPhysicalDevice device : devices) {
        if (!astc_vulkan_supports_sampled_transfer(device, image_format)) {
            continue;
        }
        uint32_t queue_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_count, nullptr);
        std::vector<VkQueueFamilyProperties> queues(queue_count);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_count, queues.data());
        for (uint32_t i = 0; i < queue_count; ++i) {
            if ((queues[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0) {
                physical_device = device;
                queue_family = i;
                timestamp_valid_bits = queues[i].timestampValidBits;
                break;
            }
        }
        if (physical_device != VK_NULL_HANDLE) {
            break;
        }
    }
    if (physical_device == VK_NULL_HANDLE) {
        vkDestroyInstance(instance, nullptr);
        return 77;
    }

    constexpr float queue_priority = 1.0f;
    const VkDeviceQueueCreateInfo queue_info{
        VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, nullptr, 0, queue_family, 1,
        &queue_priority,
    };
    const VkDeviceCreateInfo device_info{
        VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, nullptr, 0, 1, &queue_info,
        0, nullptr, 0, nullptr, nullptr,
    };
    VkDevice device = VK_NULL_HANDLE;
    if (vkCreateDevice(physical_device, &device_info, nullptr, &device) != VK_SUCCESS) {
        vkDestroyInstance(instance, nullptr);
        return 77;
    }
    VkQueue queue = VK_NULL_HANDLE;
    vkGetDeviceQueue(device, queue_family, 0, &queue);

    const uint32_t block_extent = format_name == "4x4" ? 4 :
                                  format_name == "5x5" ? 5 :
                                  format_name == "6x6" ? 6 : 8;
    const uint32_t block_height = format_name == "8x6" ? 6 : block_extent;
    const uint32_t width = supplied_width != 0 ? supplied_width :
                           benchmark ? kBenchmarkTexelExtent : block_extent;
    const uint32_t height = supplied_height != 0 ? supplied_height :
                            (benchmark ? kBenchmarkTexelExtent : block_height);
    const uint64_t block_count = ggml_vk_astc_image_block_count(
        format_name == "4x4" ? ggml_vk_astc_4x4_unorm_rgba :
        format_name == "5x5" ? ggml_vk_astc_5x5_unorm_rgba :
        format_name == "6x6" ? ggml_vk_astc_6x6_unorm_rgba :
        format_name == "8x6" ? ggml_vk_astc_8x6_unorm_rgba : ggml_vk_astc_8x8_unorm_rgba,
        width, height);
    const bool packed_matvec = q4_matvec || q3_matvec || tq1_matvec || tq2_matvec;
    const char * path_name = buffer_matvec ? "buffer" : q4_matvec ? "Q4_0" :
                              q3_matvec ? "Q3_K" : tq1_matvec ? "TQ1_0" :
                              tq2_matvec ? "TQ2_0" : sampled_f32 ? "sampled-F32" : "ASTC";
    const uint32_t packed_block_elements = q4_matvec ? 32u : 256u;
    const uint32_t packed_block_bytes = q4_matvec ? 18u : q3_matvec ? 110u : tq1_matvec ? 54u : 66u;
    VkDeviceSize staging_bytes = block_count * kAstcBlockBytes;
    const uint32_t dispatch_repeats = requested_repeats != 0 ? requested_repeats :
                                      (benchmark ? kBenchmarkDispatchRepeats : 1);
    const std::vector<uint8_t> payload = payload_path.empty() ? std::vector<uint8_t>() :
                                         read_binary<uint8_t>(payload_path);
    const std::vector<float> expected_values = reference_path.empty() ? std::vector<float>() :
                                               read_binary<float>(reference_path);
    const std::vector<float> source_weights = weights_path.empty() ? std::vector<float>() :
                                              read_binary<float>(weights_path);
    if (!buffer_matvec && !packed_matvec && !sampled_f32 && !weights_path.empty()) {
        std::fprintf(stderr, "--weights is only valid with a buffer or packed matvec shader\n");
        vkDestroyDevice(device, nullptr);
        vkDestroyInstance(instance, nullptr);
        return 2;
    }
    if ((q4_matvec || tq2_matvec) && source_weights.empty()) {
        std::fprintf(stderr, "packed matvec requires --weights with the FP32 source matrix\n");
        vkDestroyDevice(device, nullptr);
        vkDestroyInstance(instance, nullptr);
        return 2;
    }
    if (sampled_f32 && source_weights.empty()) {
        std::fprintf(stderr, "sampled FP32 matvec requires --weights with the source matrix\n");
        vkDestroyDevice(device, nullptr);
        vkDestroyInstance(instance, nullptr);
        return 2;
    }
    if ((!buffer_matvec && !packed_matvec && !sampled_f32 && !payload_path.empty() && payload.size() != staging_bytes) ||
        (packed_matvec && (width % packed_block_elements != 0 ||
                           payload.size() != static_cast<size_t>(height) *
                               (width / packed_block_elements) * packed_block_bytes)) ||
        (!reference_path.empty() && expected_values.size() != static_cast<size_t>(width) * height * 4)) {
        std::fprintf(stderr, "ASTC shader smoke payload/reference dimensions do not match image extent\n");
        vkDestroyDevice(device, nullptr);
        vkDestroyInstance(instance, nullptr);
        return 2;
    }
    if (!source_weights.empty() && source_weights.size() != static_cast<size_t>(width) * height) {
        std::fprintf(stderr, "ASTC shader smoke source weight dimensions do not match image extent\n");
        vkDestroyDevice(device, nullptr);
        vkDestroyInstance(instance, nullptr);
        return 2;
    }
    astc_vulkan_image_resources image;
    bool success = buffer_matvec || packed_matvec || astc_vulkan_create_sampled_image(
        physical_device, device, image_format, width, height, image);
    if (buffer_matvec) staging_bytes = static_cast<VkDeviceSize>(width) * height * sizeof(float);
    if (packed_matvec) staging_bytes = payload.size();
    if (sampled_f32) staging_bytes = static_cast<VkDeviceSize>(width) * height * sizeof(float);
    VkBuffer staging_buffer = VK_NULL_HANDLE;
    VkDeviceMemory staging_memory = VK_NULL_HANDLE;
    VkBuffer output_buffer = VK_NULL_HANDLE;
    VkDeviceMemory output_memory = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptor_layout = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkShaderModule shader_module = VK_NULL_HANDLE;
    VkCommandPool command_pool = VK_NULL_HANDLE;
    VkQueryPool query_pool = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    do {
        if (!success) break;
        const VkBufferCreateInfo staging_info{
            VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, nullptr, 0, staging_bytes,
            ((buffer_matvec || packed_matvec) ? VK_BUFFER_USAGE_STORAGE_BUFFER_BIT : VK_BUFFER_USAGE_TRANSFER_SRC_BIT),
            VK_SHARING_MODE_EXCLUSIVE, 0, nullptr,
        };
        if (vkCreateBuffer(device, &staging_info, nullptr, &staging_buffer) != VK_SUCCESS) break;
        VkMemoryRequirements staging_requirements{};
        vkGetBufferMemoryRequirements(device, staging_buffer, &staging_requirements);
        const uint32_t staging_type = astc_vulkan_find_memory_type(
            physical_device, staging_requirements.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (staging_type == UINT32_MAX) break;
        const VkMemoryAllocateInfo staging_allocate{
            VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, staging_requirements.size, staging_type,
        };
        if (vkAllocateMemory(device, &staging_allocate, nullptr, &staging_memory) != VK_SUCCESS ||
            vkBindBufferMemory(device, staging_buffer, staging_memory, 0) != VK_SUCCESS) break;
        void * mapped = nullptr;
        if (vkMapMemory(device, staging_memory, 0, staging_bytes, 0, &mapped) != VK_SUCCESS) break;
        if (buffer_matvec) {
            float * weights = static_cast<float *>(mapped);
            if (!source_weights.empty()) {
                std::memcpy(weights, source_weights.data(), source_weights.size() * sizeof(float));
            } else {
                for (uint32_t row = 0; row < height; ++row) {
                    for (uint32_t column = 0; column < width; ++column) {
                        const size_t texel = (static_cast<size_t>(row) * width + column) * 4;
                        const float l = (expected_values[texel] + expected_values[texel + 1] +
                                         expected_values[texel + 2]) / 3.0f;
                        weights[static_cast<size_t>(row) * width + column] =
                            scale_l * l + scale_a * expected_values[texel + 3] + offset;
                    }
                }
            }
        } else if (packed_matvec) {
            std::memcpy(mapped, payload.data(), payload.size());
        } else if (sampled_f32) {
            std::memcpy(mapped, source_weights.data(), source_weights.size() * sizeof(float));
        } else if (payload.empty()) {
            for (uint64_t block = 0; block < block_count; ++block) {
                std::memcpy(static_cast<unsigned char *>(mapped) + block * kAstcBlockBytes,
                            kConstantHalfBlock, kAstcBlockBytes);
            }
        } else {
            std::memcpy(mapped, payload.data(), payload.size());
        }
        vkUnmapMemory(device, staging_memory);

        const VkDeviceSize output_bytes = static_cast<VkDeviceSize>(width) * height * 4 * sizeof(float);
        const VkBufferCreateInfo output_info{
            VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, nullptr, 0, output_bytes,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, nullptr,
        };
        if (vkCreateBuffer(device, &output_info, nullptr, &output_buffer) != VK_SUCCESS) break;
        VkMemoryRequirements output_requirements{};
        vkGetBufferMemoryRequirements(device, output_buffer, &output_requirements);
        const uint32_t output_type = astc_vulkan_find_memory_type(
            physical_device, output_requirements.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (output_type == UINT32_MAX) break;
        const VkMemoryAllocateInfo output_allocate{
            VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, output_requirements.size, output_type,
        };
        if (vkAllocateMemory(device, &output_allocate, nullptr, &output_memory) != VK_SUCCESS ||
            vkBindBufferMemory(device, output_buffer, output_memory, 0) != VK_SUCCESS) break;

        const VkDescriptorSetLayoutBinding bindings[2] = {
            { 0, (buffer_matvec || packed_matvec) ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
            { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        };
        const VkDescriptorSetLayoutCreateInfo descriptor_info{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 2, bindings,
        };
        if (vkCreateDescriptorSetLayout(device, &descriptor_info, nullptr, &descriptor_layout) != VK_SUCCESS) break;
        VkDescriptorPoolSize pool_sizes[2]{};
        uint32_t pool_size_count = 0;
        if (buffer_matvec || packed_matvec) {
            pool_sizes[pool_size_count++] = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 };
        } else {
            pool_sizes[pool_size_count++] = { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1 };
            pool_sizes[pool_size_count++] = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1 };
        }
        const VkDescriptorPoolCreateInfo pool_info{
            VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, nullptr, 0, 1, pool_size_count, pool_sizes,
        };
        if (vkCreateDescriptorPool(device, &pool_info, nullptr, &descriptor_pool) != VK_SUCCESS) break;
        const VkDescriptorSetAllocateInfo set_info{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, descriptor_pool, 1,
            &descriptor_layout,
        };
        if (vkAllocateDescriptorSets(device, &set_info, &descriptor_set) != VK_SUCCESS) break;
        const VkDescriptorImageInfo image_descriptor{ image.sampler, image.view,
                                                       VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        const VkDescriptorBufferInfo weight_descriptor{ staging_buffer, 0, staging_bytes };
        const VkDescriptorBufferInfo buffer_descriptor{ output_buffer, 0, output_bytes };
        const VkWriteDescriptorSet writes[2] = {
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptor_set, 0, 0, 1,
              (buffer_matvec || packed_matvec) ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
              (buffer_matvec || packed_matvec) ? nullptr : &image_descriptor,
              (buffer_matvec || packed_matvec) ? &weight_descriptor : nullptr, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptor_set, 1, 0, 1,
              VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &buffer_descriptor, nullptr },
        };
        vkUpdateDescriptorSets(device, 2, writes, 0, nullptr);

        const VkShaderModuleCreateInfo shader_info{
            VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0,
            spirv.size() * sizeof(uint32_t), spirv.data(),
        };
        if (vkCreateShaderModule(device, &shader_info, nullptr, &shader_module) != VK_SUCCESS) break;
        const VkPushConstantRange push_constants{
            VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(::push_constants),
        };
        const VkPipelineLayoutCreateInfo layout_info{
            VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1,
            &descriptor_layout, 1, &push_constants,
        };
        if (vkCreatePipelineLayout(device, &layout_info, nullptr, &pipeline_layout) != VK_SUCCESS) break;
        const VkPipelineShaderStageCreateInfo stage_info{
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
            VK_SHADER_STAGE_COMPUTE_BIT, shader_module, "main", nullptr,
        };
        const VkComputePipelineCreateInfo pipeline_info{
            VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, nullptr, 0, stage_info,
            pipeline_layout, VK_NULL_HANDLE, -1,
        };
        if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info,
                                     nullptr, &pipeline) != VK_SUCCESS) break;

        const VkQueryPoolCreateInfo query_info{
            VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, nullptr, 0,
            VK_QUERY_TYPE_TIMESTAMP, 2, 0,
        };
        if (timestamp_valid_bits != 0 &&
            vkCreateQueryPool(device, &query_info, nullptr, &query_pool) != VK_SUCCESS) break;

        const VkCommandPoolCreateInfo command_pool_info{
            VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, nullptr,
            VK_COMMAND_POOL_CREATE_TRANSIENT_BIT, queue_family,
        };
        if (vkCreateCommandPool(device, &command_pool_info, nullptr, &command_pool) != VK_SUCCESS) break;
        const VkCommandBufferAllocateInfo command_allocate{
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr, command_pool,
            VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1,
        };
        VkCommandBuffer command_buffer = VK_NULL_HANDLE;
        if (vkAllocateCommandBuffers(device, &command_allocate, &command_buffer) != VK_SUCCESS) break;
        const VkCommandBufferBeginInfo begin_info{
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr,
            VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr,
        };
        if (vkBeginCommandBuffer(command_buffer, &begin_info) != VK_SUCCESS) break;
        if (query_pool != VK_NULL_HANDLE) vkCmdResetQueryPool(command_buffer, query_pool, 0, 2);
        const VkImageMemoryBarrier to_transfer{
            VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, nullptr, 0,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_QUEUE_FAMILY_IGNORED,
            VK_QUEUE_FAMILY_IGNORED, image.image, { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
        };
        if (!buffer_matvec && !packed_matvec) vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &to_transfer);
        const VkBufferImageCopy copy_region{
            0, 0, 0, { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 }, { 0, 0, 0 }, { width, height, 1 },
        };
        if (!buffer_matvec && !packed_matvec) vkCmdCopyBufferToImage(command_buffer, staging_buffer, image.image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy_region);
        const VkImageMemoryBarrier to_shader{
            VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, nullptr,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, image.image,
            { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
        };
        if (!buffer_matvec && !packed_matvec) vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &to_shader);
        if (query_pool != VK_NULL_HANDLE) {
            vkCmdWriteTimestamp(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                query_pool, 0);
        }
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline_layout, 0, 1, &descriptor_set, 0, nullptr);
        const ::push_constants dimensions{ width, height, access_pattern, scale_l, scale_a, offset };
        vkCmdPushConstants(command_buffer, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
            0, sizeof(dimensions), &dimensions);
        const uint32_t workgroup_count = matvec ? height : (width * height + 63) / 64;
        for (uint32_t repeat = 0; repeat < dispatch_repeats; ++repeat) {
            vkCmdDispatch(command_buffer, workgroup_count, 1, 1);
        }
        if (query_pool != VK_NULL_HANDLE) {
            vkCmdWriteTimestamp(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                query_pool, 1);
        }
        const VkBufferMemoryBarrier output_barrier{
            VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER, nullptr,
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, output_buffer, 0,
            output_bytes,
        };
        vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 1, &output_barrier, 0, nullptr);
        if (vkEndCommandBuffer(command_buffer) != VK_SUCCESS) break;
        const VkSubmitInfo submit_info{
            VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1,
            &command_buffer, 0, nullptr,
        };
        const VkFenceCreateInfo fence_info{ VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, nullptr, 0 };
        if (vkCreateFence(device, &fence_info, nullptr, &fence) != VK_SUCCESS ||
            vkQueueSubmit(queue, 1, &submit_info, fence) != VK_SUCCESS ||
            vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS) break;
        std::vector<float> values(static_cast<size_t>(width) * height * 4);
        void * output_mapped = nullptr;
        if (vkMapMemory(device, output_memory, 0, output_bytes, 0, &output_mapped) != VK_SUCCESS) break;
        std::memcpy(values.data(), output_mapped, static_cast<size_t>(output_bytes));
        vkUnmapMemory(device, output_memory);
        const size_t validation_count = matvec ? height : values.size();
        for (size_t value_index = 0; value_index < validation_count; ++value_index) {
            float expected = kExpectedChannel;
            if (!expected_values.empty()) {
                if (matvec) {
                    expected = 0.0f;
                    for (uint32_t column = 0; column < width; ++column) {
                        const size_t texel = (value_index * width + column) * 4;
                        float weight = 0.0f;
                        if (packed_matvec) {
                            weight = packed_weight(payload, width, static_cast<uint32_t>(value_index), column,
                                                   q4_matvec ? "q4" : q3_matvec ? "q3" : tq1_matvec ? "tq1" : "tq2");
                        } else if ((buffer_matvec || sampled_f32) && !source_weights.empty()) {
                            weight = source_weights[value_index * width + column];
                        } else {
                            const float l = (expected_values[texel] + expected_values[texel + 1] +
                                             expected_values[texel + 2]) / 3.0f;
                            weight = scale_l * l + scale_a * expected_values[texel + 3] + offset;
                        }
                        const float activation = 0.5f * std::sin(0.017f * (column + 1)) +
                                                 0.2f * std::cos(0.031f * (column + 3));
                        expected += weight * activation;
                    }
                } else {
                    expected = expected_values[value_index];
                }
            }
            const float actual = matvec ? values[value_index * 4] : values[value_index];
            if (!std::isfinite(actual) || std::fabs(actual - expected) > 1e-3f) {
                std::fprintf(stderr,
                             "%s %s %s shader smoke failed: decoded value %.7f\n",
                             path_name, format_name.c_str(), pattern_name.c_str(), actual);
                success = false;
                break;
            }
        }
        if (success && query_pool != VK_NULL_HANDLE) {
            uint64_t timestamps[2] = {};
            const VkResult query_result = vkGetQueryPoolResults(
                device, query_pool, 0, 2, sizeof(timestamps), timestamps,
                sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
            if (query_result == VK_SUCCESS) {
                VkPhysicalDeviceProperties properties{};
                vkGetPhysicalDeviceProperties(physical_device, &properties);
                const double elapsed_ns = static_cast<double>(timestamps[1] - timestamps[0]) *
                                           properties.limits.timestampPeriod;
                const double per_dispatch_ns = elapsed_ns / dispatch_repeats;
                std::printf("%s %s %s shader timestamp %.3f ns (%u dispatches)\n",
                            path_name, format_name.c_str(),
                            pattern_name.c_str(), per_dispatch_ns, dispatch_repeats);
            }
        }
    } while (false);

    if (device != VK_NULL_HANDLE) vkDeviceWaitIdle(device);
    if (fence != VK_NULL_HANDLE) vkDestroyFence(device, fence, nullptr);
    if (query_pool != VK_NULL_HANDLE) vkDestroyQueryPool(device, query_pool, nullptr);
    if (command_pool != VK_NULL_HANDLE) vkDestroyCommandPool(device, command_pool, nullptr);
    if (pipeline != VK_NULL_HANDLE) vkDestroyPipeline(device, pipeline, nullptr);
    if (pipeline_layout != VK_NULL_HANDLE) vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
    if (shader_module != VK_NULL_HANDLE) vkDestroyShaderModule(device, shader_module, nullptr);
    if (descriptor_pool != VK_NULL_HANDLE) vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
    if (descriptor_layout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(device, descriptor_layout, nullptr);
    if (output_buffer != VK_NULL_HANDLE) vkDestroyBuffer(device, output_buffer, nullptr);
    if (output_memory != VK_NULL_HANDLE) vkFreeMemory(device, output_memory, nullptr);
    if (staging_buffer != VK_NULL_HANDLE) vkDestroyBuffer(device, staging_buffer, nullptr);
    if (staging_memory != VK_NULL_HANDLE) vkFreeMemory(device, staging_memory, nullptr);
    astc_vulkan_destroy_sampled_image(device, image);
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);
    if (!success) {
        std::fprintf(stderr, "%s %s %s shader smoke failed\n",
                     path_name, format_name.c_str(), pattern_name.c_str());
        return 1;
    }
    std::printf("%s %s %s shader fetch smoke passed\n",
                path_name, format_name.c_str(), pattern_name.c_str());
    return 0;
}
