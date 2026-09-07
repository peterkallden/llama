#include "astc-vulkan-paired-dispatch.h"
#include "astc-vulkan-paired-layout.h"
#include "astc-vulkan-shared-device.h"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

namespace {

std::vector<uint8_t> read_bytes(const std::string & path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) return {};
    const std::streamsize size = input.tellg();
    if (size <= 0) return {};
    std::vector<uint8_t> result(static_cast<size_t>(size));
    input.seekg(0);
    input.read(reinterpret_cast<char *>(result.data()), size);
    return input ? result : std::vector<uint8_t>();
}

std::vector<uint32_t> read_spirv(const std::string & path) {
    const std::vector<uint8_t> bytes = read_bytes(path);
    if (bytes.empty() || bytes.size() % sizeof(uint32_t) != 0) return {};
    std::vector<uint32_t> result(bytes.size() / sizeof(uint32_t));
    std::memcpy(result.data(), bytes.data(), bytes.size());
    return result;
}

bool parse_u32(const char * value, uint32_t & result) {
    try {
        const unsigned long parsed = std::stoul(value);
        if (parsed == 0 || parsed > UINT32_MAX) return false;
        result = static_cast<uint32_t>(parsed);
        return true;
    } catch (...) { return false; }
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s paired-matvec.comp.spv payload.astc layout-map.bin "
                             "[--width N --height N --samples N --repeats N]\n", argv[0]);
        return 2;
    }
    uint32_t width = 8192, logical_height = 2048, samples = 1, repeats = 8;
    for (int i = 4; i < argc; ++i) {
        if (i + 1 == argc) return 2;
        uint32_t * target = nullptr;
        const std::string option = argv[i++];
        if (option == "--width") target = &width;
        else if (option == "--height") target = &logical_height;
        else if (option == "--samples") target = &samples;
        else if (option == "--repeats") target = &repeats;
        else return 2;
        if (!parse_u32(argv[i], *target)) return 2;
    }

    const std::vector<uint8_t> payload = read_bytes(argv[2]);
    const std::vector<uint8_t> layout = read_bytes(argv[3]);
    const std::vector<uint32_t> spirv = read_spirv(argv[1]);
    const uint32_t expected_storage_height = (logical_height + 1u) / 2u;
    if (payload.empty() || layout.empty() || spirv.empty()) {
        std::fprintf(stderr, "D2 timing smoke input could not be read\n");
        return 2;
    }

    astc_vulkan_shared_device device;
    std::string error;
    if (!device.init(error)) {
        std::fprintf(stderr, "D2 timing smoke skipped: %s\n", error.c_str());
        return 77;
    }
    if (!device.supports(astc_vulkan_footprint::k8x5)) {
        std::fprintf(stderr, "D2 timing smoke skipped: ASTC 8x5 unsupported\n");
        return 77;
    }
    const size_t expected_payload = static_cast<size_t>((width + 7u) / 8u) *
                                    ((expected_storage_height + 4u) / 5u) * 16u;
    const size_t expected_layout = astc_vulkan_paired_layout_bytes(
        astc_vulkan_footprint::k8x5, width, logical_height);
    if (payload.size() != expected_payload || layout.size() != expected_layout) {
        std::fprintf(stderr, "D2 timing smoke geometry mismatch: payload=%zu/%zu layout=%zu/%zu\n",
                     payload.size(), expected_payload, layout.size(), expected_layout);
        return 2;
    }

    astc_vulkan_tensor_record record{
        "d2-timing-smoke", width, logical_height, astc_vulkan_footprint::k8x5,
        0, static_cast<uint64_t>(payload.size())};
    record.representation = astc_vulkan_representation::kPairedD2;
    astc_vulkan_tensor_session tensor;
    if (!tensor.upload(device.physical_device(), device.device(), device.queue(), device.queue_family(),
                       record, {}, payload, error, expected_storage_height)) {
        std::fprintf(stderr, "D2 timing smoke upload failed: %s\n", error.c_str());
        return 1;
    }
    astc_vulkan_paired_matvec_session dispatch;
    if (!dispatch.init(device.physical_device(), device.device(), device.queue(), device.queue_family(),
                       tensor, layout, spirv, width, logical_height, samples, error,
                       astc_vulkan_paired_semantic::direct_rgb)) {
        std::fprintf(stderr, "D2 timing smoke dispatch init failed: %s\n", error.c_str());
        return 1;
    }
    std::vector<float> activations(static_cast<size_t>(samples) * width);
    for (size_t i = 0; i < activations.size(); ++i) {
        activations[i] = 0.5f * std::sin(0.017f * static_cast<float>(i + 1));
    }
    std::vector<float> output;
    if (!dispatch.run(activations, {}, output, error)) {
        std::fprintf(stderr, "D2 timing smoke warmup failed: %s\n", error.c_str());
        return 1;
    }
    double total_gpu_ns = 0.0;
    double total_wall_ns = 0.0;
    uint32_t timed = 0;
    for (uint32_t repeat = 0; repeat < repeats; ++repeat) {
        const auto wall_start = std::chrono::steady_clock::now();
        if (!dispatch.run(activations, {}, output, error)) {
            std::fprintf(stderr, "D2 timing smoke run failed: %s\n", error.c_str());
            return 1;
        }
        total_wall_ns += static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - wall_start).count());
        if (dispatch.last_gpu_time_ns() > 0.0) {
            total_gpu_ns += dispatch.last_gpu_time_ns();
            ++timed;
        }
    }
    std::printf("D2 8x5 paired %ux%u logical, %u samples: GPU %.3f us/dispatch (%u/%u timed)\n",
                width, logical_height, samples,
                timed != 0 ? total_gpu_ns / static_cast<double>(timed) / 1000.0 : 0.0,
                timed, repeats);
    std::printf("D2 8x5 paired host-submit-to-readback %.3f ms/dispatch\n",
                total_wall_ns / static_cast<double>(repeats) / 1.0e6);
    std::printf("D2 8x5 paired timing smoke passed\n");
    return 0;
}
