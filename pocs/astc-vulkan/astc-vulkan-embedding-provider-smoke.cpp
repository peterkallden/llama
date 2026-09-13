#include "astc-vulkan-embedding-provider.h"

#include <astcenc.h>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

int main(int argc, char ** argv) {
    if (argc == 2) {
        astc_vulkan_embedding_provider provider;
        std::string error;
        if (!provider.prepare(argv[1], error)) {
            std::cerr << "embedding annex check failed: " << error << "\n";
            return 1;
        }
        const int32_t tokens[] = {0, static_cast<int32_t>(provider.vocabulary() - 1)};
        std::vector<float> output(static_cast<size_t>(provider.dimensions()) * 2, 0.0f);
        if (!provider.run("token_embd.weight", tokens, 2, output.data(), provider.dimensions())) {
            std::cerr << "embedding annex decode failed: " << provider.last_error() << "\n";
            return 1;
        }
        for (float value : output) if (!std::isfinite(value)) return 1;
        std::cout << "embedding annex check passed: vocab=" << provider.vocabulary()
                  << " dimensions=" << provider.dimensions()
                  << " tiles/token=" << provider.tiles_per_token()
                  << " payload_bytes=" << provider.payload_bytes() << "\n";
        return 0;
    }
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() / "astc-vulkan-embedding-provider-smoke";
    std::error_code ec;
    fs::remove_all(root, ec);
    fs::create_directories(root, ec);
    if (ec) return 1;

    astcenc_config config{};
    if (astcenc_config_init(ASTCENC_PRF_LDR, 10, 5, 1, ASTCENC_PRE_FAST, 0, &config) != ASTCENC_SUCCESS) return 1;
    astcenc_context * context = nullptr;
    if (astcenc_context_alloc(&config, 1, &context) != ASTCENC_SUCCESS) return 1;
    std::vector<float> source(10 * 5 * 4, 0.0f);
    for (uint32_t x = 0; x < 10; ++x) {
        const float value = 0.15f + 0.07f * static_cast<float>(x);
        for (uint32_t y = 0; y < 5; ++y) {
            const size_t pixel = (static_cast<size_t>(y) * 10 + x) * 4;
            source[pixel + 0] = source[pixel + 1] = source[pixel + 2] = source[pixel + 3] = value;
        }
    }
    void * source_slice = source.data();
    astcenc_image image{10, 5, 1, ASTCENC_TYPE_F32, &source_slice};
    const astcenc_swizzle swizzle{ASTCENC_SWZ_R, ASTCENC_SWZ_G, ASTCENC_SWZ_B, ASTCENC_SWZ_A};
    std::vector<uint8_t> payload(16);
    if (astcenc_compress_image(context, &image, &swizzle, payload.data(), payload.size(), 0) != ASTCENC_SUCCESS) {
        astcenc_context_free(context); return 1;
    }
    astcenc_context_free(context);

    const fs::path payload_path = root / "embedding-10x5.astcpack";
    const fs::path affine_path = root / "embedding-10x5-affine.bin";
    const fs::path metadata_path = root / "embedding-10x5.astce";
    std::ofstream payload_file(payload_path, std::ios::binary);
    payload_file.write(reinterpret_cast<const char *>(payload.data()), payload.size());
    payload_file.write(reinterpret_cast<const char *>(payload.data()), payload.size());
    payload_file.close();
    std::ofstream affine(affine_path, std::ios::binary);
    const float bias = 0.0f;
    const float scale = 1.0f;
    for (int row = 0; row < 2; ++row) { affine.write(reinterpret_cast<const char *>(&bias), sizeof(bias)); affine.write(reinterpret_cast<const char *>(&scale), sizeof(scale)); }
    affine.close();
    std::ofstream metadata(metadata_path);
    metadata << "version=1\n"
             << "tensor=token_embd.weight\n"
             << "representation=e1-local\n"
             << "footprint=10x5\n"
             << "dimensions=10\n"
             << "vocab=2\n"
             << "payload=embedding-10x5.astcpack\n"
             << "affine=embedding-10x5-affine.bin\n";
    metadata.close();

    astc_vulkan_embedding_provider provider;
    std::string error;
    if (!provider.prepare(root.string(), error) || !provider.is_ready("token_embd.weight", 10, 2)) {
        std::cerr << "provider prepare failed: " << error << "\n";
        fs::remove_all(root, ec); return 1;
    }
    const int32_t tokens[] = {0, 1};
    std::vector<float> output(20, 0.0f);
    if (!provider.run("token_embd.weight", tokens, 2, output.data(), 10)) {
        std::cerr << "provider run failed: " << provider.last_error() << "\n";
        fs::remove_all(root, ec); return 1;
    }
    for (float value : output) if (!std::isfinite(value)) { fs::remove_all(root, ec); return 1; }
    if (provider.native_bind(nullptr, "token_embd.weight")) { fs::remove_all(root, ec); return 1; }
    fs::remove_all(root, ec);
    std::cout << "embedding provider smoke passed (validated E1 payload, CPU decode, hard native fallback)\n";
    return 0;
}
