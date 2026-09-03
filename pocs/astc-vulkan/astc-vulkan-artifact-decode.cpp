#include <astcenc.h>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <cstdio>
#include <string>
#include <vector>

namespace {
std::vector<uint8_t> read_bytes(const std::string & path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) return {};
    const std::streamsize size = file.tellg();
    if (size <= 0) return {};
    std::vector<uint8_t> result(static_cast<size_t>(size));
    file.seekg(0);
    file.read(reinterpret_cast<char *>(result.data()), size);
    return file ? result : std::vector<uint8_t>();
}

bool footprint(const std::string & value, unsigned int & width, unsigned int & height) {
    if (value == "4x4") { width = 4; height = 4; }
    else if (value == "5x5") { width = 5; height = 5; }
    else if (value == "6x6") { width = 6; height = 6; }
    else if (value == "8x5") { width = 8; height = 5; }
    else if (value == "10x5") { width = 10; height = 5; }
    else if (value == "8x6") { width = 8; height = 6; }
    else if (value == "10x6") { width = 10; height = 6; }
    else if (value == "8x8") { width = 8; height = 8; }
    else if (value == "10x8") { width = 10; height = 8; }
    else return false;
    return true;
}
}

int main(int argc, char ** argv) {
    std::string input, output, footprint_name;
    uint32_t width = 0, height = 0;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string option = argv[i];
        const std::string value = argv[i + 1];
        if (option == "--input") input = value;
        else if (option == "--output") output = value;
        else if (option == "--width") width = static_cast<uint32_t>(std::stoul(value));
        else if (option == "--height") height = static_cast<uint32_t>(std::stoul(value));
        else if (option == "--footprint") footprint_name = value;
        else return 2;
    }
    unsigned int block_width = 0, block_height = 0;
    const std::vector<uint8_t> compressed = read_bytes(input);
    if (input.empty() || output.empty() || width == 0 || height == 0 || compressed.empty() ||
        !footprint(footprint_name, block_width, block_height)) return 2;
    astcenc_config config{};
    const astcenc_error config_status = astcenc_config_init(
        ASTCENC_PRF_LDR, block_width, block_height, 1,
        ASTCENC_PRE_FASTEST, 0, &config);
    if (config_status != ASTCENC_SUCCESS) {
        std::fprintf(stderr, "ASTC config failed: %s\n", astcenc_get_error_string(config_status));
        return 1;
    }
    astcenc_context * context = nullptr;
    const astcenc_error alloc_status = astcenc_context_alloc(&config, 1, &context);
    if (alloc_status != ASTCENC_SUCCESS) {
        std::fprintf(stderr, "ASTC context allocation failed: %s\n",
                     astcenc_get_error_string(alloc_status));
        return 1;
    }
    const astcenc_swizzle swizzle{ASTCENC_SWZ_R, ASTCENC_SWZ_G,
                                  ASTCENC_SWZ_B, ASTCENC_SWZ_A};
    std::ofstream file(output, std::ios::binary);
    if (!file) { astcenc_context_free(context); return 1; }
    const uint32_t blocks_x = (width + block_width - 1) / block_width;
    const uint32_t blocks_y = (height + block_height - 1) / block_height;
    const uint32_t chunk_block_rows = 32;
    for (uint32_t block_y = 0; block_y < blocks_y; block_y += chunk_block_rows) {
        const uint32_t chunk_blocks = std::min(chunk_block_rows, blocks_y - block_y);
        const uint32_t y = block_y * block_height;
        const uint32_t chunk_height = std::min(height - y, chunk_blocks * block_height);
        const size_t compressed_offset = static_cast<size_t>(block_y) * blocks_x * 16;
        const size_t compressed_size = static_cast<size_t>(chunk_blocks) * blocks_x * 16;
        std::vector<float> decoded(static_cast<size_t>(width) * chunk_height * 4);
        void * slice = decoded.data();
        astcenc_image image{width, chunk_height, 1, ASTCENC_TYPE_F32, &slice};
        const astcenc_error status = astcenc_decompress_image(
            context, compressed.data() + compressed_offset, compressed_size,
            &image, &swizzle, 0);
        if (status != ASTCENC_SUCCESS) {
            std::fprintf(stderr, "ASTC decode failed at block row %u: %s\n",
                         block_y, astcenc_get_error_string(status));
            astcenc_context_free(context);
            return 1;
        }
        file.clear();
        file.seekp(static_cast<std::streamoff>(y) * width * 4 * sizeof(float), std::ios::beg);
        if (!file) {
            std::fprintf(stderr, "ASTC output seek failed at row %u\n", y);
            astcenc_context_free(context);
            return 1;
        }
        file.write(reinterpret_cast<const char *>(decoded.data()),
                   static_cast<std::streamsize>(decoded.size() * sizeof(float)));
        if (!file) {
            std::fprintf(stderr, "ASTC output write failed at row %u\n", y);
            astcenc_context_free(context);
            return 1;
        }
    }
    astcenc_context_free(context);
    return 0;
}
