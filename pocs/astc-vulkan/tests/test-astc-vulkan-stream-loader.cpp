#include "astc-vulkan-stream-loader.h"

#include <cassert>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

namespace {

void write_fixture(const std::string & path) {
    std::ofstream file(path, std::ios::binary);
    for (unsigned int i = 0; i < 160; ++i) {
        const unsigned char value = static_cast<unsigned char>(i);
        file.write(reinterpret_cast<const char *>(&value), 1);
    }
    assert(file.good());
}

} // namespace

int main() {
    std::string error;
    astc_vulkan_stream_geometry d1;
    assert(astc_vulkan_make_stream_geometry(
        astc_vulkan_footprint::k8x6, 16, 13, false, d1, error));
    assert(d1.blocks_x == 2 && d1.blocks_y == 3 && d1.payload_bytes == 96);
    std::vector<astc_vulkan_stream_band> bands;
    assert(astc_vulkan_plan_stream(d1, 64, bands, error));
    assert(bands.size() == 2 && bands[0].payload_offset == 0 &&
           bands[0].payload_size == 64 && bands[1].payload_offset == 64 &&
           bands[1].payload_size == 32 && bands[1].logical_row_base == 12 &&
           bands[1].logical_row_count == 1);

    astc_vulkan_stream_geometry d2;
    assert(astc_vulkan_make_stream_geometry(
        astc_vulkan_footprint::k8x5, 16, 21, true, d2, error));
    assert(d2.physical_height == 11 && d2.blocks_x == 2 && d2.blocks_y == 3);
    assert(astc_vulkan_make_stream_band(d2, 1, 1, bands.emplace_back(), error));
    assert(bands.back().physical_y == 5 && bands.back().logical_row_base == 10 &&
           bands.back().logical_row_count == 10);

    const std::string fixture = "astc-vulkan-stream-range-test.bin";
    write_fixture(fixture);
    std::vector<uint8_t> bytes;
    assert(astc_vulkan_read_file_range(fixture, 12, 7, bytes, error));
    assert(bytes.size() == 7 && bytes.front() == 12 && bytes.back() == 18);
    assert(!astc_vulkan_read_file_range(fixture, 159, 2, bytes, error));
    astc_vulkan_stream_payload_reader reader;
    assert(reader.open(fixture, 8, d1, error));
    assert(reader.read_band(bands[0], bytes, error));
    assert(bytes.size() == 64 && bytes.front() == 8 && bytes.back() == 71);
    std::vector<uint8_t> streamed;
    for (const auto & band : bands) {
        std::vector<uint8_t> part;
        assert(reader.read_band(band, part, error));
        streamed.insert(streamed.end(), part.begin(), part.end());
    }
    assert(streamed.size() == d1.payload_bytes);
    for (size_t index = 0; index < streamed.size(); ++index) {
        assert(streamed[index] == static_cast<uint8_t>(index + 8));
    }
    assert(!reader.read_band(astc_vulkan_stream_band{3, 1}, bytes, error));
    reader.close();
    std::remove(fixture.c_str());
    std::puts("ASTC shared stream geometry/range contract passed");
    return 0;
}
