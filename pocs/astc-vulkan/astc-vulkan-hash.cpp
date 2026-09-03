#include "astc-vulkan-hash.h"

#include <iomanip>
#include <sstream>

uint64_t astc_vulkan_fnv1a64(const void * data, size_t size) {
    const auto * bytes = static_cast<const uint8_t *>(data);
    uint64_t hash = 1469598103934665603ULL;
    for (size_t index = 0; index < size; ++index) {
        hash ^= bytes[index];
        hash *= 1099511628211ULL;
    }
    return hash;
}

std::string astc_vulkan_fnv1a64_tagged(const void * data, size_t size) {
    std::ostringstream stream;
    stream << "fnv1a64-" << std::hex << std::setw(16) << std::setfill('0')
           << astc_vulkan_fnv1a64(data, size);
    return stream.str();
}
