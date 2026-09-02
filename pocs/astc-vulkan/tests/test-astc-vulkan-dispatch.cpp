#include "astc-vulkan-dispatch.h"

#include <cassert>
#include <cstdio>

int main() {
    astc_vulkan_matvec_session session;
    astc_vulkan_tensor_session tensor;
    std::string error;
    const std::vector<uint32_t> spirv;

    assert(!session.init(VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE,
                         UINT32_MAX, tensor, spirv, 1536, 32, 1, error));
    assert(error == "invalid ASTC matvec session configuration");
    std::vector<float> output;
    assert(!session.run({}, {}, output, error));
    assert(error == "invalid ASTC matvec run inputs");

    session.reset();
    std::puts("ASTC Vulkan dispatch error-path contract passed");
    return 0;
}
