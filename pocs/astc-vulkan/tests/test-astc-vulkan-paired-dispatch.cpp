#include "astc-vulkan-paired-dispatch.h"

#include <cassert>
#include <string>
#include <vector>

int main() {
    astc_vulkan_paired_matvec_session session;
    std::string error;
    std::vector<float> output;
    assert(!session.ready());
    assert(!session.run({}, {}, output, error));
    assert(error == "invalid paired-D2 ASTC matvec run inputs");
    astc_vulkan_tensor_session tensor;
    assert(!session.init(VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0, tensor,
                         {}, {}, 8, 10, 1, error));
    assert(error == "invalid paired-D2 ASTC matvec session configuration");
    assert(!session.ready());
    assert(!session.record_external(VK_NULL_HANDLE, VK_NULL_HANDLE, 0, 0,
                                    VK_NULL_HANDLE, 0, 0, {}, 0, 1, error));
    assert(error == "invalid paired-D2 external matvec recording inputs");
    return 0;
}
