#include "astc-vulkan-hash.h"

#include <cassert>

int main() {
    const char hello[] = "hello";
    assert(astc_vulkan_fnv1a64(hello, 5) == 0xa430d84680aabd0bull);
    assert(astc_vulkan_fnv1a64_tagged(hello, 5) == "fnv1a64-a430d84680aabd0b");
    return 0;
}
