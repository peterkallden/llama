#include "astc-vulkan-block-ldlq.h"

#include <cassert>
#include <string>

int main() {
    assert(std::string(astc_vulkan_ldlq_order_name(astc_vulkan_ldlq_order::forward)) == "forward");
    assert(std::string(astc_vulkan_ldlq_order_name(astc_vulkan_ldlq_order::reverse)) == "reverse");
    assert(std::string(astc_vulkan_ldlq_order_name(astc_vulkan_ldlq_order::pivot)) == "pivot");
    return 0;
}
