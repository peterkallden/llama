#include "astc-vulkan-block-ldlq.h"

#include <cassert>
#include <string>
#include <vector>

int main() {
    assert(std::string(astc_vulkan_ldlq_order_name(astc_vulkan_ldlq_order::forward)) == "forward");
    assert(std::string(astc_vulkan_ldlq_order_name(astc_vulkan_ldlq_order::reverse)) == "reverse");
    assert(std::string(astc_vulkan_ldlq_order_name(astc_vulkan_ldlq_order::pivot)) == "pivot");
    const std::vector<double> gram{ 2.0, 0.0, 0.0, 4.0 };
    const std::vector<double> rhs{ 2.0, 8.0 };
    std::vector<double> output;
    assert(astc_vulkan_ldlq_solve_damped_block(gram, 2, 0, 2, 0.0, rhs, output));
    assert(output.size() == 2 && output[0] == 1.0 && output[1] == 2.0);
    return 0;
}
