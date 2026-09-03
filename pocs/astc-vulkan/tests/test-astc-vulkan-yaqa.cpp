#include "astc-vulkan-yaqa.h"

#include <cassert>
#include <cmath>

int main() {
    const std::vector<float> error{ 1.0f, 2.0f, 3.0f, 4.0f };
    const std::vector<double> identity{ 1.0, 0.0, 0.0, 1.0 };
    const double frobenius = astc_vulkan_yaqa_two_sided_score(error, 2, 2, identity, identity);
    assert(std::abs(frobenius - 30.0) < 1e-12);

    const std::vector<double> output{ 2.0, 0.0, 0.0, 1.0 };
    const std::vector<double> input{ 3.0, 0.0, 0.0, 1.0 };
    const double weighted = astc_vulkan_yaqa_two_sided_score(error, 2, 2, output, input);
    assert(std::abs(weighted - 54.0) < 1e-12);
    const std::vector<float> input_trace{ 1.0f, 0.0f, 0.0f, 1.0f };
    const std::vector<float> output_trace{ 1.0f, 0.0f, 0.0f, 1.0f };
    const double low_rank = astc_vulkan_yaqa_trace_score(error, 2, 2, input_trace, output_trace, 2);
    assert(std::abs(low_rank - frobenius) < 1e-12);
    assert(std::isnan(astc_vulkan_yaqa_two_sided_score(error, 0, 2, identity, identity)));
    return 0;
}
