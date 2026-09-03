#include "astc-vulkan-gauge.h"

#include <cassert>
#include <cmath>
#include <string>

int main() {
    using basis = astc_vulkan_gauge_basis;
    const auto ordinary = astc_vulkan_make_gauge_factors(false, true, false, false, false);
    assert(ordinary.size() == 7);
    assert(ordinary[0].gauge == 0.0f);
    assert(ordinary[6].gauge == 0.75f);

    const auto pv = astc_vulkan_make_gauge_factors(false, true, true, true, false);
    assert(pv.size() == 19);
    assert(pv[0].basis == basis::constant);
    assert(pv[1].basis == basis::x_ramp && pv[1].gauge == -0.25f);
    assert(pv.back().basis == basis::saddle && pv.back().gauge == 0.75f);
    assert(std::abs(astc_vulkan_gauge_basis_value(basis::saddle, 0.5f, -0.5f) + 0.25f) < 1e-7f);
    assert(std::string(astc_vulkan_gauge_basis_name(basis::y_ramp)) == "y-ramp");
    return 0;
}
