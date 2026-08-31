#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>

namespace {

double quadratic_error(const std::array<double, 2> & reference,
                       const std::array<double, 2> & candidate,
                       const std::array<std::array<double, 2>, 2> & hessian) {
    const std::array<double, 2> error = {
        reference[0] - candidate[0], reference[1] - candidate[1] };
    return error[0] * (hessian[0][0] * error[0] + hessian[0][1] * error[1]) +
           error[1] * (hessian[1][0] * error[0] + hessian[1][1] * error[1]);
}

} // namespace

int main() {
    const std::array<double, 2> reference = { 0.49, 0.45 };
    const std::array<std::array<double, 2>, 2> hessian = {{
        {{ 1.0, 0.9 }},
        {{ 0.9, 1.0 }},
    }};
    const std::array<double, 2> local = { 0.0, 0.0 };

    // Conditional block-LDLQ target update. For a committed first-block
    // error e_C = q_C - W_C, the continuous optimum for the future block is
    // W_F - H_FF^-1 H_FC e_C. The ASTC candidate is then chosen around that
    // shifted target, rather than around the original target.
    const double first_error = local[0] - reference[0];
    const double regenerated_target = reference[1] - hessian[1][0] / hessian[1][1] * first_error;
    const std::array<double, 2> regenerated = {
        local[0], regenerated_target >= 0.5 ? 1.0 : 0.0 };
    const double local_error = quadratic_error(reference, local, hessian);
    const double regenerated_error = quadratic_error(reference, regenerated, hessian);

    assert(regenerated_target > 0.5);
    assert(regenerated[1] == 1.0);
    assert(regenerated_error < local_error);
    std::printf("block-ldlq regenerated-target=%.8f local-error=%.8f regenerated-error=%.8f\n",
                regenerated_target, local_error, regenerated_error);
    return 0;
}
