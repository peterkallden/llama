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
    // A positive correlated input Hessian. Each block has the same local
    // {0, 1} candidates, so the ordinary baseline rounds independently.
    const std::array<double, 2> reference = { 0.49, 0.45 };
    const std::array<std::array<double, 2>, 2> hessian = {{
        {{ 1.0, 0.9 }},
        {{ 0.9, 1.0 }},
    }};
    const std::array<double, 2> greedy = { 0.0, 0.0 };
    const double greedy_error = quadratic_error(reference, greedy, hessian);

    // After committing block 0, the conditional quadratic optimum for the
    // second block is shifted by H21/H22 times the first block's error.
    const double first_error = reference[0] - greedy[0];
    const double feedback_target = reference[1] +
        hessian[1][0] / hessian[1][1] * first_error;
    const std::array<double, 2> feedback = { 0.0, feedback_target >= 0.5 ? 1.0 : 0.0 };
    const double feedback_error = quadratic_error(reference, feedback, hessian);

    assert(greedy[1] == 0.0);
    assert(feedback[1] == 1.0);
    assert(feedback_error < greedy_error);
    std::printf("error-shaping feedback-target=%.8f greedy-error=%.8f feedback-error=%.8f\n",
                feedback_target, greedy_error, feedback_error);
    return 0;
}
