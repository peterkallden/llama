#include "flydelta-coefficient-sensitivity.h"
#include <cmath>
#include <algorithm>

// Local sensitivity proposal, not an outcome evaluator.
// s_j = (m(+a e_j)-m(-a e_j))/(2a), using actually applied coefficients.
// Solve (s s^T + ridge B^T B + ridge I) dc = target * s, then
// constrain dc to the local trust ball. The extra identity regularizer keeps
// dependent bases well posed. This is a bounded heuristic after clipping,
// not an exact trust-region solution. It never establishes HELPED or safety.
bool common_flydelta_coefficient_sensitivity_propose(
    const common_flydelta_sensitivity_config & config,
    const std::vector<std::vector<float>> & basis,
    const std::vector<common_flydelta_sensitivity_sample> & samples,
    std::vector<float> & proposal, std::string & reason) {
    proposal.clear();
    reason = "insufficient matched sensitivity probes";
    const size_t n = basis.size();
    if (!config.enabled || !n || n > 16 || basis[0].empty() ||
        !std::isfinite(config.ridge) || config.ridge <= 0 ||
        !std::isfinite(config.trust_radius) || config.trust_radius <= 0 ||
        !std::isfinite(config.target_margin_delta) || config.target_margin_delta <= 0) return false;
    for (const auto & v : basis) {
        if (v.size() != basis[0].size()) return false;
        for (float x : v) if (!std::isfinite(x)) return false;
    }
    std::vector<double> s(n, 0);
    for (size_t j = 0; j < n; ++j) {
        bool found = false;
        for (const auto & a : samples) for (const auto & b : samples) {
            if (a.applied_coefficients.size() != n || b.applied_coefficients.size() != n ||
                !std::isfinite(a.margin) || !std::isfinite(b.margin)) continue;
            const float step = a.applied_coefficients[j];
            if (!std::isfinite(step) || step <= 1e-8f ||
                std::abs(b.applied_coefficients[j] + step) > step * 1e-4f) continue;
            bool coordinate = true;
            for (size_t k = 0; k < n; ++k) {
                if (!std::isfinite(a.applied_coefficients[k]) || !std::isfinite(b.applied_coefficients[k]) ||
                    (k != j && (a.applied_coefficients[k] != 0 || b.applied_coefficients[k] != 0))) coordinate = false;
            }
            if (!coordinate) continue;
            s[j] = (a.margin - b.margin) / (2.0 * step);
            found = true;
            break;
        }
        if (!found) return false;
    }
    double signal = 0;
    for (double x : s) signal += x*x;
    if (!std::isfinite(signal) || signal < 1e-12) { reason = "flat sensitivity"; return false; }
    std::vector<std::vector<double>> matrix(n, std::vector<double>(n+1));
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double gram = 0;
            for (size_t k = 0; k < basis[i].size(); ++k) gram += double(basis[i][k])*basis[j][k];
            matrix[i][j] = s[i]*s[j] + config.ridge*(gram + (i == j ? 1 : 0));
        }
        matrix[i][n] = config.target_margin_delta*s[i];
    }
    for (size_t i = 0; i < n; ++i) {
        size_t pivot = i;
        for (size_t k = i+1; k < n; ++k) if (std::abs(matrix[k][i]) > std::abs(matrix[pivot][i])) pivot = k;
        std::swap(matrix[i], matrix[pivot]);
        const double divisor = matrix[i][i];
        if (!std::isfinite(divisor) || std::abs(divisor) < 1e-12) return false;
        for (size_t j = i; j <= n; ++j) matrix[i][j] /= divisor;
        for (size_t k = 0; k < n; ++k) if (k != i) {
            const double factor = matrix[k][i];
            for (size_t j = i; j <= n; ++j) matrix[k][j] -= factor*matrix[i][j];
        }
    }
    double length = 0;
    for (size_t i = 0; i < n; ++i) length += matrix[i][n]*matrix[i][n];
    length = std::sqrt(length);
    if (!std::isfinite(length) || length < 1e-12) return false;
    const double scale = std::min(1.0, double(config.trust_radius)/length);
    for (size_t i = 0; i < n; ++i) proposal.push_back(float(matrix[i][n]*scale));
    reason = "local sensitivity proposal";
    return true;
}
