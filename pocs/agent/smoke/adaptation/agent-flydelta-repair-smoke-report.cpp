#include "agent-flydelta-repair-smoke-report.h"

#include <ostream>

void print_margin(
        std::ostream & output,
        const char * prefix,
        const common_flydelta_decision_margin & margin,
        const common_flydelta_decision_margin * baseline) {
    output << ' ' << prefix << "_available=" << (margin.available ? "yes" : "no");
    if (!margin.available) return;
    output << ' ' << prefix << "_total_delta=" << margin.total_delta()
           << ' ' << prefix << "_normalized_delta=" << margin.normalized_delta()
           << ' ' << prefix << "_positive_logprob=" << margin.positive_total_logprob
           << ' ' << prefix << "_negative_logprob=" << margin.negative_total_logprob
           << ' ' << prefix << "_positive_tokens=" << margin.positive_token_count
           << ' ' << prefix << "_negative_tokens=" << margin.negative_token_count;
    if (baseline && baseline->available) {
        output << ' ' << prefix << "_delta_from_baseline="
               << margin.normalized_delta() - baseline->normalized_delta();
    }
}

void print_representation_diagnostics(
        std::ostream & output,
        const common_flydelta_representation_diagnostics & diagnostics) {
    output << " cosine=" << diagnostics.cosine
           << " progress=" << diagnostics.progress
           << " leakage=" << diagnostics.leakage
           << " shift_norm=" << diagnostics.shift_norm;
}

void print_representation_diagnostics(
        std::ostream & output,
        const common_flydelta_scale_geometry & geometry) {
    output << " cosine=" << geometry.cosine
           << " progress=" << geometry.progress
           << " leakage=" << geometry.leakage
           << " shift_norm=" << geometry.shift_norm;
}
