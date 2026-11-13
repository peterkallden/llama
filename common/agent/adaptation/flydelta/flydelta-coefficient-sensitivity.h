#pragma once
#include <vector>
#include <string>
#include <functional>

// Optional, local hypothesis generator. No execution, persistence or authority.
struct common_flydelta_sensitivity_config {
    bool enabled = false;
    // Host-owned observer; never grants execution or outcome authority.
    std::function<void(const std::string &, size_t, const std::vector<float> &)> trace;
    float target_margin_delta = 0.1f;
    float ridge = 0.01f;
    float trust_radius = 0.05f;
};
struct common_flydelta_sensitivity_sample {
    std::vector<float> applied_coefficients;
    float margin = 0;
};
// Samples must be matched +/- coordinate probes at the zero center, measured
// against the same host-owned fixture, basis and normalized decision margin.
// Returns false (with reason) when no trustworthy proposal can be formed.
bool common_flydelta_coefficient_sensitivity_propose(
    const common_flydelta_sensitivity_config & config,
    const std::vector<std::vector<float>> & basis,
    const std::vector<common_flydelta_sensitivity_sample> & samples,
    std::vector<float> & proposal, std::string & reason);
