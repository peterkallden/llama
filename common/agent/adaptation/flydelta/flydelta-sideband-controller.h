#pragma once

#include "agent/adaptation/flydelta/flydelta-promotion.h"

// Host-owned lifecycle seam for a FlyDelta artifact. The controller makes
// the state transitions explicit but delegates evidence and compatibility
// checks to the existing promotion and registry contracts.
class common_flydelta_sideband_controller {
public:
    explicit common_flydelta_sideband_controller(
            common_flydelta_sideband_registry & registry)
        : registry_(registry) {}

    // Candidate -> canary. This never activates the artifact.
    bool promote_to_canary(
            const common_flydelta_sideband_manifest & manifest,
            const common_flydelta_promotion_summary & summary,
            const common_flydelta_evaluation_report & evaluation,
            const common_flydelta_promotion_policy & policy,
            bool explicit_host_approval,
            std::string & error);

    // Canary -> active. Activation is a separate explicit host decision.
    bool activate(const std::string & id, bool explicit_host_approval,
            std::string & error);
    bool retire(const std::string & id, std::string & error);
    bool revoke(const std::string & id, const std::string & reason,
            std::string & error);

private:
    common_flydelta_sideband_registry & registry_;
};
