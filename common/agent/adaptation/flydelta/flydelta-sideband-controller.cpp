#include "agent/adaptation/flydelta/flydelta-sideband-controller.h"

bool common_flydelta_sideband_controller::promote_to_canary(
        const common_flydelta_sideband_manifest & manifest,
        const common_flydelta_promotion_summary & summary,
        const common_flydelta_evaluation_report & evaluation,
        const common_flydelta_promotion_policy & policy,
        bool explicit_host_approval,
        std::string & error) {
    return common_flydelta_promote_verified_sideband(
        registry_, manifest, summary, evaluation, policy,
        explicit_host_approval, error);
}

bool common_flydelta_sideband_controller::activate(
        const std::string & id, bool explicit_host_approval, std::string & error) {
    if (!explicit_host_approval) {
        error = "FlyDelta activation requires explicit host approval";
        return false;
    }
    return registry_.activate(id, error);
}

bool common_flydelta_sideband_controller::retire(
        const std::string & id, std::string & error) {
    return registry_.retire(id, error);
}

bool common_flydelta_sideband_controller::revoke(
        const std::string & id, const std::string & reason, std::string & error) {
    return registry_.revoke(id, reason, error);
}
