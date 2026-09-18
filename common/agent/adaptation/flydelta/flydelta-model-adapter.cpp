#include "agent/adaptation/flydelta/flydelta-model-adapter.h"

bool common_flydelta_model_adapter_validate(
        const common_flydelta_model_adapter & adapter,
        std::string & error) {
    error.clear();
    if (!adapter.worker_callback) {
        error = "FlyDelta model adapter has no worker callback";
        return false;
    }
    if (!adapter.capabilities.bootstrap_zoom &&
            !adapter.capabilities.adaptive_alpha &&
            !adapter.capabilities.teacher_forced_margin &&
            !adapter.capabilities.orthogonal_search &&
            !adapter.capabilities.representation_augmentation &&
            !adapter.capabilities.host_verification) {
        error = "FlyDelta model adapter advertises no capabilities";
        return false;
    }
    return true;
}

bool common_flydelta_model_adapter_supports_search(
        const common_flydelta_model_adapter & adapter) {
    return static_cast<bool>(adapter.worker_callback) &&
        (adapter.capabilities.bootstrap_zoom ||
         adapter.capabilities.adaptive_alpha ||
         adapter.capabilities.orthogonal_search ||
         adapter.capabilities.representation_augmentation);
}

