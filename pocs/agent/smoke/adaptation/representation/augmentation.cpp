#include "agent/adaptation/flydelta/flydelta-representation-augmentation.h"

#include <iostream>

int main() {
    std::string error;
    common_flydelta_representation_augmentation_config config;
    if (!common_flydelta_representation_augmentation_config_validate(config, error)) {
        std::cerr << error << '\n';
        return 1;
    }
    common_flydelta_representation_latent_delta latent;
    if (!common_flydelta_build_residualized_latent_delta(
            "smoke://donor", 24, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 0.0f},
            {{1.0f, 0.0f, 0.0f}}, config.minimum_residual_norm, latent, error) ||
            !latent.available) {
        std::cerr << error << '\n';
        return 1;
    }
    std::vector<std::vector<float>> controls;
    if (!common_flydelta_propose_representation_augmentation_controls(
            config, controls, error) || controls.size() != config.max_controls) {
        std::cerr << error << '\n';
        return 1;
    }
    std::cout << "flydelta_representation_augmentation_smoke=ok"
              << " controls=" << controls.size()
              << " residual_norm=" << latent.residual_norm << '\n';
    return 0;
}
