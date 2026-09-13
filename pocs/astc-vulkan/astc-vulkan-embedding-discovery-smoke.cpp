#include "astc-vulkan-embedding-layout.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <string>
#include <vector>

namespace {

bool check(bool condition, const char * expression) {
    if (!condition) std::fprintf(stderr, "check failed: %s\n", expression);
    return condition;
}

double pair_difference_energy(const std::vector<float> & rows,
                              uint32_t token_count,
                              uint32_t dimensions,
                              const astc_vulkan_embedding_layout & layout) {
    double energy = 0.0;
    const uint32_t values = astc_vulkan_embedding_values_per_tile(layout.profile);
    const uint32_t lanes = layout.profile.lanes_per_texel;
    std::vector<uint32_t> physical_to_logical(dimensions, 0);
    for (uint32_t logical = 0; logical < dimensions; ++logical) {
        physical_to_logical[layout.logical_to_physical[logical]] = logical;
    }
    for (uint32_t tile_base = 0; tile_base < dimensions; tile_base += values) {
        for (uint32_t in_tile = 0; in_tile + 1 < values &&
                                     tile_base + in_tile + 1 < dimensions; in_tile += lanes) {
            for (uint32_t lane_pair = 0; lane_pair + 1 < lanes; lane_pair += 2) {
                const uint32_t physical_a = tile_base + in_tile + lane_pair;
                const uint32_t physical_b = physical_a + 1;
                const uint32_t logical_a = physical_to_logical[physical_a];
                const uint32_t logical_b = physical_to_logical[physical_b];
                for (uint32_t token = 0; token < token_count; ++token) {
                    const float delta = rows[static_cast<size_t>(token) * dimensions + logical_a] -
                                        rows[static_cast<size_t>(token) * dimensions + logical_b];
                    energy += static_cast<double>(delta) * delta;
                }
            }
        }
    }
    return energy;
}

} // namespace

int main() {
    constexpr uint32_t token_count = 64;
    constexpr uint32_t dimensions = 256;
    std::vector<float> rows(static_cast<size_t>(token_count) * dimensions);
    for (uint32_t token = 0; token < token_count; ++token) {
        for (uint32_t dimension = 0; dimension < dimensions; ++dimension) {
            const uint32_t pair = dimension / 2;
            const float latent = std::sin(0.071f * static_cast<float>(token) +
                                          0.37f * static_cast<float>(pair));
            const float noise = 0.002f * std::cos(0.13f * static_cast<float>(token + dimension));
            rows[static_cast<size_t>(token) * dimensions + dimension] =
                (dimension & 1u) ? 0.8f * latent + noise : latent + noise;
        }
    }

    astc_vulkan_embedding_profile profile;
    std::string error;
    if (!check(astc_vulkan_embedding_make_profile(
            astc_vulkan_embedding_representation::kE2LocalLA,
            astc_vulkan_footprint::k8x5, profile, error),
            "make E2 profile")) return 1;

    astc_vulkan_embedding_layout identity;
    identity.profile = profile;
    identity.dimensions = dimensions;
    identity.logical_to_physical.resize(dimensions);
    std::iota(identity.logical_to_physical.begin(), identity.logical_to_physical.end(), 0);
    if (!check(astc_vulkan_embedding_validate_layout(identity, error), "validate identity")) return 1;

    // Deliberately break the known correlated adjacent pairs: this is a
    // negative control for the future global dimension-pair discovery pass.
    astc_vulkan_embedding_layout broken = identity;
    for (uint32_t base = 0; base + 3 < dimensions; base += 4) {
        broken.logical_to_physical[base + 1] = base + 2;
        broken.logical_to_physical[base + 2] = base + 1;
    }
    if (!check(astc_vulkan_embedding_validate_layout(broken, error), "validate broken map")) return 1;

    const double identity_energy = pair_difference_energy(rows, token_count, dimensions, identity);
    const double broken_energy = pair_difference_energy(rows, token_count, dimensions, broken);
    if (!check(identity_energy < broken_energy, "correlated pairing beats negative control")) return 1;

    const auto affine = astc_vulkan_embedding_affine_from_row(
        std::vector<float>(rows.begin(), rows.begin() + dimensions));
    std::vector<float> normalized;
    if (!check(astc_vulkan_embedding_normalize(
            std::vector<float>(rows.begin(), rows.begin() + dimensions), affine, normalized, error),
            "normalize first token row")) return 1;
    float max_abs = 0.0f;
    for (float value : normalized) max_abs = std::max(max_abs, std::abs(value));
    if (!check(max_abs <= 1.00001f, "affine normalization bounds row")) return 1;

    std::printf("embedding-discovery-smoke tokens=%u dimensions=%u e2=8x5 bytes/token=%llu "
                "pair-energy identity=%.6g broken=%.6g ratio=%.6g affine-max=%.6g\n",
                token_count, dimensions,
                static_cast<unsigned long long>(astc_vulkan_embedding_payload_bytes_per_token(identity)),
                identity_energy, broken_energy, identity_energy / broken_energy, max_abs);
    std::puts("embedding discovery smoke passed (pre-codec structural proxy)");
    return 0;
}
