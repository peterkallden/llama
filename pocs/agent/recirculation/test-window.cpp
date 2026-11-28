#include "window.h"
#include <cassert>

int main() {
    using namespace recirculation;
    const std::vector<float> original{3.0f, 4.0f};
    const std::vector<float> source{0.0f, 2.0f};
    float sn = 0, dn = 0;
    std::vector<float> initial;
    assert(recirculation_mix(original.data(), source.data(), 2, 0.1f, 1.0f, initial, sn, dn));
    assert(initial[0] == 3.0f && initial[1] == 4.5f);
    const auto masked = window_masked_patch(original, source, {0, 1}, 0.1f);
    assert(masked == initial);
    const auto other_mask = window_masked_patch(original, {3.0f, 4.0f}, {1, 0}, 0.1f);
    assert(std::abs(window_patch_displacement_norm(other_mask, original) - 0.5) < 1e-6);
    bool mask_rejected = false;
    try { (void) window_masked_patch(original, source, {1, 0}, 0.1f); }
    catch (const std::invalid_argument &) { mask_rejected = true; }
    assert(mask_rejected);
    mask_rejected = false;
    try { (void) window_masked_patch(original, source, {2, 0}, 0.1f); }
    catch (const std::invalid_argument &) { mask_rejected = true; }
    assert(mask_rejected);
    const auto next = window_next_patch(initial, original, source, 0.1f, 0.5f);
    assert(next == initial);
    const auto changed = window_next_patch(initial, original, {2.0f, 0.0f}, 0.1f, 0.5f);
    assert(changed[0] > initial[0] && changed[1] < initial[1]);
    assert(std::abs(window_patch_displacement_norm(initial, original) - 0.5) < 1e-6);
    assert(std::abs(window_patch_direction_cosine(initial, initial, original) - 1.0) < 1e-6);
    assert(std::abs(window_patch_direction_cosine({3.0f, 4.5f}, {3.0f, 3.5f}, original) + 1.0) < 1e-6);
    bool geometry_rejected = false;
    try { (void) window_patch_direction_cosine(original, initial, original); }
    catch (const std::invalid_argument &) { geometry_rejected = true; }
    assert(geometry_rejected);
    assert(std::abs(*window_downstream_projection_delta({0, 0}, {1, 2}, {0, 1}) - 2.0) < 1e-6);
    assert(!window_downstream_projection_delta({0, 0}, {1, 2}, {0, 0}));
    geometry_rejected = false;
    try { (void) window_downstream_projection_delta({0, 0}, {1}, {1, 0}); }
    catch (const std::invalid_argument &) { geometry_rejected = true; }
    assert(geometry_rejected);
    assert(std::abs(window_entropy({0, 0}) - 1.0) < 1e-12);
    assert(window_entropy({10, 0}) < window_entropy({0, 0}));
    double signal = 0;
    const std::vector<std::vector<float>> curve{{0, 0}, {1, 0}, {1, 1}, {0, 1}, {0, 2}};
    assert(window_select_target({0.1, 0.1, 0.3, 0.3, 0.3}, curve,
        window_target_mode::uncertainty, 0.1, signal) == 2 && signal > 0.19);
    assert(window_select_target({0.1, 0.1, 0.3, 0.3, 0.3}, curve,
        window_target_mode::uncertainty, 0.3, signal) == -1);
    assert(window_select_target({0.1, 0.1, 0.1, 0.1, 0.1}, curve,
        window_target_mode::curvature, 0.5, signal) == 2 && signal > 1);
    bool rejected = false;
    try { window_next_patch(initial, original, {0, 0}, 0.1f, 0.5f); }
    catch (const std::runtime_error &) { rejected = true; }
    assert(rejected);
    rejected = false;
    try { window_next_patch(initial, original, source, 0.1f, 0); }
    catch (const std::runtime_error &) { rejected = true; }
    assert(rejected);
}
