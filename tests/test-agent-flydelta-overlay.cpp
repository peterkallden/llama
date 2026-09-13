#include "agent/adaptation/flydelta/flydelta-overlay.h"

#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (false)

int main() {
    std::string error;
    common_flydelta_basis_direction direction;
    direction.layer_index = 2;
    direction.values = {1.0f, 2.0f};
    common_flydelta_gate_decision gate;
    gate.apply = true;
    gate.scale = 0.5f;
    common_flydelta_static_overlay overlay;
    CHECK(common_flydelta_compose_static_overlay(
        "flydelta://artifact/1", 2, 4, 1, 3, {direction}, {0.5f}, gate, 1024,
        overlay, error));
    CHECK(overlay.enabled && overlay.data.size() == 6);
    CHECK(overlay.data[2] == 0.25f && overlay.data[3] == 0.5f);
    CHECK(overlay.data[0] == 0.0f && overlay.data[4] == 0.0f);

    common_flydelta_gate_decision no_op;
    CHECK(common_flydelta_compose_static_overlay(
        "", 2, 4, 1, 3, {direction}, {0.5f}, no_op, 1024, overlay, error));
    CHECK(!overlay.enabled && overlay.data.empty());

    auto wrong_layer = direction;
    wrong_layer.layer_index = 3;
    CHECK(!common_flydelta_compose_static_overlay(
        "flydelta://artifact/1", 2, 4, 1, 2, {wrong_layer}, {0.5f}, gate, 1024,
        overlay, error));
    auto wrong_dimension = direction;
    wrong_dimension.values = {1.0f};
    CHECK(!common_flydelta_compose_static_overlay(
        "flydelta://artifact/1", 2, 4, 1, 3, {wrong_dimension}, {0.5f}, gate, 1024,
        overlay, error));
    return 0;
}
