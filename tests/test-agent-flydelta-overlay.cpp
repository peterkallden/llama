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

    common_flydelta_basis_direction second_direction;
    second_direction.layer_index = 3;
    second_direction.values = {3.0f, 4.0f};
    common_flydelta_basis_direction duplicate_layer_direction;
    duplicate_layer_direction.layer_index = 2;
    duplicate_layer_direction.values = {1.0f, 1.0f};
    common_flydelta_sparse_overlay sparse;
    CHECK(common_flydelta_compose_sparse_overlay(
        "flydelta://artifact/1", 2, 4, 1, 3,
        {direction, second_direction, duplicate_layer_direction},
        {0.5f, 1.0f, 0.5f}, gate, 1024, sparse, error));
    CHECK(sparse.enabled && sparse.layer_indices == std::vector<uint32_t>({2, 3}));
    CHECK(sparse.data.size() == 4);
    CHECK(sparse.data[0] == 0.5f && sparse.data[1] == 0.75f);
    CHECK(sparse.data[2] == 1.5f && sparse.data[3] == 2.0f);

    common_flydelta_static_overlay expanded;
    CHECK(common_flydelta_expand_sparse_overlay(
        sparse, 2, 4, 1024, expanded, error));
    CHECK(expanded.enabled && expanded.data.size() == overlay.data.size());
    CHECK(expanded.data[0] == 0.0f && expanded.data[1] == 0.0f);
    CHECK(expanded.data[2] == 0.5f && expanded.data[3] == 0.75f);
    CHECK(expanded.data[4] == 1.5f && expanded.data[5] == 2.0f);

    common_flydelta_sparse_overlay second_sparse = sparse;
    second_sparse.artifact_id = "flydelta://artifact/2";
    second_sparse.data[0] += 0.25f;
    common_flydelta_sparse_overlay_batch batch;
    batch.overlays = {sparse, second_sparse};
    CHECK(common_flydelta_sparse_overlay_batch_validate(
        batch, 2, 4, 1024, error));
    std::vector<common_flydelta_static_overlay> expanded_batch;
    CHECK(common_flydelta_expand_sparse_overlay_batch(
        batch, 2, 4, 1024, expanded_batch, error));
    CHECK(expanded_batch.size() == 2);
    CHECK(expanded_batch[0].artifact_id == sparse.artifact_id);
    CHECK(expanded_batch[1].artifact_id == second_sparse.artifact_id);
    CHECK(expanded_batch[0].data == expanded.data);
    CHECK(expanded_batch[1].data[2] == 0.75f);
    CHECK(expanded_batch[1].data[3] == 0.75f);
    auto duplicate_identity = batch;
    duplicate_identity.overlays[1].artifact_id = duplicate_identity.overlays[0].artifact_id;
    CHECK(!common_flydelta_sparse_overlay_batch_validate(
        duplicate_identity, 2, 4, 1024, error));

    auto unsorted = sparse;
    unsorted.layer_indices = {3, 2};
    CHECK(!common_flydelta_sparse_overlay_validate(
        unsorted, 2, 4, 1024, error));

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
