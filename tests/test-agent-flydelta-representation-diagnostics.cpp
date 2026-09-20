#include "agent/adaptation/flydelta/flydelta-representation-diagnostics.h"

#include <cmath>

#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (false)

static common_flydelta_hidden_state_capture capture(
        const std::vector<float> & values) {
    common_flydelta_hidden_state_capture value;
    value.captured = true;
    value.model_profile_fingerprint = "sha256:model";
    value.capture_layout_revision = "layer-input:v1";
    value.layer_indices = {2, 4};
    value.n_embd = 2;
    value.token_index = 3;
    value.values = values;
    return value;
}

static common_flydelta_behavior_delta delta(
        int32_t layer,
        const std::vector<float> & values) {
    common_flydelta_behavior_delta value;
    value.id = "flydelta://delta/diagnostics";
    value.behavior_key = "structured_tool_selection";
    value.capture_manifest_id = "flydelta://capture/diagnostics";
    value.host_evidence_ref = "evidence:diagnostics";
    value.model_profile_fingerprint = "sha256:model";
    value.execution_context_fingerprint = "sha256:execution-context";
    value.capture_layout_revision = "layer-input:v1";
    value.layer_index = layer;
    value.values = values;
    return value;
}

int main() {
    std::string error;
    common_flydelta_representation_diagnostics diagnostics;

    // At layer 4, shift=(2, 2) and repair delta=(2, 0).
    CHECK(common_flydelta_representation_diagnostics_from_captures(
        capture({0.0f, 0.0f, 1.0f, 1.0f}),
        capture({0.0f, 0.0f, 3.0f, 3.0f}),
        delta(4, {2.0f, 0.0f}), 1024, diagnostics, error));
    CHECK(std::fabs(diagnostics.cosine - 0.7071067f) < 0.0001f);
    CHECK(std::fabs(diagnostics.progress - 1.0f) < 0.0001f);
    CHECK(std::fabs(diagnostics.leakage - 1.0f) < 0.0001f);
    CHECK(std::fabs(diagnostics.shift_norm - 2.8284271f) < 0.0001f);

    // The vector form is the CPU reference oracle for a compact device
    // reduction. It must agree with the capture form without requiring a
    // full capture object or a full device-to-host transfer.
    CHECK(common_flydelta_representation_diagnostics_from_vectors(
        std::vector<float>{0.0f, 0.0f}, std::vector<float>{2.0f, 2.0f},
        delta(4, {2.0f, 0.0f}), diagnostics, error));
    CHECK(std::fabs(diagnostics.cosine - 0.7071067f) < 0.0001f);
    CHECK(std::fabs(diagnostics.progress - 1.0f) < 0.0001f);
    CHECK(std::fabs(diagnostics.leakage - 1.0f) < 0.0001f);

    // A zero shift is valid for an UNKNOWN arm and remains diagnostic zero.
    CHECK(common_flydelta_representation_diagnostics_from_captures(
        capture({0.0f, 0.0f, 1.0f, 1.0f}),
        capture({0.0f, 0.0f, 1.0f, 1.0f}),
        delta(4, {2.0f, 0.0f}), 1024, diagnostics, error));
    CHECK(diagnostics.cosine == 0.0f && diagnostics.progress == 0.0f &&
        diagnostics.leakage == 0.0f && diagnostics.shift_norm == 0.0f);

    auto mismatched = capture({0.0f, 0.0f, 3.0f, 3.0f});
    mismatched.token_index = 4;
    CHECK(!common_flydelta_representation_diagnostics_from_captures(
        capture({0.0f, 0.0f, 1.0f, 1.0f}), mismatched,
        delta(4, {2.0f, 0.0f}), 1024, diagnostics, error));
    CHECK(!common_flydelta_representation_diagnostics_from_vectors(
        std::vector<float>{0.0f}, std::vector<float>{1.0f, 2.0f},
        delta(4, {2.0f, 0.0f}), diagnostics, error));
    return 0;
}
