#pragma once

#include "agent/adaptation/flydelta/flydelta-capture.h"
#include "agent/adaptation/flydelta/flydelta-evidence.h"
#include "agent/adaptation/flydelta/flydelta-hidden-state-hook.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// A host-captured, redacted difference between aligned baseline and candidate
// activations. This is an input contract for an offline basis builder, not a
// dynamic llama.cpp hook.
struct common_flydelta_behavior_delta {
    int schema_version = 1;
    std::string id;
    common_adaptation_evidence_source source = common_adaptation_evidence_source::tool_repair;
    std::string behavior_key;
    std::string capture_manifest_id;
    std::string host_evidence_ref;
    // Optional admission invariants. They are asserted when supplied by the
    // host, but are deliberately not part of the semantic aggregation key.
    std::string scope_fingerprint;
    std::string tokenizer_fingerprint;
    std::string template_fingerprint;
    std::string generation_semantics_fingerprint;
    std::string model_profile_fingerprint;
    std::string execution_context_fingerprint;
    std::string capture_layout_revision;
    int32_t layer_index = -1;
    std::vector<float> values;
};

bool common_flydelta_behavior_delta_validate(
        const common_flydelta_behavior_delta & delta,
        size_t expected_dimension,
        size_t max_bytes,
        std::string & error);

// Produces one candidate-minus-baseline delta per aligned captured layer.
// Both captures must belong to the manifest's verified execution pair and must
// have identical layout, token and dimensions. No model inference occurs.
bool common_flydelta_behavior_deltas_from_captures(
        const common_flydelta_capture_manifest & manifest,
        const common_flydelta_hidden_state_capture & failed,
        const common_flydelta_hidden_state_capture & repaired,
        const std::string & host_evidence_ref,
        size_t max_capture_bytes,
        size_t max_delta_bytes,
        std::vector<common_flydelta_behavior_delta> & deltas,
        std::string & error);

// Host-certified transition adapter. It checks that the manifest and both
// captures describe the same baseline/candidate relation before delegating to
// the bounded per-layer delta builder above.
bool common_flydelta_behavior_deltas_from_verified_transition(
        const common_flydelta_behavior_transition & transition,
        const common_adaptation_evidence & evidence,
        const common_flydelta_capture_manifest & manifest,
        const common_flydelta_hidden_state_capture & failed,
        const common_flydelta_hidden_state_capture & repaired,
        size_t max_capture_bytes,
        size_t max_delta_bytes,
        std::vector<common_flydelta_behavior_delta> & deltas,
        std::string & error);

struct common_flydelta_basis_config {
    size_t dimension = 0;
    size_t max_directions = 16;
    float cluster_similarity = 0.85f;
    common_adaptation_evidence_source source = common_adaptation_evidence_source::tool_repair;
    std::string behavior_key;
    std::string model_profile_fingerprint;
    std::string execution_context_fingerprint;
    std::string capture_layout_revision;
};

bool common_flydelta_basis_config_validate(
        const common_flydelta_basis_config & config,
        std::string & error);

struct common_flydelta_basis_direction {
    int32_t layer_index = -1;
    std::vector<float> values;
    size_t helped_observations = 0;
    size_t neutral_observations = 0;
    size_t harmed_observations = 0;
};

class common_flydelta_basis_builder {
public:
    explicit common_flydelta_basis_builder(common_flydelta_basis_config config);

    // HELPED can create or update a direction. NEUTRAL/HARMED only update an
    // existing sufficiently similar direction. UNKNOWN is intentionally a
    // no-op and cannot affect the basis.
    bool add(
            const common_flydelta_behavior_delta & delta,
            const common_flydelta_intervention_credit & credit,
            std::string & error);

    const common_flydelta_basis_config & config() const { return config_; }
    const std::vector<common_flydelta_basis_direction> & directions() const { return directions_; }

private:
    common_flydelta_basis_config config_;
    std::vector<common_flydelta_basis_direction> directions_;
};
