#pragma once

#include "agent/adaptation/flydelta/flydelta-evidence.h"

#include <cstddef>
#include <string>
#include <vector>

// A host-captured, redacted difference between aligned failed and repaired
// activations. This is an input contract for an offline basis builder, not a
// dynamic llama.cpp hook.
struct common_flydelta_repair_delta {
    int schema_version = 1;
    std::string id;
    std::string capture_manifest_id;
    std::string host_evidence_ref;
    std::vector<float> values;
};

bool common_flydelta_repair_delta_validate(
        const common_flydelta_repair_delta & delta,
        size_t expected_dimension,
        size_t max_bytes,
        std::string & error);

struct common_flydelta_basis_config {
    size_t dimension = 0;
    size_t max_directions = 16;
    float cluster_similarity = 0.85f;
};

bool common_flydelta_basis_config_validate(
        const common_flydelta_basis_config & config,
        std::string & error);

struct common_flydelta_basis_direction {
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
            const common_flydelta_repair_delta & delta,
            const common_flydelta_intervention_credit & credit,
            std::string & error);

    const common_flydelta_basis_config & config() const { return config_; }
    const std::vector<common_flydelta_basis_direction> & directions() const { return directions_; }

private:
    common_flydelta_basis_config config_;
    std::vector<common_flydelta_basis_direction> directions_;
};
