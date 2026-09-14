#pragma once

#include "agent/adaptation/flydelta/flydelta-contracts.h"
#include "agent/adaptation/flydelta/flydelta.h"

#include <string>

struct common_flydelta_gate_config {
    bool enabled = false;
    float min_familiarity = 0.80f;
    float max_novelty = 0.20f;
    float max_scale = 0.25f;
};

struct common_flydelta_gate_request {
    bool explicit_opt_in = false;
    common_flydelta_candidate_status candidate_status = common_flydelta_candidate_status::observed;
    bool basis_available = false;
    float familiarity = 0.0f;
    float novelty = 1.0f;
    float requested_scale = 0.0f;
};

struct common_flydelta_gate_decision {
    bool apply = false;
    float scale = 0.0f;
    std::string reason;
};

bool common_flydelta_gate_config_validate(
        const common_flydelta_gate_config & config,
        std::string & error);

// A valid refusal is represented as apply=false with a stable reason. This
// gate has no side effects and cannot resolve artifacts or mutate inference.
bool common_flydelta_gate_decide(
        const common_flydelta_gate_config & config,
        const common_flydelta_gate_request & request,
        common_flydelta_gate_decision & decision,
        std::string & error);

// Derives only the context-dependent part of a gate request. An empty
// recognition memory therefore produces familiarity=0/novelty=1 and lets the
// normal gate return a stable no-op decision.
bool common_flydelta_gate_request_from_context(
        const common_flydelta_recognition_memory & recognition,
        const common_flydelta_sparse_code & code,
        bool explicit_opt_in,
        common_flydelta_candidate_status candidate_status,
        bool basis_available,
        float requested_scale,
        common_flydelta_gate_request & request,
        std::string & error);
