#pragma once

#include "agent/adaptation/adaptation-evidence-routing.h"
#include "agent/adaptation/flydelta/flydelta-contracts.h"
#include "agent/adaptation/flydelta/flydelta-hidden-state-hook.h"
#include "agent/adaptation/learning-transaction.h"

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

// A runtime-discovered, host-owned request to capture representations for a
// verified transition. It is not an activation and contains no raw content.
struct common_flydelta_capture_candidate {
    int schema_version = 1;
    std::string id;
    std::string transaction_id;
    common_adaptation_evidence_source source = common_adaptation_evidence_source::tool_repair;
    std::vector<std::string> evidence_refs;
    std::string model_profile_fingerprint;
    std::string capture_layout_revision;
    bool candidate_ready = false;
};

bool common_flydelta_capture_candidate_validate(
        const common_flydelta_capture_candidate & candidate,
        std::string & error);

// Builds a manifest only after the host has supplied matching, verified
// evidence and an explicit redaction attestation. captured_bytes describes
// the already bounded capture payload; this helper does not capture data.
bool common_flydelta_capture_manifest_from_candidate(
        const common_flydelta_capture_candidate & candidate,
        const common_adaptation_evidence & evidence,
        const std::string & template_fingerprint,
        const std::string & evidence_hash,
        size_t captured_bytes,
        bool redaction_attested,
        common_flydelta_capture_manifest & manifest,
        std::string & error);

// Bridges the generic learning observer's source callback to a bounded
// FlyDelta capture queue. Only source matches marked candidate_ready are
// queued; reflection and user-correction observations still require an
// explicit host relation before capture.
class common_flydelta_capture_candidate_collector {
public:
    common_flydelta_capture_candidate_collector(
            std::string model_profile_fingerprint,
            std::string capture_layout_revision,
            size_t max_candidates = 64);

    bool observe(
            const common_adaptation_evidence_source_match & match,
            const common_learning_transaction & transaction,
            std::string & error);

    // Completes a discovered source with a host-verified, reference-only
    // relation. This is the explicit path for reflection/user-correction and
    // other sources that are never candidate-ready from broad runtime flags.
    bool observe_verified_relation(
            const common_adaptation_evidence_relation & relation,
            const common_adaptation_evidence & evidence,
            const common_learning_transaction & transaction,
            std::string & error);

    std::function<bool(
            const common_adaptation_evidence_source_match &,
            const common_learning_transaction &,
            std::string &)> source_observer();

    const std::vector<common_flydelta_capture_candidate> & candidates() const { return queue; }

private:
    std::string model_profile_fingerprint;
    std::string capture_layout_revision;
    size_t max_candidates;
    std::vector<common_flydelta_capture_candidate> queue;
};
