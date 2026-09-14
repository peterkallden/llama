#include "agent/adaptation/flydelta/flydelta-capture.h"

#include <algorithm>
#include <utility>

namespace {

bool bounded(const std::string & value) {
    return !value.empty() && value.size() <= 512;
}

} // namespace

bool common_flydelta_capture_candidate_validate(
        const common_flydelta_capture_candidate & candidate,
        std::string & error) {
    error.clear();
    if (candidate.schema_version != 1 || !bounded(candidate.id) ||
            !bounded(candidate.transaction_id) || !candidate.candidate_ready ||
            candidate.evidence_refs.empty() || candidate.evidence_refs.size() > 16 ||
            !bounded(candidate.model_profile_fingerprint) ||
            !bounded(candidate.capture_layout_revision)) {
        error = "FlyDelta capture candidate is incomplete or outside bounds";
        return false;
    }
    for (const auto & ref : candidate.evidence_refs) {
        if (!bounded(ref)) {
            error = "FlyDelta capture candidate evidence reference is invalid";
            return false;
        }
    }
    return true;
}

common_flydelta_capture_candidate_collector::common_flydelta_capture_candidate_collector(
        std::string profile_fingerprint,
        std::string layout_revision,
        size_t candidate_bound)
    : model_profile_fingerprint(std::move(profile_fingerprint)),
      capture_layout_revision(std::move(layout_revision)),
      max_candidates(candidate_bound) {}

bool common_flydelta_capture_candidate_collector::observe(
        const common_adaptation_evidence_source_match & match,
        const common_learning_transaction & transaction,
        std::string & error) {
    error.clear();
    if (!match.candidate_ready) return true;
    if (max_candidates == 0 || queue.size() >= max_candidates) {
        error = "FlyDelta capture candidate queue is full";
        return false;
    }
    common_flydelta_capture_candidate candidate;
    // A turn may expose more than one qualified source. Include the source
    // in the identity so a reflection relation cannot collide with a tool
    // repair relation that happens to reference the same transaction.
    candidate.id = std::string("flydelta://capture-candidate/") +
        common_adaptation_evidence_source_name(match.source) + "/" + transaction.id;
    candidate.transaction_id = transaction.id;
    candidate.source = match.source;
    candidate.evidence_refs = match.evidence_refs;
    candidate.model_profile_fingerprint = model_profile_fingerprint;
    candidate.capture_layout_revision = capture_layout_revision;
    candidate.candidate_ready = true;
    if (!common_flydelta_capture_candidate_validate(candidate, error)) return false;
    const auto duplicate = std::find_if(queue.begin(), queue.end(), [&](const auto & item) {
        return item.id == candidate.id;
    });
    if (duplicate == queue.end()) queue.push_back(std::move(candidate));
    return true;
}

bool common_flydelta_capture_candidate_collector::observe_verified_relation(
        const common_adaptation_evidence_relation & relation,
        const common_adaptation_evidence & evidence,
        const common_learning_transaction & transaction,
        std::string & error) {
    error.clear();
    if (relation.source != evidence.source) {
        error = "FlyDelta relation and evidence sources differ";
        return false;
    }
    if (!common_adaptation_evidence_validate(evidence, 64, error)) return false;
    if (!relation.host_verified || !evidence.host_verified) return true;

    common_adaptation_evidence_source_match match;
    match.source = relation.source;
    match.candidate_ready = true;
    match.evidence_refs = {
        evidence.id, evidence.baseline_ref, evidence.candidate_ref,
        evidence.verifier_ref,
    };
    match.evidence_refs.insert(match.evidence_refs.end(),
        evidence.transaction_ids.begin(), evidence.transaction_ids.end());
    return observe(match, transaction, error);
}

std::function<bool(
        const common_adaptation_evidence_source_match &,
        const common_learning_transaction &,
        std::string &)>
common_flydelta_capture_candidate_collector::source_observer() {
    return [this](const auto & match, const auto & transaction, std::string & error) {
        return observe(match, transaction, error);
    };
}
