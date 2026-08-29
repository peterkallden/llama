#include "agent/adaptation/training-candidate.h"

static bool valid_score(float value) { return value >= 0.0f && value <= 1.0f; }

const char * common_learning_destination_name(common_learning_destination destination) {
    switch (destination) {
        case common_learning_destination::retain: return "retain";
        case common_learning_destination::memory: return "memory";
        case common_learning_destination::procedure: return "procedure";
        case common_learning_destination::training_candidate: return "training_candidate";
        case common_learning_destination::reject: return "reject";
    }
    return "reject";
}

const char * common_training_candidate_status_name(common_training_candidate_status status) {
    switch (status) {
        case common_training_candidate_status::observed: return "observed";
        case common_training_candidate_status::eligible: return "eligible";
        case common_training_candidate_status::approved: return "approved";
        case common_training_candidate_status::rejected: return "rejected";
        case common_training_candidate_status::revoked: return "revoked";
    }
    return "rejected";
}

common_learning_destination common_learning_destination_for(
        const common_learning_observation & observation) {
    if (observation.cause == common_learning_cause::project_knowledge) return common_learning_destination::memory;
    if (observation.cause == common_learning_cause::host_contract || observation.cause == common_learning_cause::policy) {
        return common_learning_destination::retain;
    }
    if (observation.cause != common_learning_cause::model_behavior) return common_learning_destination::retain;
    if (observation.verification == common_learning_verification::unverified) return common_learning_destination::retain;
    return common_learning_destination::training_candidate;
}

bool common_training_candidate_qualifies(
        const common_training_candidate & candidate,
        const common_training_candidate_policy & policy,
        std::string & error) {
    error.clear();
    if (candidate.schema_version != 1) { error = "unsupported training candidate schema"; return false; }
    if (candidate.id.empty()) { error = "training candidate requires id"; return false; }
    if (candidate.transaction_ids.empty() || candidate.transaction_ids.size() > policy.max_transaction_ids) {
        error = "training candidate transaction bound is invalid"; return false;
    }
    if (candidate.hypothesis.empty() || candidate.hypothesis.size() > policy.max_text_size ||
            candidate.approved_prompt.empty() || candidate.approved_prompt.size() > policy.max_text_size ||
            candidate.approved_target.empty() || candidate.approved_target.size() > policy.max_text_size) {
        error = "training candidate contains invalid bounded text"; return false;
    }
    if (candidate.cause != common_learning_cause::model_behavior) {
        error = "training candidate cause is not model behavior"; return false;
    }
    if (candidate.observed_occurrences < policy.min_occurrences) { error = "training candidate has insufficient occurrences"; return false; }
    if (candidate.verified_recoveries < policy.min_verified_recoveries) { error = "training candidate has insufficient verified recoveries"; return false; }
    if (candidate.contradictions != 0) { error = "training candidate has unresolved contradictions"; return false; }
    if (!valid_score(candidate.confidence) || candidate.confidence < policy.min_confidence) { error = "training candidate confidence is below threshold"; return false; }
    if (candidate.status != common_training_candidate_status::approved) {
        error = "training candidate requires explicit approval";
        return false;
    }
    return true;
}
