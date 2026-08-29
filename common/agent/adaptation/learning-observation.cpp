#include "agent/adaptation/learning-observation.h"

#include <cstdint>
#include <iomanip>
#include <sstream>

const char * common_learning_cause_name(common_learning_cause cause) {
    switch (cause) {
        case common_learning_cause::unknown: return "unknown";
        case common_learning_cause::model_behavior: return "model_behavior";
        case common_learning_cause::host_contract: return "host_contract";
        case common_learning_cause::policy: return "policy";
        case common_learning_cause::missing_evidence: return "missing_evidence";
        case common_learning_cause::project_knowledge: return "project_knowledge";
    }
    return "unknown";
}

const char * common_learning_verification_name(common_learning_verification verification) {
    switch (verification) {
        case common_learning_verification::unverified: return "unverified";
        case common_learning_verification::host_verified: return "host_verified";
        case common_learning_verification::user_confirmed: return "user_confirmed";
    }
    return "unverified";
}

bool common_learning_observation_qualifies(const common_learning_observation & observation) {
    if (!observation.collection_allowed) return false;
    for (const auto & signal : observation.signals) {
        switch (signal.type) {
            case common_learning_signal_type::tool_failure:
            case common_learning_signal_type::successful_recovery:
            case common_learning_signal_type::reflection_hint:
            case common_learning_signal_type::user_correction:
                return true;
        }
    }
    return false;
}

bool common_learning_observation_validate(
        const common_learning_observation & observation,
        size_t max_evidence,
        std::string & error) {
    error.clear();
    if (observation.schema_version != 1) { error = "unsupported learning observation schema"; return false; }
    if (observation.id.empty()) { error = "learning observation requires id"; return false; }
    if (observation.source_turn_id.empty()) { error = "learning observation requires source turn"; return false; }
    if (observation.source_plan_id.empty()) { error = "learning observation requires source plan"; return false; }
    if (observation.idempotency_key.empty()) { error = "learning observation requires idempotency key"; return false; }
    if (observation.signals.empty()) { error = "learning observation requires signal"; return false; }
    if (observation.evidence_ids.size() > max_evidence) { error = "learning observation exceeds evidence bound"; return false; }
    for (const auto & evidence_id : observation.evidence_ids) {
        if (evidence_id.empty()) { error = "learning observation contains empty evidence id"; return false; }
    }
    if (!observation.collection_allowed) { error = "learning collection is not allowed for scope"; return false; }
    if (observation.content_hash.empty()) { error = "learning observation requires content hash"; return false; }
    return true;
}

std::string common_learning_observation_canonical(const common_learning_observation & observation) {
    std::ostringstream out;
    out << observation.schema_version << '\n'
        << observation.id << '\n'
        << observation.scope.namespace_id << '\n'
        << observation.scope.session_id << '\n'
        << observation.scope.project_id << '\n'
        << observation.scope.turn_id << '\n'
        << observation.source_turn_id << '\n'
        << observation.source_plan_id << '\n'
        << common_learning_cause_name(observation.cause) << '\n'
        << common_learning_verification_name(observation.verification) << '\n'
        << observation.idempotency_key << '\n'
        << observation.collection_allowed << '\n';
    for (const auto & signal : observation.signals) {
        out << common_learning_signal_type_name(signal.type) << '\x1f'
            << signal.plan_id << '\x1f' << signal.step_id << '\x1f'
            << signal.tool_name << '\x1f' << signal.evidence_id << '\x1f'
            << signal.summary << '\x1f' << signal.tool_family << '\x1f'
            << signal.provider_kind << '\n';
    }
    for (const auto & evidence_id : observation.evidence_ids) out << evidence_id << '\n';
    return out.str();
}

std::string common_learning_observation_hash(const common_learning_observation & observation) {
    // This is a stable identity hash for the contract and deduplication key,
    // not a cryptographic content hash.  Resource/artifact integrity hashes
    // remain SHA-256 values owned by their respective stores.
    uint64_t hash = 1469598103934665603ULL;
    for (const unsigned char byte : common_learning_observation_canonical(observation)) {
        hash ^= byte;
        hash *= 1099511628211ULL;
    }
    std::ostringstream out;
    out << "identity:fnv1a64:" << std::hex << std::setw(16) << std::setfill('0') << hash;
    return out.str();
}
