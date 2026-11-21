#include "agent/adaptation/flydelta/flydelta-runtime-observer.h"

#include "agent/adaptation/flydelta/flydelta-candidate-lifecycle.h"

#include <algorithm>
#include <utility>

common_flydelta_runtime_candidate_observer::common_flydelta_runtime_candidate_observer(
        common_flydelta_capture_candidate_collector & value,
        common_learning_lifecycle_store * store,
        std::function<bool(
                const common_flydelta_capture_candidate &,
                std::string &)> enqueue)
    : collector(value), lifecycle_store(store), enqueue_capture_job(std::move(enqueue)) {}

bool common_flydelta_runtime_candidate_observer::observe(
        const common_adaptation_evidence_source_match & match,
        const common_learning_transaction & transaction,
        std::string & error) {
    error.clear();
    if (!collector.observe(match, transaction, error)) return false;
    if ((!lifecycle_store && !enqueue_capture_job) || !match.candidate_ready) return true;

    const auto candidate = std::find_if(
        collector.candidates().begin(), collector.candidates().end(),
        [&](const auto & value) {
            return value.transaction_id == transaction.id &&
                value.source == match.source &&
                value.behavior_key == match.behavior_key;
        });
    if (candidate == collector.candidates().end()) {
        error = "FlyDelta runtime observer could not resolve its capture candidate";
        return false;
    }

    if (lifecycle_store) {
        common_flydelta_lifecycle_event_context context;
        context.event_id = candidate->id;
        context.idempotency_key = candidate->id;
        context.source_id = transaction.id;
        context.scope = transaction.observation.scope;
        context.content_hash = transaction.observation.content_hash;
        context.created_at = transaction.created_at;
        if (!common_flydelta_append_capture_candidate_lifecycle(
                *lifecycle_store, context, *candidate, transaction, error)) return false;
    }
    if (enqueue_capture_job) {
        // Queue pressure or a temporary worker outage must not fail the user
        // turn. The candidate remains in the collector/lifecycle store and
        // the host may retry enqueueing it later.
        std::string enqueue_error;
        enqueue_capture_job(*candidate, enqueue_error);
    }
    return true;
}

bool common_flydelta_runtime_candidate_observer::observe_verified_relation(
        const common_adaptation_evidence_relation & relation,
        const common_adaptation_evidence & evidence,
        const common_learning_transaction & transaction,
        std::string & error) {
    error.clear();
    if (!collector.observe_verified_relation(relation, evidence, transaction, error)) return false;
    if ((!lifecycle_store && !enqueue_capture_job) || !relation.host_verified || !evidence.host_verified) return true;

    const auto candidate = std::find_if(
        collector.candidates().begin(), collector.candidates().end(),
        [&](const auto & value) {
            return value.transaction_id == transaction.id &&
                value.source == relation.source &&
                value.behavior_key == relation.behavior_key;
        });
    if (candidate == collector.candidates().end()) {
        error = "FlyDelta runtime observer could not resolve its verified capture candidate";
        return false;
    }
    if (lifecycle_store) {
        common_flydelta_lifecycle_event_context context;
        context.event_id = candidate->id;
        context.idempotency_key = candidate->id;
        context.source_id = transaction.id;
        context.scope = transaction.observation.scope;
        context.content_hash = transaction.observation.content_hash;
        context.created_at = transaction.created_at;
        if (!common_flydelta_append_capture_candidate_lifecycle(
                *lifecycle_store, context, *candidate, transaction, error)) return false;
    }
    if (enqueue_capture_job) {
        std::string enqueue_error;
        enqueue_capture_job(*candidate, enqueue_error);
    }
    return true;
}

std::function<bool(
        const common_adaptation_evidence_source_match &,
        const common_learning_transaction &,
        std::string &)>
common_flydelta_runtime_candidate_observer::source_observer() {
    return [this](const auto & match, const auto & transaction, std::string & error) {
        return observe(match, transaction, error);
    };
}
