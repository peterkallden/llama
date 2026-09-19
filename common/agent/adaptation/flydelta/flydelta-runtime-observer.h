#pragma once

#include "agent/adaptation/flydelta/flydelta-capture.h"
#include "agent/adaptation/lifecycle-store.h"

#include <functional>
#include <string>

// Host bridge for the real runtime observer path. The capture collector
// remains responsible for the bounded candidate queue; this bridge optionally
// journals the same discovered candidate in the separate lifecycle store.
// It never infers an outcome and never schedules a search by itself.
class common_flydelta_runtime_candidate_observer {
public:
    common_flydelta_runtime_candidate_observer(
            common_flydelta_capture_candidate_collector & collector,
            common_learning_lifecycle_store * lifecycle_store = nullptr,
            std::function<bool(
                    const common_flydelta_capture_candidate &,
                    std::string &)> enqueue_capture_job = {});

    bool observe(
            const common_adaptation_evidence_source_match & match,
            const common_learning_transaction & transaction,
            std::string & error);

    bool observe_verified_relation(
            const common_adaptation_evidence_relation & relation,
            const common_adaptation_evidence & evidence,
            const common_learning_transaction & transaction,
            std::string & error);

    std::function<bool(
            const common_adaptation_evidence_source_match &,
            const common_learning_transaction &,
            std::string &)> source_observer();

private:
    common_flydelta_capture_candidate_collector & collector;
    common_learning_lifecycle_store * lifecycle_store;
    std::function<bool(
            const common_flydelta_capture_candidate &,
            std::string &)> enqueue_capture_job;
};
