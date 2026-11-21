#pragma once

#include "agent/adaptation/corpus-builder.h"

#include <cstddef>
#include <string>
#include <vector>

// Automatic training is a host/maintenance decision, never a side effect of
// inference. The group identity prevents unrelated behavior, transports or
// model profiles from satisfying the same threshold.
struct common_learning_training_group {
    std::string learning_domain;
    std::string tool_family;
    std::string provider_kind;
};

struct common_learning_auto_training_policy {
    bool enabled = false;
    size_t min_qualified_examples = 6;
    size_t max_scan = 4096;
};

struct common_learning_auto_training_result {
    bool ready = false;
    size_t qualified_examples = 0;
    size_t distinct_transaction_ids = 0;
    // The exact approved rows selected for the next corpus build. This is
    // deliberately candidate-based: transaction_ids are provenance and do
    // not turn one prompt/target row into several training rows.
    std::vector<std::string> qualified_candidate_ids;
    std::string group_key;
    std::string reason;
};

// Counts approved, fully qualified candidates in one explicit group. A
// candidate must have one distinct transaction; duplicate candidate ids are
// counted once. UNKNOWN/NEUTRAL FlyDelta trials are not common_training_candidate
// rows and therefore cannot satisfy this trigger.
bool common_learning_auto_training_ready(
        const std::vector<common_training_candidate> & candidates,
        const common_learning_training_group & group,
        const common_learning_auto_training_policy & policy,
        common_learning_auto_training_result & result,
        std::string & error);
