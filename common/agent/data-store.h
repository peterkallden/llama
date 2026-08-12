#pragma once

#include "agent/dataset-contracts.h"

#include <string>

struct common_agent_data_store_config {
    std::string backend = "auto";
    std::string path;
};

// Host-owned backend seam for structured data tools. Tool arguments remain
// semantic JSON; the selected backend owns storage and query translation.
class common_agent_data_store {
public:
    virtual ~common_agent_data_store() = default;

    // Optional host-owned ingestion seam. Implementations may stream bounded
    // rows into their existing backend without returning the row set through
    // the model-facing tool protocol.
    virtual bool put_row(
            const std::string & dataset,
            const std::string & row_id,
            const std::string & row_json,
            std::string & error) {
        (void) dataset;
        (void) row_id;
        (void) row_json;
        error = "data store does not support dataset ingestion";
        return false;
    }

    virtual bool put_dataset_descriptor(
            const common_agent_dataset_descriptor & descriptor,
            std::string & error) {
        (void) descriptor;
        error = "data store does not support dataset descriptors";
        return false;
    }

    virtual bool get_dataset_descriptor(
            const std::string & dataset_uri,
            common_agent_dataset_descriptor & descriptor,
            std::string & error) {
        (void) dataset_uri;
        descriptor = {};
        error = "data store does not support dataset descriptors";
        return false;
    }

    virtual bool execute(
        const std::string & operation,
        const std::string & request_json,
        std::string & result_json,
        std::string & error) = 0;
};
