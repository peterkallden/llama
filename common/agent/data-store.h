#pragma once

#include "agent/dataset-contracts.h"

#include <string>
#include <vector>

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

    // Optional registry lookup used by the model-facing dataset.select and
    // dataset.list contracts. Backends that do not persist descriptors keep
    // the legacy path-based dataset adapter instead.
    virtual bool list_dataset_descriptors(
            std::vector<common_agent_dataset_descriptor> & descriptors,
            std::string & error) {
        descriptors.clear();
        error = "data store does not support dataset descriptor listing";
        return false;
    }

    virtual bool find_dataset_by_name(
            const std::string & name,
            common_agent_dataset_descriptor & descriptor,
            std::string & error) {
        (void) name;
        descriptor = {};
        error = "data store does not support dataset name lookup";
        return false;
    }

    virtual bool execute(
        const std::string & operation,
        const std::string & request_json,
        std::string & result_json,
        std::string & error) = 0;
};
