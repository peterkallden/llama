#pragma once

#include "agent/dataset-contracts.h"
#include "agent/investigation-contracts.h"

#include <string>
#include <utility>
#include <vector>

struct common_agent_data_store_config {
    std::string backend = "auto";
    std::string path;
};

class common_agent_virtual_dataset_source {
public:
    virtual ~common_agent_virtual_dataset_source() = default;
    virtual bool authorize_source(const std::string &, const common_agent_scope &, std::string & error) const {
        error = "investigation source authorization is unavailable";
        return false;
    }
    virtual bool execute(
        const common_agent_dataset_descriptor & descriptor,
        const std::string & operation,
        const std::string & request_json,
        const common_agent_scope & scope,
        bool allow_network,
        std::string & result_json,
        std::string & error) = 0;
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

    // Batch ingestion keeps importers independent of backend transaction
    // details. Backends override this to reduce per-row database overhead.
    virtual bool put_rows(
            const std::string & dataset,
            const std::vector<std::pair<std::string, std::string>> & rows,
            std::string & error) {
        for (const auto & row : rows) {
            if (!put_row(dataset, row.first, row.second, error)) return false;
        }
        error.clear();
        return true;
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

    // Find a fully published materialization for the currently authorized
    // resource bytes. A descriptor is published only after all rows have been
    // written, so this lookup is safe before parsing/normalizing the source.
    virtual bool find_dataset_materialization(
            const std::string & source_resource_uri,
            const std::string & content_hash,
            const std::string & importer_version,
            common_agent_dataset_descriptor & descriptor,
            std::string & error) {
        descriptor = {};
        std::vector<common_agent_dataset_descriptor> candidates;
        if (!list_dataset_descriptors(candidates, error)) return false;
        const common_agent_dataset_descriptor * match = nullptr;
        for (const auto & candidate : candidates) {
            if (candidate.ref.source_resource_uri != source_resource_uri ||
                    candidate.ref.content_hash != content_hash ||
                    candidate.import_processor_version != importer_version) continue;
            if (match != nullptr) {
                error = "multiple complete dataset materializations match the source resource";
                return false;
            }
            match = &candidate;
        }
        if (match == nullptr) {
            error.clear();
            return false;
        }
        if (!validate_common_agent_dataset_descriptor(*match, common_agent_dataset_limits{}, error)) {
            descriptor = {};
            error = "cached dataset materialization is invalid: " + error;
            return false;
        }
        descriptor = *match;
        error.clear();
        return true;
    }

    virtual bool execute(
        const std::string & operation,
        const std::string & request_json,
        std::string & result_json,
        std::string & error) = 0;

    // Scoped execution is used by host-backed virtual datasets. Remote
    // sources must explicitly honor the selected scope and resolved network
    // permission; existing local stores retain their established behavior.
    virtual bool execute_scoped(
        const std::string & operation,
        const std::string & request_json,
        const common_agent_scope & scope,
        bool allow_network,
        std::string & result_json,
        std::string & error) {
        (void) scope;
        (void) allow_network;
        return execute(operation, request_json, result_json, error);
    }

    virtual bool get_investigation_case(const std::string &, const common_agent_scope &,
            common_investigation_case_revision &, std::string &) {
        return false;
    }
    virtual bool list_investigation_cases(const common_agent_scope &, size_t,
            const std::string &, std::vector<common_investigation_case_summary> &,
            std::string &, std::string &) {
        return false;
    }
    virtual bool publish_investigation_case(const common_investigation_case_revision &,
            uint64_t, const std::string &, common_investigation_case_revision &,
            std::string &) {
        return false;
    }
    virtual bool publish_investigation_report(const common_agent_scope &,
            const common_investigation_report_index_entry &,
            common_investigation_report_index_entry &, std::string &) {
        return false;
    }
    virtual bool list_investigation_reports(const common_agent_scope &, const std::string &,
            uint64_t, std::vector<common_investigation_report_index_entry> &, std::string &) {
        return false;
    }
    virtual bool set_virtual_dataset_source(common_agent_virtual_dataset_source *) { return false; }
    virtual bool authorize_investigation_source(const std::string & source_id,
            const common_agent_scope & scope, std::string & error) const {
        (void) source_id; (void) scope;
        error = "investigation source authorization is unavailable";
        return false;
    }
};
