#pragma once

#include "agent/data-store.h"
#include "agent/dataset-contracts.h"

#include <string>
#include <optional>
#include <vector>

// Host-normalized worksheet envelope. Resource processors may use a richer
// intermediate (for example Pandoc AST) but must normalize it before dataset
// materialization. Rows stay in the host store and never become a tool result.
struct agent_dataset_import_request {
    std::string source_resource_uri;
    std::string source_workbook_name;
    // Stable representation kind for provenance. The default preserves the
    // existing spreadsheet import contract; document tables use
    // "document:table".
    std::string source_representation = "xlsx:worksheet";
    std::string source_representation_uri;
    std::string import_processor_id;
    std::string import_processor_version;
    std::string worksheet_json;
    // Optional host-owned selection. When absent, all bounded worksheets are
    // imported; when present, exactly the matching worksheet is imported.
    std::optional<std::string> sheet_name;
    std::optional<size_t> sheet_index;
    common_agent_dataset_limits limits;
};

bool import_agent_worksheet_envelope(
        common_agent_data_store & store,
        const agent_dataset_import_request & request,
        std::vector<common_agent_dataset_descriptor> & imported,
        std::string & error);

// Normalizes the Pandoc JSON AST emitted by the shared Pandoc processor into
// the bounded worksheet envelope consumed by the dataset importer.
bool normalize_agent_pandoc_workbook_json(
        const std::string & pandoc_json,
        std::string & worksheet_json,
        std::string & error);

// The Pandoc AST shape is shared by workbook and document representations.
// Keep the workbook-named entry point for compatibility, while exposing the
// document meaning explicitly at the caller boundary.
bool normalize_agent_pandoc_document_json(
        const std::string & pandoc_json,
        std::string & worksheet_json,
        std::string & error);

// Converts a bounded CSV resource into the same worksheet envelope used by
// spreadsheet and document-table import. The caller owns the byte bound;
// this function does not create a second data representation.
bool normalize_agent_csv_text(
        const std::string & csv_text,
        const std::string & dataset_name,
        std::string & worksheet_json,
        std::string & error);

bool make_agent_document_table_catalog(
        const std::string & worksheet_json,
        common_agent_document_table_catalog & catalog,
        std::string & error);
