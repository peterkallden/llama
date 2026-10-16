#pragma once

#include "agent/data-store.h"
#include "agent/dataset-contracts.h"

#include <string>
#include <optional>
#include <vector>
#include <cstdint>

struct agent_dataset_import_stats {
    size_t rows_imported = 0;
    size_t batches_written = 0;
    bool reused_existing = false;
    uint64_t write_elapsed_us = 0;
    uint64_t descriptor_elapsed_us = 0;
};

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
    // A stable hash allows safe reuse of an already materialized resource.
    // Empty keeps legacy imports uncached.
    std::string source_content_hash;
    // Stable source family used for common dataset provenance. The source
    // adapter supplies this while the shared importer owns validation/storage.
    std::string origin_kind = "spreadsheet";
    std::string import_processor_id;
    std::string import_processor_version;
    std::string worksheet_json;
    // Optional host-owned selection. When absent, all bounded worksheets are
    // imported; when present, exactly the matching worksheet is imported.
    std::optional<std::string> sheet_name;
    std::optional<size_t> sheet_index;
    common_agent_dataset_limits limits;
    size_t row_batch_size = 256;
};

bool import_agent_worksheet_envelope(
        common_agent_data_store & store,
        const agent_dataset_import_request & request,
        std::vector<common_agent_dataset_descriptor> & imported,
        std::string & error,
        agent_dataset_import_stats * stats = nullptr);

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
        std::string & error,
        char delimiter = ',');

// Recognizes Eurostat's documented language-specific tab-separated TOC,
// preserves its metadata, and derives each row's category path from the
// indentation hierarchy. Unrecognized plain text is not accepted.
bool normalize_agent_eurostat_toc_text(
        const std::string & toc_text,
        std::string & worksheet_json,
        std::string & error);

// Converts bounded JSON objects/collections or JSON-stat 2.0 into the same
// host-owned worksheet envelope used by CSV, workbook and document imports.
bool normalize_agent_json_dataset(
        const std::string & json_text,
        const std::string & dataset_name,
        std::string & worksheet_json,
        std::string & error,
        const common_agent_dataset_limits & limits = {});

// Converts bounded, record-oriented XML into rows. Eurostat's TOC and SDMX
// XML have dedicated projections so their hierarchy and references survive.
bool normalize_agent_xml_dataset(
        const std::string & xml_text,
        const std::string & dataset_name,
        std::string & worksheet_json,
        std::string & error,
        const common_agent_dataset_limits & limits = {});

bool make_agent_document_table_catalog(
        const std::string & worksheet_json,
        common_agent_document_table_catalog & catalog,
        std::string & error);
