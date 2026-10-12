#pragma once

#include "resource/resource-contract.h"

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

// Dataset is a structured analytical handle. It is intentionally separate
// from common_runtime_resource_ref, which remains the authority for source
// bytes, media type and resource lineage.
enum class common_agent_dataset_column_type {
    null_,
    boolean,
    integer,
    decimal,
    string,
    date,
    datetime,
    binary,
    unknown,
};

const char * common_agent_dataset_column_type_name(
        common_agent_dataset_column_type type);

struct common_agent_dataset_column {
    std::string name;
    common_agent_dataset_column_type type = common_agent_dataset_column_type::unknown;
    bool nullable = true;
};

struct common_agent_dataset_ref {
    std::string uri;
    std::string name;
    size_t row_count = 0;
    size_t column_count = 0;
    std::string source_resource_uri;
    std::string source_representation;

    // Host-owned provenance for derived API datasets. source_request_json is
    // canonicalized and sanitized before it reaches this contract.
    std::string source_provider;
    std::string source_operation;
    std::string source_request_json;
    int64_t retrieved_at = 0;
    std::string content_hash;
};

// Dataset refs cross several boundaries. Keep the canonical field mapping in
// the dataset contract library instead of duplicating it in SQLite, daemon or
// provider code. A projection is named because model-facing output must not
// accidentally expose persistence-only provenance.
enum class common_agent_dataset_ref_json_projection {
    compact,
    full,
};

nlohmann::ordered_json common_agent_dataset_ref_to_json(
        const common_agent_dataset_ref & ref,
        common_agent_dataset_ref_json_projection projection =
            common_agent_dataset_ref_json_projection::compact);

bool common_agent_dataset_ref_from_json(
        const nlohmann::ordered_json & value,
        common_agent_dataset_ref & ref,
        std::string & error);

// Returns the bounded URI component used for host-owned derived datasets.
// Parsing remains structural: callers must still validate the resulting URI
// against the active turn and resource authority.
std::string common_agent_dataset_uri_scope_component(const std::string & value);

bool common_agent_dataset_uri_is_current_turn(
        const std::string & uri,
        const std::string & turn_id);

struct common_agent_dataset_lineage {
    std::vector<std::string> parent_dataset_uris;
    std::string operation;
    std::string operation_summary;
};

enum class common_agent_table_header_mode {
    explicit_,
    first_row,
    first_column,
    both,
    none,
    ambiguous,
};

const char * common_agent_table_header_mode_name(
        common_agent_table_header_mode mode);

struct common_agent_dataset_origin {
    std::string kind;
    std::string source_representation_uri;
    std::string source_node_id;
    size_t table_index = 0;
    std::vector<std::string> section_path;
    std::string caption;
    std::vector<std::string> notes;
    common_agent_table_header_mode header_mode = common_agent_table_header_mode::none;
    double header_confidence = 0.0;
    std::string header_reason;
};

// A bounded model-facing catalog entry. The canonical identity remains the
// host-owned node or dataset URI; name is a convenience lookup key.
struct common_agent_document_table_entry {
    size_t table_index = 0;
    std::string name;
    std::string caption;
    std::string node_id;
    std::string dataset_uri;
};

struct common_agent_document_table_catalog {
    std::string source_resource_uri;
    std::string source_representation_uri;
    std::vector<common_agent_document_table_entry> tables;
};

struct common_agent_document_table_locator {
    std::optional<size_t> table_index;
    std::string name;
    std::string node_id;
};

// Resolves a host-validated table locator. Names are trimmed, whitespace-
// collapsed and compared case-insensitively. A non-unique name fails closed.
bool resolve_common_agent_document_table(
        const common_agent_document_table_catalog & catalog,
        const common_agent_document_table_locator & locator,
        common_agent_document_table_entry & resolved,
        std::string & error);

struct common_agent_dataset_descriptor {
    common_agent_dataset_ref ref;
    std::vector<common_agent_dataset_column> columns;
    std::string source_workbook_name;
    std::string source_sheet_name;
    std::optional<size_t> source_sheet_index;
    std::string source_range;
    std::string source_object;
    std::string import_processor_id;
    std::string import_processor_version;
    common_agent_dataset_lineage lineage;
    common_agent_dataset_origin origin;
};

// Classifies a bounded rectangular table sample. The host may materialize a
// dataset automatically only for explicit or high-confidence simple headers;
// ambiguous candidates remain available for a later normalization step.
common_agent_table_header_mode classify_common_agent_table_headers(
        const std::vector<std::vector<std::string>> & rows,
        double & confidence,
        std::string & reason);

struct common_agent_dataset_limits {
    size_t max_source_bytes = 128 * 1024 * 1024;
    size_t max_sheets = 256;
    size_t max_rows = 1'000'000;
    size_t max_columns = 512;
    size_t max_cells = 10'000'000;
    size_t max_generated_datasets = 256;
};

bool validate_common_agent_dataset_ref(
        const common_agent_dataset_ref & ref,
        std::string & error);

bool validate_common_agent_dataset_descriptor(
        const common_agent_dataset_descriptor & descriptor,
        const common_agent_dataset_limits & limits,
        std::string & error);

// Converts the small model-friendly predicate form used by dataset tools into
// the canonical field/operator/value condition list consumed by data stores.
// Only dataset query/filter arguments are changed; unrelated tool arguments
// are left untouched.
bool normalize_common_agent_dataset_tool_arguments(
        const std::string & tool_name,
        nlohmann::ordered_json & arguments,
        std::string & error);
