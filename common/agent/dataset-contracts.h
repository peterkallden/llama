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
};

struct common_agent_dataset_lineage {
    std::vector<std::string> parent_dataset_uris;
    std::string operation;
    std::string operation_summary;
};

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
};

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
