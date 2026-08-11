#include "dataset-contracts.h"

#include <limits>

const char * common_agent_dataset_column_type_name(
        common_agent_dataset_column_type type) {
    switch (type) {
        case common_agent_dataset_column_type::null_: return "null";
        case common_agent_dataset_column_type::boolean: return "boolean";
        case common_agent_dataset_column_type::integer: return "integer";
        case common_agent_dataset_column_type::decimal: return "decimal";
        case common_agent_dataset_column_type::string: return "string";
        case common_agent_dataset_column_type::date: return "date";
        case common_agent_dataset_column_type::datetime: return "datetime";
        case common_agent_dataset_column_type::binary: return "binary";
        case common_agent_dataset_column_type::unknown: return "unknown";
    }
    return "unknown";
}

bool validate_common_agent_dataset_ref(
        const common_agent_dataset_ref & ref,
        std::string & error) {
    if (ref.uri.empty() || ref.name.empty()) {
        error = "dataset reference requires uri and name";
        return false;
    }
    if (ref.source_resource_uri.empty()) {
        error = "dataset reference requires source resource provenance";
        return false;
    }
    if (ref.column_count == 0 && ref.row_count != 0) {
        error = "dataset reference cannot have rows without columns";
        return false;
    }
    error.clear();
    return true;
}

bool validate_common_agent_dataset_descriptor(
        const common_agent_dataset_descriptor & descriptor,
        const common_agent_dataset_limits & limits,
        std::string & error) {
    if (!validate_common_agent_dataset_ref(descriptor.ref, error)) return false;
    if (descriptor.ref.row_count > limits.max_rows ||
            descriptor.ref.column_count > limits.max_columns) {
        error = "dataset shape exceeds host limits";
        return false;
    }
    if (descriptor.ref.column_count != descriptor.columns.size()) {
        error = "dataset column count does not match schema";
        return false;
    }
    if (descriptor.ref.row_count != 0 &&
            descriptor.ref.column_count > std::numeric_limits<size_t>::max() / descriptor.ref.row_count) {
        error = "dataset cell count overflows host limits";
        return false;
    }
    if (descriptor.ref.row_count * descriptor.ref.column_count > limits.max_cells) {
        error = "dataset cell count exceeds host limits";
        return false;
    }
    for (const auto & column : descriptor.columns) {
        if (column.name.empty()) {
            error = "dataset schema contains an unnamed column";
            return false;
        }
    }
    if (descriptor.import_processor_id.empty()) {
        error = "dataset descriptor requires import processor provenance";
        return false;
    }
    error.clear();
    return true;
}

