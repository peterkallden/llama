#include "agent-dataset-importer.h"

#include <nlohmann/json.hpp>

#include <algorithm>

using json = nlohmann::ordered_json;

namespace {

common_agent_dataset_column_type column_type(const std::string & value) {
    if (value == "null") return common_agent_dataset_column_type::null_;
    if (value == "boolean") return common_agent_dataset_column_type::boolean;
    if (value == "integer") return common_agent_dataset_column_type::integer;
    if (value == "decimal") return common_agent_dataset_column_type::decimal;
    if (value == "string") return common_agent_dataset_column_type::string;
    if (value == "date") return common_agent_dataset_column_type::date;
    if (value == "datetime") return common_agent_dataset_column_type::datetime;
    if (value == "binary") return common_agent_dataset_column_type::binary;
    return common_agent_dataset_column_type::unknown;
}

bool read_size(const json & object, const char * key, size_t & value) {
    if (!object.contains(key) || !object[key].is_number_unsigned()) return false;
    value = object[key].get<size_t>();
    return true;
}

} // namespace

bool import_agent_worksheet_envelope(
        common_agent_data_store & store,
        const agent_dataset_import_request & request,
        std::vector<common_agent_dataset_descriptor> & imported,
        std::string & error) {
    imported.clear();
    if (request.source_resource_uri.empty() || request.source_workbook_name.empty() ||
            request.import_processor_id.empty()) {
        error = "dataset import requires source and processor provenance";
        return false;
    }
    const auto root = json::parse(request.worksheet_json, nullptr, false);
    if (!root.is_object() || !root.contains("worksheets") || !root["worksheets"].is_array()) {
        error = "worksheet envelope requires worksheets array";
        return false;
    }
    if (root["worksheets"].size() > request.limits.max_sheets ||
            root["worksheets"].size() > request.limits.max_generated_datasets) {
        error = "worksheet count exceeds host limits";
        return false;
    }

    for (const auto & worksheet : root["worksheets"]) {
        if (!worksheet.is_object() || !worksheet.contains("name") ||
                !worksheet["name"].is_string() ||
                !worksheet.contains("columns") || !worksheet["columns"].is_array() ||
                !worksheet.contains("rows") || !worksheet["rows"].is_array()) {
            error = "worksheet requires name, columns and rows";
            return false;
        }
        const std::string sheet_name = worksheet["name"].get<std::string>();
        if (sheet_name.empty() || worksheet["columns"].size() > request.limits.max_columns ||
                worksheet["rows"].size() > request.limits.max_rows) {
            error = "worksheet shape exceeds host limits";
            return false;
        }
        common_agent_dataset_descriptor descriptor;
        descriptor.ref.uri = "dataset://import/" + std::to_string(imported.size()) + "/" + sheet_name;
        descriptor.ref.name = sheet_name;
        descriptor.ref.row_count = worksheet["rows"].size();
        descriptor.ref.column_count = worksheet["columns"].size();
        descriptor.ref.source_resource_uri = request.source_resource_uri;
        descriptor.ref.source_representation = "xlsx:worksheet";
        descriptor.source_workbook_name = request.source_workbook_name;
        descriptor.source_sheet_name = sheet_name;
        if (worksheet.contains("index") && worksheet["index"].is_number_unsigned()) {
            descriptor.source_sheet_index = worksheet["index"].get<size_t>();
        }
        descriptor.source_range = worksheet.value("range", std::string());
        descriptor.source_object = worksheet.value("object", std::string());
        descriptor.import_processor_id = request.import_processor_id;
        descriptor.import_processor_version = request.import_processor_version;
        for (const auto & column : worksheet["columns"]) {
            if (!column.is_object() || !column.contains("name") ||
                    !column["name"].is_string()) {
                error = "worksheet contains an invalid column";
                return false;
            }
            descriptor.columns.push_back({
                column["name"].get<std::string>(),
                column_type(column.value("type", std::string("unknown"))),
                column.value("nullable", true),
            });
        }
        if (!validate_common_agent_dataset_descriptor(descriptor, request.limits, error)) return false;

        for (size_t row_index = 0; row_index < worksheet["rows"].size(); ++row_index) {
            const auto & row = worksheet["rows"][row_index];
            if (!row.is_object()) {
                error = "worksheet rows must be JSON objects";
                return false;
            }
            std::string row_id = descriptor.ref.uri + "/row/" + std::to_string(row_index);
            if (!store.put_row(descriptor.ref.uri, row_id, row.dump(), error)) return false;
        }
        imported.push_back(std::move(descriptor));
    }
    error.clear();
    return true;
}
