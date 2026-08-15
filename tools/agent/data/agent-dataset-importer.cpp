#include "agent-dataset-importer.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdlib>
#include <sstream>

using json = nlohmann::ordered_json;

namespace {

std::string inline_text(const json & value) {
    if (value.is_string()) return value.get<std::string>();
    if (value.is_array()) {
        std::string result;
        for (const auto & item : value) result += inline_text(item);
        return result;
    }
    if (!value.is_object() || !value.contains("t")) return {};
    const auto type = value["t"].get<std::string>();
    if (type == "Str" || type == "Code" || type == "Math") {
        if (value["c"].is_string()) return value["c"].get<std::string>();
        if (value["c"].is_array() && !value["c"].empty() && value["c"].back().is_string()) return value["c"].back().get<std::string>();
    }
    if (type == "Space" || type == "SoftBreak" || type == "LineBreak") return " ";
    if (value["c"].is_array()) {
        std::string result;
        for (const auto & item : value["c"]) result += inline_text(item);
        return result;
    }
    return {};
}

std::string block_text(const json & blocks) {
    std::string result;
    if (!blocks.is_array()) return result;
    for (const auto & block : blocks) {
        if (!block.is_object() || !block.contains("t")) continue;
        if (block["t"] == "Plain" || block["t"] == "Para") {
            for (const auto & item : block.value("c", json::array())) result += inline_text(item);
        }
    }
    return result;
}

bool table_rows(const json & rows, std::vector<json> & output, std::string & error) {
    if (!rows.is_array()) { error = "Pandoc table rows are invalid"; return false; }
    for (const auto & row : rows) {
        if (!row.is_array() || row.size() < 2 || !row[1].is_array()) { error = "Pandoc table row is invalid"; return false; }
        json values = json::array();
        for (const auto & cell : row[1]) {
            if (!cell.is_array() || cell.size() < 5) { error = "Pandoc table cell is invalid"; return false; }
            values.push_back(block_text(cell[4]));
        }
        output.push_back(std::move(values));
    }
    return true;
}

common_agent_dataset_column_type inferred_type(const std::string & value) {
    if (value.empty()) return common_agent_dataset_column_type::null_;
    if (value == "true" || value == "false") return common_agent_dataset_column_type::boolean;
    char * end = nullptr;
    std::strtoll(value.c_str(), &end, 10);
    if (end != nullptr && *end == '\0') return common_agent_dataset_column_type::integer;
    std::strtod(value.c_str(), &end);
    if (end != nullptr && *end == '\0') return common_agent_dataset_column_type::decimal;
    return common_agent_dataset_column_type::string;
}

json typed_value(const std::string & value, common_agent_dataset_column_type type) {
    if (type == common_agent_dataset_column_type::null_) return nullptr;
    if (type == common_agent_dataset_column_type::boolean) return value == "true";
    if (type == common_agent_dataset_column_type::integer) return std::stoll(value);
    if (type == common_agent_dataset_column_type::decimal) return std::stod(value);
    return value;
}

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

std::vector<std::string> csv_row(const std::string & line) {
    std::vector<std::string> fields;
    std::string field;
    bool quoted = false;
    for (size_t index = 0; index < line.size(); ++index) {
        const char value = line[index];
        if (value == '"') {
            if (quoted && index + 1 < line.size() && line[index + 1] == '"') {
                field += '"';
                ++index;
            } else quoted = !quoted;
        } else if (value == ',' && !quoted) {
            fields.push_back(field);
            field.clear();
        } else field += value;
    }
    fields.push_back(std::move(field));
    return fields;
}

std::string stable_source_key(const std::string & value) {
    uint64_t hash = 1469598103934665603ULL;
    for (const unsigned char character : value) {
        hash ^= character;
        hash *= 1099511628211ULL;
    }
    return std::to_string(hash);
}

} // namespace

bool normalize_agent_pandoc_workbook_json(
        const std::string & pandoc_json,
        std::string & worksheet_json,
        std::string & error) {
    const auto root = json::parse(pandoc_json, nullptr, false);
    if (!root.is_object() || !root.contains("blocks") || !root["blocks"].is_array()) {
        error = "Pandoc workbook JSON requires blocks";
        return false;
    }
    json envelope = {{"worksheets", json::array()}};
    std::string sheet_name;
    size_t sheet_index = 0;
    for (const auto & block : root["blocks"]) {
        if (!block.is_object() || !block.contains("t")) continue;
        if (block["t"] == "Header") {
            const auto content = block.value("c", json::array());
            if (content.is_array() && content.size() >= 3) sheet_name = inline_text(content[2]);
            continue;
        }
        if (block["t"] != "Table") continue;
        const auto content = block.value("c", json::array());
        if (!content.is_array() || content.size() < 5) { error = "Pandoc table shape is unsupported"; return false; }
        std::vector<json> header_rows;
        const auto & head = content[3];
        if (head.is_array() && head.size() >= 2) if (!table_rows(head[1], header_rows, error)) return false;
        std::vector<json> value_rows;
        for (const auto & body : content[4]) {
            if (body.is_array() && body.size() >= 4 && !table_rows(body[3], value_rows, error)) return false;
        }
        std::string header_mode = "explicit";
        std::string header_reason = "Pandoc supplied an explicit table header";
        double header_confidence = 1.0;
        if (header_rows.empty()) {
            std::vector<std::vector<std::string>> sample;
            for (size_t row_index = 0; row_index < value_rows.size() && row_index < 8; ++row_index) {
                std::vector<std::string> row;
                for (const auto & value : value_rows[row_index]) row.push_back(value.get<std::string>());
                sample.push_back(std::move(row));
            }
            const auto mode = classify_common_agent_table_headers(sample, header_confidence, header_reason);
            header_mode = common_agent_table_header_mode_name(mode);
            if (mode != common_agent_table_header_mode::first_row) {
                error = "Pandoc table header requires normalization: " + header_reason;
                return false;
            }
            if (value_rows.empty()) { error = "Pandoc table has no rows"; return false; }
            header_rows.push_back(value_rows.front());
            value_rows.erase(value_rows.begin());
        }
        std::vector<std::string> names;
        for (size_t index = 0; index < header_rows.front().size(); ++index) {
            std::string name = header_rows.front()[index].get<std::string>();
            names.push_back(name.empty() ? "column_" + std::to_string(index + 1) : name);
        }
        std::vector<common_agent_dataset_column_type> types(names.size(), common_agent_dataset_column_type::null_);
        for (const auto & values : value_rows) for (size_t index = 0; index < names.size() && index < values.size(); ++index) {
            const auto type = inferred_type(values[index].get<std::string>());
            if (type == common_agent_dataset_column_type::string) types[index] = type;
            else if (types[index] == common_agent_dataset_column_type::null_) types[index] = type;
            else if (types[index] != type) types[index] = common_agent_dataset_column_type::string;
        }
        const auto table_index = sheet_index++;
        json worksheet = {{"name", sheet_name.empty() ? "Sheet_" + std::to_string(table_index + 1) : sheet_name},
            {"index", table_index}, {"table_index", table_index}, {"node_id", "document-node://table/" + std::to_string(table_index)},
            {"header_mode", header_mode}, {"header_confidence", header_confidence},
            {"header_reason", header_reason}, {"columns", json::array()}, {"rows", json::array()}};
        for (size_t index = 0; index < names.size(); ++index) worksheet["columns"].push_back({
            {"name", names[index]}, {"type", common_agent_dataset_column_type_name(types[index])}, {"nullable", true}});
        for (const auto & values : value_rows) {
            json row = json::object();
            for (size_t index = 0; index < names.size(); ++index) {
                const std::string value = index < values.size() ? values[index].get<std::string>() : std::string();
                row[names[index]] = typed_value(value, types[index]);
            }
            worksheet["rows"].push_back(std::move(row));
        }
        envelope["worksheets"].push_back(std::move(worksheet));
        sheet_name.clear();
    }
    if (envelope["worksheets"].empty()) { error = "Pandoc workbook JSON contains no tables"; return false; }
    worksheet_json = envelope.dump();
    error.clear();
    return true;
}

bool normalize_agent_pandoc_document_json(
        const std::string & pandoc_json,
        std::string & worksheet_json,
        std::string & error) {
    return normalize_agent_pandoc_workbook_json(pandoc_json, worksheet_json, error);
}

bool normalize_agent_csv_text(
        const std::string & csv_text,
        const std::string & dataset_name,
        std::string & worksheet_json,
        std::string & error) {
    std::istringstream input(csv_text);
    std::string line;
    if (!std::getline(input, line)) {
        error = "CSV resource is empty";
        return false;
    }
    if (!line.empty() && line.back() == '\r') line.pop_back();
    const auto names = csv_row(line);
    if (names.empty() || names.size() > 512) {
        error = "CSV header exceeds the bounded column limit";
        return false;
    }
    std::vector<std::vector<std::string>> rows;
    while (rows.size() < 1'000'000 && std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        rows.push_back(csv_row(line));
    }
    if (!input.eof() && rows.size() >= 1'000'000) {
        error = "CSV resource exceeds the bounded row limit";
        return false;
    }
    std::vector<common_agent_dataset_column_type> types(
        names.size(), common_agent_dataset_column_type::null_);
    for (const auto & row : rows) {
        for (size_t index = 0; index < names.size() && index < row.size(); ++index) {
            const auto type = inferred_type(row[index]);
            if (type == common_agent_dataset_column_type::string) types[index] = type;
            else if (types[index] == common_agent_dataset_column_type::null_) types[index] = type;
            else if (types[index] != type) types[index] = common_agent_dataset_column_type::string;
        }
    }
    json worksheet = {
        {"name", dataset_name.empty() ? "CSV" : dataset_name},
        {"index", 0}, {"table_index", 0},
        {"node_id", "resource-node://csv/0"},
        {"header_mode", "explicit"}, {"header_confidence", 1.0},
        {"header_reason", "CSV supplied an explicit first-row header"},
        {"columns", json::array()}, {"rows", json::array()}};
    for (size_t index = 0; index < names.size(); ++index) {
        worksheet["columns"].push_back({
            {"name", names[index].empty() ? "column_" + std::to_string(index + 1) : names[index]},
            {"type", common_agent_dataset_column_type_name(types[index])},
            {"nullable", true}});
    }
    for (const auto & row : rows) {
        json object = json::object();
        for (size_t index = 0; index < names.size(); ++index) {
            const std::string value = index < row.size() ? row[index] : std::string();
            object[worksheet["columns"][index]["name"].get<std::string>()] =
                typed_value(value, types[index]);
        }
        worksheet["rows"].push_back(std::move(object));
    }
    worksheet_json = json({{"worksheets", json::array({std::move(worksheet)})}}).dump();
    error.clear();
    return true;
}

bool make_agent_document_table_catalog(
        const std::string & worksheet_json,
        common_agent_document_table_catalog & catalog,
        std::string & error) {
    const auto root = json::parse(worksheet_json, nullptr, false);
    if (!root.is_object() || !root.contains("worksheets") || !root["worksheets"].is_array()) {
        error = "document table catalog requires a worksheet envelope";
        return false;
    }
    catalog = {};
    if (root["worksheets"].size() > 64) {
        error = "document table catalog exceeds the bounded table limit";
        return false;
    }
    for (const auto & worksheet : root["worksheets"]) {
        if (!worksheet.is_object() || !worksheet.contains("name") || !worksheet["name"].is_string()) {
            error = "document table catalog contains an invalid table";
            return false;
        }
        common_agent_document_table_entry entry;
        entry.table_index = worksheet.value("table_index", worksheet.value("index", size_t(0)));
        entry.name = worksheet["name"].get<std::string>();
        entry.caption = worksheet.value("caption", std::string());
        entry.node_id = worksheet.value("node_id", "document-node://table/" + std::to_string(entry.table_index));
        catalog.tables.push_back(std::move(entry));
    }
    error.clear();
    return true;
}

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
        const auto worksheet_index = worksheet.contains("index") && worksheet["index"].is_number_unsigned()
            ? std::optional<size_t>(worksheet["index"].get<size_t>()) : std::nullopt;
        if (request.sheet_name.has_value() && request.sheet_name.value() != sheet_name) continue;
        if (request.sheet_index.has_value() && (!worksheet_index.has_value() || request.sheet_index.value() != worksheet_index.value())) continue;
        if (sheet_name.empty() || worksheet["columns"].size() > request.limits.max_columns ||
                worksheet["rows"].size() > request.limits.max_rows) {
            error = "worksheet shape exceeds host limits";
            return false;
        }
        common_agent_dataset_descriptor descriptor;
        descriptor.ref.uri = "dataset://import/" + stable_source_key(request.source_resource_uri) + "/" +
            std::to_string(imported.size()) + "/" + sheet_name;
        descriptor.ref.name = sheet_name;
        descriptor.ref.row_count = worksheet["rows"].size();
        descriptor.ref.column_count = worksheet["columns"].size();
        descriptor.ref.source_resource_uri = request.source_resource_uri;
        descriptor.ref.source_representation = request.source_representation.empty()
            ? "xlsx:worksheet" : request.source_representation;
        descriptor.source_workbook_name = request.source_workbook_name;
        descriptor.source_sheet_name = sheet_name;
        descriptor.source_sheet_index = worksheet_index;
        descriptor.source_range = worksheet.value("range", std::string());
        descriptor.source_object = worksheet.value("object", std::string());
        descriptor.origin.kind = descriptor.ref.source_representation == "xlsx:worksheet"
            ? "spreadsheet" : "document_table";
        descriptor.origin.source_representation_uri = worksheet.value(
            "semantic_resource_uri", request.source_representation_uri);
        descriptor.origin.source_node_id = worksheet.value("node_id", "document-node://table/" + std::to_string(worksheet_index.value_or(0)));
        descriptor.origin.table_index = worksheet.value("table_index", worksheet_index.value_or(0));
        descriptor.origin.caption = worksheet.value("caption", std::string());
        descriptor.origin.header_mode = worksheet.value("header_mode", std::string("explicit")) == "explicit"
            ? common_agent_table_header_mode::explicit_ : common_agent_table_header_mode::ambiguous;
        descriptor.origin.header_confidence = worksheet.value("header_confidence", 0.0);
        descriptor.origin.header_reason = worksheet.value("header_reason", std::string("Pandoc supplied an explicit table header"));
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
        if (!store.put_dataset_descriptor(descriptor, error)) return false;
        imported.push_back(std::move(descriptor));
    }
    if ((request.sheet_name.has_value() || request.sheet_index.has_value()) && imported.empty()) {
        error = "requested worksheet was not found";
        return false;
    }
    error.clear();
    return true;
}
