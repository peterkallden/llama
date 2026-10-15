#include "agent-dataset-importer.h"

#include <nlohmann/json.hpp>
#include <libxml/parser.h>
#include <libxml/tree.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <limits>
#include <map>
#include <set>
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

common_agent_dataset_column_type json_type(const json & value) {
    if (value.is_null()) return common_agent_dataset_column_type::null_;
    if (value.is_boolean()) return common_agent_dataset_column_type::boolean;
    if (value.is_number_integer() || value.is_number_unsigned()) return common_agent_dataset_column_type::integer;
    if (value.is_number()) return common_agent_dataset_column_type::decimal;
    if (value.is_string()) return common_agent_dataset_column_type::string;
    return common_agent_dataset_column_type::unknown;
}

common_agent_dataset_column_type merge_type(
        common_agent_dataset_column_type left,
        common_agent_dataset_column_type right) {
    if (left == common_agent_dataset_column_type::null_) return right;
    if (right == common_agent_dataset_column_type::null_) return left;
    if (left == right) return left;
    if ((left == common_agent_dataset_column_type::integer || left == common_agent_dataset_column_type::decimal) &&
            (right == common_agent_dataset_column_type::integer || right == common_agent_dataset_column_type::decimal)) {
        return common_agent_dataset_column_type::decimal;
    }
    return common_agent_dataset_column_type::string;
}

bool make_json_worksheet(
        const std::string & name,
        const std::vector<json> & rows,
        const common_agent_dataset_limits & limits,
        std::string & worksheet_json,
        std::string & error) {
    if (rows.size() > limits.max_rows) {
        error = "structured resource exceeds the host row limit";
        return false;
    }
    std::vector<std::string> columns;
    std::map<std::string, common_agent_dataset_column_type> types;
    std::set<std::string> seen;
    for (const auto & row : rows) {
        if (!row.is_object()) {
            error = "structured dataset rows must be objects";
            return false;
        }
        for (const auto & field : row.items()) {
            if (field.value().is_object() || field.value().is_array()) {
                error = "structured dataset columns must contain scalar or null values";
                return false;
            }
            if (seen.insert(field.key()).second) columns.push_back(field.key());
            auto & type = types[field.key()];
            type = merge_type(type, json_type(field.value()));
        }
        if (columns.size() > limits.max_columns) {
            error = "structured resource exceeds the host column limit";
            return false;
        }
    }
    if (columns.empty() && !rows.empty()) {
        error = "structured dataset has no scalar columns";
        return false;
    }
    if (rows.empty()) {
        error = "structured dataset has no records from which to infer columns";
        return false;
    }
    json worksheet = {{"name", name.empty() ? "JSON" : name}, {"index", 0},
        {"table_index", 0}, {"node_id", "resource-node://structured/0"},
        {"header_mode", "explicit"}, {"header_confidence", 1.0},
        {"header_reason", "Host parser identified structured records"},
        {"columns", json::array()}, {"rows", json::array()}};
    for (const auto & column : columns) {
        const auto type = types[column];
        worksheet["columns"].push_back({{"name", column},
            {"type", common_agent_dataset_column_type_name(type)}, {"nullable", true}});
    }
    for (auto row : rows) {
        for (const auto & column : columns) if (!row.contains(column)) row[column] = nullptr;
        worksheet["rows"].push_back(std::move(row));
    }
    if (rows.size() > limits.max_cells / std::max<size_t>(size_t(1), columns.size())) {
        error = "structured resource exceeds the host cell limit";
        return false;
    }
    worksheet_json = json({{"worksheets", json::array({std::move(worksheet)})}}).dump();
    error.clear();
    return true;
}

std::string xml_local_name(const xmlNode * node) {
    return node == nullptr || node->name == nullptr
        ? std::string() : reinterpret_cast<const char *>(node->name);
}

std::string xml_content(const xmlNode * node) {
    if (node == nullptr) return {};
    xmlChar * content = xmlNodeGetContent(const_cast<xmlNode *>(node));
    if (content == nullptr) return {};
    std::string result(reinterpret_cast<const char *>(content));
    xmlFree(content);
    const auto first = result.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    const auto last = result.find_last_not_of(" \t\r\n");
    return result.substr(first, last - first + 1);
}

std::string xml_attribute(const xmlNode * node, const char * name) {
    if (node == nullptr) return {};
    xmlChar * value = xmlGetProp(const_cast<xmlNode *>(node), BAD_CAST name);
    if (value == nullptr) return {};
    std::string result(reinterpret_cast<const char *>(value));
    xmlFree(value);
    return result;
}

const xmlNode * xml_child(const xmlNode * node, const char * name) {
    for (const auto * child = node == nullptr ? nullptr : node->children;
            child != nullptr; child = child->next) {
        if (child->type == XML_ELEMENT_NODE && xml_local_name(child) == name) return child;
    }
    return nullptr;
}

std::string xml_named_child_text(const xmlNode * node, const char * name) {
    return xml_content(xml_child(node, name));
}

std::string xml_english_name(const xmlNode * node) {
    for (const auto * child = node == nullptr ? nullptr : node->children;
            child != nullptr; child = child->next) {
        if (child->type != XML_ELEMENT_NODE || (xml_local_name(child) != "Name" &&
                xml_local_name(child) != "title" && xml_local_name(child) != "Title")) continue;
        xmlChar * xml_language = xmlGetNsProp(const_cast<xmlNode *>(child), BAD_CAST "lang",
            BAD_CAST "http://www.w3.org/XML/1998/namespace");
        std::string language = xml_language == nullptr ? std::string() :
            reinterpret_cast<const char *>(xml_language);
        if (xml_language != nullptr) xmlFree(xml_language);
        if (language.empty()) language = xml_attribute(child, "lang");
        if (language.empty()) language = xml_attribute(child, "language");
        if (language.empty() || language == "en") return xml_content(child);
    }
    return {};
}

bool xml_tree_within_limits(const xmlNode * node, size_t depth, size_t & nodes,
        const common_agent_dataset_limits & limits, std::string & error) {
    if (depth > 64 || ++nodes > limits.max_cells + limits.max_rows * 8) {
        error = "XML resource exceeds the host structure limits";
        return false;
    }
    for (const auto * child = node->children; child != nullptr; child = child->next) {
        if (child->type == XML_ELEMENT_NODE && !xml_tree_within_limits(child, depth + 1, nodes, limits, error)) return false;
    }
    return true;
}

bool normalize_eurostat_toc(
        const xmlNode * root,
        const std::string & dataset_name,
        const common_agent_dataset_limits & limits,
        std::string & worksheet_json,
        std::string & error) {
    std::vector<json> rows;
    std::vector<std::string> category;
    std::function<bool(const xmlNode *)> visit = [&](const xmlNode * node) {
        for (const auto * child = node->children; child != nullptr; child = child->next) {
            if (child->type != XML_ELEMENT_NODE) continue;
            const auto kind = xml_local_name(child);
            if (kind == "branch") {
                const auto label = xml_english_name(child);
                const auto code = xml_named_child_text(child, "code");
                category.push_back(label.empty() ? code : label);
                if (!visit(child)) return false;
                category.pop_back();
            } else if (kind == "children") {
                if (!visit(child)) return false;
            } else if (kind == "leaf" && xml_attribute(child, "type") == "table") {
                const auto code = xml_named_child_text(child, "code");
                const auto title = xml_english_name(child);
                if (code.empty() || title.empty()) continue;
                if (rows.size() >= limits.max_rows) {
                    error = "Eurostat table of contents exceeds the host row limit";
                    return false;
                }
                std::string path;
                for (const auto & part : category) {
                    if (part.empty()) continue;
                    if (!path.empty()) path += " / ";
                    path += part;
                }
                json row = {{"dataset_code", code}, {"title", title}, {"category_path", path}};
                for (const auto & pair : {std::pair<const char *, const char *> {"last_update", "lastUpdate"},
                        {"data_start", "dataStart"}, {"data_end", "dataEnd"}, {"values", "values"}}) {
                    const auto value = xml_named_child_text(child, pair.second);
                    if (value.empty()) continue;
                    if (std::string(pair.first) == "values") {
                        char * end = nullptr;
                        const auto parsed = std::strtoull(value.c_str(), &end, 10);
                        row[pair.first] = end != nullptr && *end == '\0' ? json(parsed) : json(value);
                    } else row[pair.first] = value;
                }
                const auto metadata = xml_child(child, "metadata");
                if (metadata != nullptr) row["metadata_url"] = xml_content(metadata);
                rows.push_back(std::move(row));
            }
        }
        return true;
    };
    if (!visit(root)) return false;
    if (rows.empty()) {
        error = "Eurostat table of contents contains no table entries";
        return false;
    }
    return make_json_worksheet(dataset_name.empty() ? "Eurostat table of contents" : dataset_name,
        rows, limits, worksheet_json, error);
}

bool normalize_sdmx_structure(
        const xmlNode * root,
        const std::string & dataset_name,
        const common_agent_dataset_limits & limits,
        std::string & worksheet_json,
        std::string & error) {
    std::vector<json> rows;
    std::function<void(const xmlNode *, std::string, std::string, std::string, std::string)> visit =
        [&](const xmlNode * node, std::string dataflow, std::string structure,
                std::string codelist, std::string constraint) {
            if (node == nullptr || !error.empty()) return;
            const std::string name = xml_local_name(node);
            std::string next_dataflow = dataflow;
            std::string next_structure = structure;
            std::string next_codelist = codelist;
            std::string next_constraint = constraint;
            if (name == "Dataflow") next_dataflow = xml_attribute(node, "id");
            if (name == "DataStructure") next_structure = xml_attribute(node, "id");
            if (name == "Codelist") next_codelist = xml_attribute(node, "id");
            if (name == "ContentConstraint") next_constraint = xml_attribute(node, "id");
            json row;
            bool emit = false;
            const auto id = xml_attribute(node, "id");
            if (name == "Dataflow" || name == "DataStructure" || name == "Dimension" ||
                    name == "TimeDimension" || name == "PrimaryMeasure" || name == "Attribute" ||
                    name == "Code" || name == "ContentConstraint" || name == "KeyValue" || name == "Value") {
                emit = true;
                row["artifact_kind"] = name;
                row["artifact_id"] = id;
                row["dataflow_id"] = next_dataflow;
                row["structure_id"] = next_structure;
                row["codelist_id"] = next_codelist;
                row["constraint_id"] = next_constraint;
                const auto label = xml_english_name(node);
                if (!label.empty()) row["label"] = label;
                const auto agency = xml_attribute(node, "agencyID");
                if (!agency.empty()) row["agency_id"] = agency;
                const auto version = xml_attribute(node, "version");
                if (!version.empty()) row["version"] = version;
                const auto position = xml_attribute(node, "position");
                if (!position.empty()) {
                    char * end = nullptr;
                    const auto value = std::strtoull(position.c_str(), &end, 10);
                    row["position"] = end != nullptr && *end == '\0' ? json(value) : json(position);
                }
                if (name == "Dimension" || name == "TimeDimension" || name == "PrimaryMeasure" || name == "Attribute") {
                    row["dimension_id"] = id;
                    row["parent_id"] = next_structure;
                    std::function<void(const xmlNode *)> find_ref = [&](const xmlNode * candidate) {
                        if (candidate == nullptr || !row.value("codelist_id", std::string()).empty()) return;
                        if (xml_local_name(candidate) == "Ref") {
                            const auto ref = xml_attribute(candidate, "id");
                            if (!ref.empty()) row["codelist_id"] = ref;
                            return;
                        }
                        for (const auto * part = candidate->children; part != nullptr; part = part->next)
                            if (part->type == XML_ELEMENT_NODE) find_ref(part);
                    };
                    find_ref(node);
                } else if (name == "Code") {
                    row["code"] = id;
                    row["parent_id"] = next_codelist;
                } else if (name == "KeyValue") {
                    row["dimension_id"] = xml_attribute(node, "id");
                    row["parent_id"] = next_constraint;
                } else if (name == "Value") {
                    row["code"] = xml_content(node);
                    row["parent_id"] = next_constraint;
                }
                if (name == "Dataflow") {
                    std::function<void(const xmlNode *)> find_ref = [&](const xmlNode * candidate) {
                        if (candidate == nullptr || row.contains("structure_id_ref")) return;
                        if (xml_local_name(candidate) == "Ref") {
                            const auto ref = xml_attribute(candidate, "id");
                            if (!ref.empty()) row["structure_id_ref"] = ref;
                            return;
                        }
                        for (const auto * part = candidate->children; part != nullptr; part = part->next)
                            if (part->type == XML_ELEMENT_NODE) find_ref(part);
                    };
                    find_ref(node);
                }
                if (rows.size() >= limits.max_rows) error = "SDMX structure exceeds the host row limit";
                else rows.push_back(std::move(row));
            }
            for (const auto * child = node->children; child != nullptr; child = child->next)
                if (child->type == XML_ELEMENT_NODE) visit(child, next_dataflow, next_structure,
                    next_codelist, next_constraint);
        };
    visit(root, {}, {}, {}, {});
    if (!error.empty()) return false;
    if (rows.empty()) {
        error = "SDMX XML contains no supported structural records";
        return false;
    }
    return make_json_worksheet(dataset_name.empty() ? "SDMX structure" : dataset_name,
        rows, limits, worksheet_json, error);
}

bool normalize_generic_xml_records(
        const xmlNode * root,
        const std::string & dataset_name,
        const common_agent_dataset_limits & limits,
        std::string & worksheet_json,
        std::string & error) {
    struct group { const xmlNode * parent = nullptr; std::string name; std::vector<const xmlNode *> nodes; size_t depth = 0; };
    std::vector<group> groups;
    std::function<void(const xmlNode *, size_t)> find_groups = [&](const xmlNode * parent, size_t depth) {
        if (depth > 64) return;
        for (const auto * child = parent->children; child != nullptr; child = child->next) {
            if (child->type != XML_ELEMENT_NODE) continue;
            auto found = std::find_if(groups.begin(), groups.end(), [&](const group & value) {
                return value.parent == parent && value.name == xml_local_name(child);
            });
            if (found == groups.end()) groups.push_back({parent, xml_local_name(child), {child}, depth});
            else found->nodes.push_back(child);
            find_groups(child, depth + 1);
        }
    };
    find_groups(root, 0);
    std::vector<const group *> candidates;
    size_t min_depth = std::numeric_limits<size_t>::max();
    for (const auto & value : groups) if (value.nodes.size() >= 2) min_depth = std::min(min_depth, value.depth);
    for (const auto & value : groups) if (value.nodes.size() >= 2 && value.depth == min_depth) candidates.push_back(&value);
    if (candidates.size() != 1) {
        error = candidates.empty() ? "XML contains no repeated record elements" :
            "XML has multiple equally plausible repeated record groups";
        return false;
    }
    const auto & records = candidates.front()->nodes;
    if (records.size() > limits.max_rows) {
        error = "XML record group exceeds the host row limit";
        return false;
    }
    std::vector<json> rows;
    rows.reserve(records.size());
    for (const auto * record : records) {
        json row = json::object();
        for (const auto * attr = record->properties; attr != nullptr; attr = attr->next) {
            xmlChar * content = xmlNodeListGetString(record->doc, attr->children, 1);
            if (content != nullptr) {
                row["@" + std::string(reinterpret_cast<const char *>(attr->name))] =
                    reinterpret_cast<const char *>(content);
                xmlFree(content);
            }
        }
        std::set<std::string> child_names;
        for (const auto * child = record->children; child != nullptr; child = child->next) {
            if (child->type != XML_ELEMENT_NODE) continue;
            const auto key = xml_local_name(child);
            if (!child_names.insert(key).second) {
                error = "XML record has repeated nested fields; use a format-specific projection";
                return false;
            }
            for (const auto * nested = child->children; nested != nullptr; nested = nested->next) {
                if (nested->type == XML_ELEMENT_NODE) {
                    error = "XML record has nested structures; use a format-specific projection";
                    return false;
                }
            }
            row[key] = xml_content(child);
        }
        rows.push_back(std::move(row));
    }
    return make_json_worksheet(dataset_name, rows, limits, worksheet_json, error);
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

bool normalize_agent_json_dataset(
        const std::string & json_text,
        const std::string & dataset_name,
        std::string & worksheet_json,
        std::string & error,
        const common_agent_dataset_limits & limits) {
    const auto root = json::parse(json_text, nullptr, false);
    if (root.is_discarded()) {
        error = "structured JSON resource is invalid";
        return false;
    }
    const json * records = nullptr;
    if (root.is_array()) records = &root;
    else if (root.is_object() && root.value("class", std::string()) == "dataset" &&
            root.contains("dimension") && root.contains("value")) {
        if (!root["dimension"].is_object() || !root["value"].is_array() && !root["value"].is_object() ||
                !root.contains("id") || !root["id"].is_array() || !root.contains("size") ||
                !root["size"].is_array() || root["id"].size() != root["size"].size()) {
            error = "JSON-stat dataset has invalid dimensions, sizes or values";
            return false;
        }
        const size_t dimension_count = root["id"].size();
        if (dimension_count == 0 || dimension_count > limits.max_columns / 2) {
            error = "JSON-stat dimension count exceeds the host column limit";
            return false;
        }
        std::vector<std::string> dimension_ids;
        std::vector<std::vector<std::string>> codes;
        std::vector<std::map<std::string, std::string>> labels;
        std::set<std::string> unique_ids;
        size_t cell_count = 1;
        for (size_t index = 0; index < dimension_count; ++index) {
            if (!root["id"][index].is_string() || !root["size"][index].is_number_unsigned()) {
                error = "JSON-stat dimension id or size is invalid";
                return false;
            }
            const auto id = root["id"][index].get<std::string>();
            if (id.empty() || !unique_ids.insert(id).second || id == "value" || id == "status" ||
                    !root["dimension"].contains(id) || !root["dimension"][id].is_object()) {
                error = "JSON-stat dimension ids must be unique and usable as dataset columns";
                return false;
            }
            const size_t size = root["size"][index].get<size_t>();
            if (size == 0 || cell_count > limits.max_cells / size) {
                error = "JSON-stat Cartesian cell count exceeds the host cell limit";
                return false;
            }
            cell_count *= size;
            const auto & dimension = root["dimension"][id];
            if (!dimension.contains("category") || !dimension["category"].is_object() ||
                    !dimension["category"].contains("index")) {
                error = "JSON-stat dimension is missing category indexes";
                return false;
            }
            const auto & index_map = dimension["category"]["index"];
            std::vector<std::pair<size_t, std::string>> ordered_codes;
            if (index_map.is_array()) {
                for (size_t code_index = 0; code_index < index_map.size(); ++code_index) {
                    if (!index_map[code_index].is_string()) {
                        error = "JSON-stat category index array contains a non-string code";
                        return false;
                    }
                    ordered_codes.emplace_back(code_index, index_map[code_index].get<std::string>());
                }
            } else if (index_map.is_object()) {
                for (const auto & item : index_map.items()) {
                    if (!item.value().is_number_unsigned()) {
                        error = "JSON-stat category index map contains an invalid index";
                        return false;
                    }
                    ordered_codes.emplace_back(item.value().get<size_t>(), item.key());
                }
            } else {
                error = "JSON-stat category index must be an array or object";
                return false;
            }
            std::sort(ordered_codes.begin(), ordered_codes.end());
            if (ordered_codes.size() != size) {
                error = "JSON-stat category count does not match the declared dimension size";
                return false;
            }
            std::vector<std::string> dimension_codes(size);
            for (size_t code_index = 0; code_index < ordered_codes.size(); ++code_index) {
                if (ordered_codes[code_index].first != code_index) {
                    error = "JSON-stat category indexes are not contiguous";
                    return false;
                }
                dimension_codes[code_index] = ordered_codes[code_index].second;
            }
            std::map<std::string, std::string> dimension_labels;
            if (dimension["category"].contains("label") && dimension["category"]["label"].is_object()) {
                for (const auto & item : dimension["category"]["label"].items())
                    if (item.value().is_string()) dimension_labels[item.key()] = item.value().get<std::string>();
            }
            dimension_ids.push_back(id);
            codes.push_back(std::move(dimension_codes));
            labels.push_back(std::move(dimension_labels));
        }
        const auto & values = root["value"];
        const bool sparse = values.is_object();
        if ((!sparse && values.size() != cell_count) ||
                (sparse && values.size() > limits.max_rows)) {
            error = "JSON-stat value count does not match dimensions or exceeds the host row limit";
            return false;
        }
        const json * statuses = root.contains("status") ? &root["status"] : nullptr;
        if (statuses != nullptr && ((!statuses->is_array() && !statuses->is_object()) ||
                (!statuses->is_object() && statuses->size() != cell_count))) {
            error = "JSON-stat status values do not match the value representation";
            return false;
        }
        std::vector<std::pair<size_t, json>> indexed_values;
        if (sparse) {
            indexed_values.reserve(values.size());
            for (const auto & item : values.items()) {
                char * end = nullptr;
                const auto cell = std::strtoull(item.key().c_str(), &end, 10);
                if (end == nullptr || *end != '\0' || cell >= cell_count ||
                        item.value().is_object() || item.value().is_array()) {
                    error = "JSON-stat sparse values contain an invalid cell index or value";
                    return false;
                }
                indexed_values.emplace_back(static_cast<size_t>(cell), item.value());
            }
            std::sort(indexed_values.begin(), indexed_values.end(), [](const auto & left, const auto & right) {
                return left.first < right.first;
            });
        } else {
            indexed_values.reserve(values.size());
            for (size_t cell = 0; cell < values.size(); ++cell) {
                if (values[cell].is_object() || values[cell].is_array()) {
                    error = "JSON-stat values must be scalar or null";
                    return false;
                }
                indexed_values.emplace_back(cell, values[cell]);
            }
        }
        if (indexed_values.size() > limits.max_rows) {
            error = "JSON-stat observations exceed the host row limit";
            return false;
        }
        std::vector<json> rows;
        rows.reserve(indexed_values.size());
        for (const auto & indexed : indexed_values) {
            size_t remainder = indexed.first;
            std::vector<size_t> coordinates(dimension_count);
            for (size_t reverse = dimension_count; reverse-- > 0;) {
                coordinates[reverse] = remainder % codes[reverse].size();
                remainder /= codes[reverse].size();
            }
            json row = json::object();
            for (size_t dimension = 0; dimension < dimension_count; ++dimension) {
                const auto & code = codes[dimension][coordinates[dimension]];
                row[dimension_ids[dimension]] = code;
                const auto label = labels[dimension].find(code);
                if (label != labels[dimension].end()) row[dimension_ids[dimension] + "_label"] = label->second;
            }
            row["value"] = indexed.second;
            if (statuses != nullptr) {
                const json * status = nullptr;
                if (statuses->is_array()) status = &(*statuses)[indexed.first];
                else {
                    const auto found = statuses->find(std::to_string(indexed.first));
                    if (found != statuses->end()) status = &found.value();
                }
                row["status"] = status == nullptr ? json(nullptr) : *status;
            }
            rows.push_back(std::move(row));
        }
        if (!make_json_worksheet(dataset_name, rows, limits, worksheet_json, error)) return false;
        auto normalized = json::parse(worksheet_json, nullptr, false);
        if (normalized.is_object() && normalized.contains("worksheets") && normalized["worksheets"].is_array() &&
                !normalized["worksheets"].empty()) {
            normalized["worksheets"][0]["source_format"] = "json-stat-2.0";
            worksheet_json = normalized.dump();
        }
        return true;
    }

    if (root.is_object()) {
        if (root.contains("results") && root["results"].is_array()) records = &root["results"];
        else if (root.contains("items") && root["items"].is_array()) records = &root["items"];
        else if (root.contains("data") && root["data"].is_array()) records = &root["data"];
    }
    if (records == nullptr || !records->is_array()) {
        error = "JSON resource is neither a JSON-stat dataset nor a supported array of records";
        return false;
    }
    std::vector<json> rows;
    if (records->size() > limits.max_rows) {
        error = "JSON collection exceeds the host row limit";
        return false;
    }
    rows.reserve(records->size());
    for (const auto & item : *records) {
        if (!item.is_object()) {
            error = "JSON collection contains a non-object row";
            return false;
        }
        rows.push_back(item);
    }
    return make_json_worksheet(dataset_name, rows, limits, worksheet_json, error);
}

bool normalize_agent_xml_dataset(
        const std::string & xml_text,
        const std::string & dataset_name,
        std::string & worksheet_json,
        std::string & error,
        const common_agent_dataset_limits & limits) {
    if (xml_text.empty() || xml_text.size() > limits.max_source_bytes ||
            xml_text.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
        error = "XML source is empty or exceeds the host source-byte limit";
        return false;
    }
    const int flags = XML_PARSE_NONET | XML_PARSE_NOBLANKS | XML_PARSE_NOERROR |
        XML_PARSE_NOWARNING | XML_PARSE_COMPACT | XML_PARSE_NSCLEAN;
    xmlDocPtr document = xmlReadMemory(xml_text.data(), static_cast<int>(xml_text.size()),
        "agent-resource.xml", nullptr, flags);
    if (document == nullptr) {
        error = "XML resource is malformed";
        return false;
    }
    struct document_guard { xmlDocPtr value; ~document_guard() { xmlFreeDoc(value); } } guard{document};
    if (document->intSubset != nullptr || document->extSubset != nullptr) {
        error = "XML DTDs and entity declarations are not accepted";
        return false;
    }
    const xmlNode * root = xmlDocGetRootElement(document);
    if (root == nullptr) {
        error = "XML resource has no document element";
        return false;
    }
    size_t node_count = 0;
    if (!xml_tree_within_limits(root, 0, node_count, limits, error)) return false;
    const std::string namespace_uri = root->ns == nullptr || root->ns->href == nullptr
        ? std::string() : reinterpret_cast<const char *>(root->ns->href);
    if (namespace_uri.find("eurostat.navtree") != std::string::npos) {
        return normalize_eurostat_toc(root, dataset_name, limits, worksheet_json, error);
    }
    if (namespace_uri.find("sdmx") != std::string::npos || xml_local_name(root) == "Structure") {
        return normalize_sdmx_structure(root, dataset_name, limits, worksheet_json, error);
    }
    return normalize_generic_xml_records(root, dataset_name, limits, worksheet_json, error);
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
        descriptor.origin.kind = request.origin_kind;
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
