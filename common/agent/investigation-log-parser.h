#pragma once

#include <nlohmann/json.hpp>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

// A host-executed declarative parser. Field mapping keys are normalized field
// names and values are exact CSV headers or JSON pointers (including / escapes).
// Parsing never interprets source text as instructions or executable code.
struct common_investigation_log_page {
    nlohmann::ordered_json rows = nlohmann::ordered_json::array();
    nlohmann::ordered_json rejected = nlohmann::ordered_json::array();
    size_t next_record = 0;
    size_t rejected_count = 0;
    bool exhausted = false;
};

inline bool common_investigation_parse_log(const std::string & text,
        const std::string & format, const std::map<std::string, std::string> & fields,
        size_t start_record, size_t limit, common_investigation_log_page & result,
        std::string & error) {
    using json = nlohmann::ordered_json;
    result = {};
    error.clear();
    const std::set<std::string> allowed = {"@timestamp", "event.type", "event.id", "event.outcome", "user.id", "user.name",
        "host.id", "host.name", "source.ip", "destination.ip", "process.entity_id", "process.pid", "process.start",
        "process.parent.entity_id", "process.parent.pid", "process.parent.start", "file.path", "file.hash.sha256",
        "trace.id", "message.id"};
    if ((format != "jsonl" && format != "csv") || fields.empty() || fields.size() > 32 ||
            !fields.count("event.type") || limit == 0 || limit > 500 || text.size() > 4 * 1024 * 1024) {
        error = "log parser requires jsonl/csv, an explicit event.type mapping and bounded input"; return false;
    }
    for (const auto & field : fields) if (!allowed.count(field.first) || field.second.empty() || field.second.size() > 256) {
        error = "log parser mapping contains an unsupported field or invalid source path"; return false;
    }
    if (format == "jsonl") for (const auto & field : fields) if (field.second.front() == '/') {
        try { (void) json::json_pointer(field.second); }
        catch (const std::exception &) { error = "log parser mapping contains an invalid JSON pointer"; return false; }
    }
    size_t offset = 0, line = 1, record = 0;
    std::vector<std::string> headers;
    bool header_read = format != "csv";
    while (offset < text.size()) {
        const size_t begin = offset, first_line = line;
        bool valid = true;
        std::vector<std::string> cells;
        if (format == "jsonl") {
            const auto end = text.find('\n', offset);
            offset = end == std::string::npos ? text.size() : end + 1;
            if (end != std::string::npos) ++line;
        } else {
            std::string cell;
            bool quoted = false, closed = false;
            while (offset < text.size()) {
                const char c = text[offset++];
                if (quoted) {
                    if (c == '"') {
                        if (offset < text.size() && text[offset] == '"') { cell += '"'; ++offset; }
                        else { quoted = false; closed = true; }
                    } else { cell += c; if (c == '\n') ++line; }
                } else if (c == ',') { cells.push_back(cell); cell.clear(); closed = false; }
                else if (c == '\n' || c == '\r') {
                    if (c == '\r' && offset < text.size() && text[offset] == '\n') ++offset;
                    ++line; break;
                } else if (c == '"' && cell.empty() && !closed) quoted = true;
                else { if (closed || c == '"') valid = false; cell += c; }
            }
            if (quoted) valid = false;
            cells.push_back(std::move(cell));
        }
        size_t length = offset - begin;
        while (length && (text[begin + length - 1] == '\n' || text[begin + length - 1] == '\r')) --length;
        if (length == 0) continue;
        if (!header_read) {
            headers = cells; header_read = true;
            std::set<std::string> unique;
            for (const auto & h : headers) if (h.empty() || !unique.insert(h).second) valid = false;
            for (const auto & field : fields) if (!unique.count(field.second)) valid = false;
            if (!valid || length > 65536) { error = "CSV headers must be unique and include every mapped field"; return false; }
            continue;
        }
        if (record < start_record) { ++record; continue; }
        if (record - start_record >= limit) { offset = begin; break; }
        ++record;
        json input, normalized = json::object();
        if (length > 65536) valid = false;
        if (format == "jsonl" && valid) {
            try {
                input = json::parse(text.substr(begin, length), [](int depth, json::parse_event_t, json &) {
                    if (depth > 64) throw std::runtime_error("log record nesting exceeds the parser limit");
                    return true;
                }, false);
                if (!input.is_object()) valid = false;
            } catch (const std::exception &) { valid = false; }
        }
        if (format == "csv" && cells.size() != headers.size()) valid = false;
        if (valid) for (const auto & field : fields) {
            json value;
            if (format == "csv") {
                for (size_t i = 0; i < headers.size(); ++i) if (headers[i] == field.second) { value = cells[i]; break; }
            } else if (field.second.front() == '/') {
                const json::json_pointer pointer(field.second);
                if (input.contains(pointer)) value = input.at(pointer);
            } else if (input.contains(field.second)) value = input.at(field.second);
            if (value.is_object() || value.is_array() || value.is_boolean() ||
                    (value.is_string() && value.get_ref<const std::string &>().size() > 2048)) { valid = false; break; }
            normalized[field.first] = std::move(value);
        }
        if (!normalized.contains("event.type") || !normalized["event.type"].is_string() ||
                normalized["event.type"].get<std::string>().empty()) valid = false;
        if (!valid) {
            ++result.rejected_count;
            if (result.rejected.size() < 32) result.rejected.push_back({{"record",record - 1},
                {"line",first_line},{"byte_offset",begin},{"byte_length",length},
                {"reason","invalid_record_or_mapped_field_type"}});
        } else {
            // File positions, not untrusted event IDs, determine record identity.
            normalized["@source.id"] = std::to_string(begin);
            normalized["@source.byte_offset"] = begin;
            normalized["@source.byte_length"] = length;
            normalized["@source.line"] = first_line;
            result.rows.push_back(std::move(normalized));
        }
    }
    if (!header_read || start_record > record) { error = "log parser cursor exceeds the input records"; return false; }
    result.next_record = record;
    result.exhausted = offset >= text.size();
    return true;
}
