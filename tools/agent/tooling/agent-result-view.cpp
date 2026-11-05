#include "agent-result-view.h"
#include "agent/context/inverted-index-text.h"

#include <cstdint>
#include <cstddef>
#include <map>
#include <algorithm>
#include <cctype>
#include <nlohmann/json.hpp>
#include <set>
#include <string>
#include <utility>

// Host-declared, bounded projections of JSON response bodies. The input is
// always the original response; caller-owned metadata remains authoritative.
using agent_result_view_json = nlohmann::ordered_json;

inline bool agent_result_view_pointer(const std::string & value) {
    if (value.empty() || value[0] != '/' || value.size() > 1024) return false;
    for (size_t i = 0; i < value.size(); ++i) {
        if (value[i] == '~' && (i + 1 == value.size() || (value[i + 1] != '0' && value[i + 1] != '1'))) return false;
        if (value[i] == '~') ++i;
    }
    return true;
}

bool agent_result_view_validate(const std::string & policy, std::string & error) {
    const auto j = agent_result_view_json::parse(policy, nullptr, false);
    auto fail = [&](const std::string & reason) { error = reason; return false; };
    if (policy.size() > 32768 || !j.is_object()) return fail("result_view must be a bounded object");
    const std::set<std::string> allowed = {"kind", "items_pointer", "output_pointer", "fields", "derived_fields", "max_items", "max_text_bytes", "max_bytes", "materialize", "resource_scope"};
    for (const auto & e : j.items()) if (!allowed.count(e.key())) return fail("unknown result_view property: " + e.key());
    if (!j.contains("kind") || !j["kind"].is_string()) return fail("result_view kind is required");
    const auto kind = j["kind"].get<std::string>();
    if (kind != "object" && kind != "collection") return fail("result_view kind must be object or collection");
    if (kind == "collection" && (!j.contains("items_pointer") || !j["items_pointer"].is_string() || (!j["items_pointer"].get<std::string>().empty() && !agent_result_view_pointer(j["items_pointer"].get<std::string>())))) return fail("collection requires a valid items_pointer");
    if (j.contains("output_pointer") && (kind != "collection" || !j["output_pointer"].is_string() || !agent_result_view_pointer(j["output_pointer"].get<std::string>()))) return fail("invalid output_pointer");
    if (kind == "object" && (j.contains("items_pointer") || j.contains("max_items"))) return fail("object cannot declare collection settings");
    if (j.contains("resource_scope") && (!j["resource_scope"].is_string() ||
            (j["resource_scope"] != "turn" && j["resource_scope"] != "session"))) return fail("invalid resource_scope");
    if (j.contains("materialize") && (!j["materialize"].is_string() || (j["materialize"] != "resource" && j["materialize"] != "dataset_if_tabular"))) return fail("invalid materialize policy");
    for (const auto & limit : {std::pair<const char *, size_t>{"max_items", 100}, {"max_text_bytes", 32768}, {"max_bytes", 65536}}) {
        if (j.contains(limit.first) && (!j[limit.first].is_number_unsigned() || j[limit.first].get<size_t>() == 0 || j[limit.first].get<size_t>() > limit.second)) return fail(std::string("invalid ") + limit.first);
    }
    const std::set<std::string> reserved = {"resource_ref", "dataset", "source_resource", "materialized", "representation", "pagination", "host_item_parameter", "host_item_candidates", "host_item_candidates_truncated", "result_count", "preview_omitted_items", "view_truncated", "view_invalid_fields", "materialization_reason", "mime_type", "size_bytes", "rows", "columns", "resources", "dataset_refs"};
    std::set<std::string> names;
    if (!j.contains("fields") || !j["fields"].is_object() || j["fields"].size() > 64) return fail("fields must be a bounded object");
    auto field_name_ok = [&](const std::string & key) { return !key.empty() && key.size() <= 128 && !reserved.count(key) && names.insert(key).second; };
    for (const auto & e : j["fields"].items()) {
        if (!field_name_ok(e.key()) || !e.value().is_string() || !agent_result_view_pointer(e.value().get<std::string>())) return fail("invalid field: " + e.key());
    }
    if (j.contains("derived_fields")) {
        if (!j["derived_fields"].is_object() || j["derived_fields"].size() > 32) return fail("derived_fields must be a bounded object");
        for (const auto & e : j["derived_fields"].items()) {
            if (!field_name_ok(e.key()) || !e.value().is_object()) return fail("invalid derived field: " + e.key());
            const auto & d = e.value();
            if (d.size() != 2 || !d.contains("transform") || !d["transform"].is_string() || !d.contains("from")) return fail("invalid transform for " + e.key());
            const auto name = d["transform"].get<std::string>();
            if (name == "inverted_index_to_text") {
                if (reserved.count(e.key() + "_truncated") || j["fields"].contains(e.key() + "_truncated") || j["derived_fields"].contains(e.key() + "_truncated")) return fail("derived truncation field collision");
                if (!d["from"].is_string() || !agent_result_view_pointer(d["from"].get<std::string>())) return fail("invalid inverted index input");
            } else if (name == "first_nonempty_url") {
                if (!d["from"].is_array() || d["from"].empty() || d["from"].size() > 16) return fail("invalid URL inputs");
                for (const auto & path : d["from"]) if (!path.is_string() || !agent_result_view_pointer(path.get<std::string>())) return fail("invalid URL pointer");
                if (reserved.count(e.key() + "_source") || j["fields"].contains(e.key() + "_source") || j["derived_fields"].contains(e.key() + "_source")) return fail("derived source field collision");
            } else return fail("unknown transform: " + name);
        }
    }
    if (kind == "collection") {
        const auto input_path = j["items_pointer"].get<std::string>();
        const auto path = j.value("output_pointer", input_path.empty() ? std::string("/items") : input_path);
        for (const auto & name : reserved) if (path == "/" + name || path.rfind("/" + name + "/", 0) == 0) return fail("items_pointer collides with host metadata");
    }
    error.clear();
    return true;
}

inline const agent_result_view_json * agent_result_view_at(const agent_result_view_json & value, const std::string & path) {
    if (path.empty()) return &value;
    try { const agent_result_view_json::json_pointer p(path); return value.contains(p) ? &value.at(p) : nullptr; }
    catch (const nlohmann::json::exception &) { return nullptr; }
}

inline std::string agent_result_view_trim(std::string text, size_t limit, bool & truncated) {
    if (text.size() <= limit) return text;
    size_t end = limit;
    while (end && (static_cast<unsigned char>(text[end]) & 0xc0) == 0x80) --end;
    truncated = true;
    return text.substr(0, end);
}

inline agent_result_view_json agent_result_view_row(const agent_result_view_json & row, const agent_result_view_json & policy, bool & truncated, size_t & invalid_fields) {
    auto out = agent_result_view_json::object();
    const size_t text_limit = policy.value("max_text_bytes", size_t(1200));
    for (const auto & field : policy["fields"].items()) {
        const auto * value = agent_result_view_at(row, field.value().get<std::string>());
        if (!value || value->is_null()) continue;
        if (value->is_array() || value->is_object()) { ++invalid_fields; continue; }
        out[field.key()] = value->is_string() ? agent_result_view_json(agent_result_view_trim(value->get<std::string>(), text_limit, truncated)) : *value;
    }
    if (!policy.contains("derived_fields")) return out;
    for (const auto & field : policy["derived_fields"].items()) {
        const auto & d = field.value();
        if (d["transform"] == "inverted_index_to_text") {
            const auto * value = agent_result_view_at(row, d["from"].get<std::string>());
            if (value && !value->is_null() && !value->is_object()) { ++invalid_fields; continue; }
            if (value && value->is_object()) {
                bool limited = false;
                auto text = common_agent_inverted_index_text(*value, limited);
                text = agent_result_view_trim(std::move(text), text_limit, limited);
                truncated = truncated || limited;
                if (!text.empty()) out[field.key()] = std::move(text);
                if (limited) out[field.key() + "_truncated"] = true;
            }
        } else {
            for (const auto & path : d["from"]) {
                const auto * value = agent_result_view_at(row, path.get<std::string>());
                if (!value || value->is_null()) continue;
                if (!value->is_string()) { ++invalid_fields; continue; }
                const auto url = value->get<std::string>();
                if (url.size() > text_limit || (url.rfind("https://", 0) != 0 && url.rfind("http://", 0) != 0) ||
                        std::any_of(url.begin(), url.end(), [](unsigned char ch) { return std::isspace(ch) || std::iscntrl(ch); })) continue;
                out[field.key()] = url;
                out[field.key() + "_source"] = path;
                break;
            }
        }
    }
    return out;
}

bool agent_result_view_project(const std::string & policy_json, const std::string & source_json, std::string & output_json, std::string & error) {
    if (!agent_result_view_validate(policy_json, error)) return false;
    const auto policy = agent_result_view_json::parse(policy_json);
    const auto source = agent_result_view_json::parse(source_json, nullptr, false);
    auto output = agent_result_view_json::parse(output_json, nullptr, false);
    if (source.is_discarded()) { error = "result_view source is not JSON"; return false; }
    if (!output.is_object()) output = agent_result_view_json::object();
    const size_t max_bytes = policy.value("max_bytes", size_t(16384));
    bool truncated = false;
    size_t invalid_fields = 0;
    if (policy["kind"] == "object") {
        if (!source.is_object()) { error = "result_view expected an object"; return false; }
        auto row = agent_result_view_row(source, policy, truncated, invalid_fields);
        for (const auto & field : row.items()) {
            output[field.key()] = field.value();
            if (output.dump().size() > max_bytes) { output.erase(field.key()); truncated = true; break; }
        }
    } else {
        const auto path = policy["items_pointer"].get<std::string>();
        const auto output_path = policy.value("output_pointer", path.empty() ? std::string("/items") : path);
        const auto * rows = agent_result_view_at(source, path);
        if (!rows || !rows->is_array()) { error = "result_view collection not found"; return false; }
        agent_result_view_json preview = agent_result_view_json::array();
        const auto count = rows->size();
        for (const auto & row : *rows) {
            if (preview.size() >= policy.value("max_items", size_t(5))) { truncated = true; break; }
            if (!row.is_object()) { truncated = true; break; }
            preview.push_back(agent_result_view_row(row, policy, truncated, invalid_fields));
            auto candidate = output;
            try { candidate[agent_result_view_json::json_pointer(output_path)] = preview; }
            catch (const nlohmann::json::exception &) { error = "result_view cannot create collection path"; return false; }
            if (candidate.dump().size() > max_bytes) { preview.erase(preview.size() - 1); truncated = true; break; }
        }
        try { output[agent_result_view_json::json_pointer(output_path)] = preview; }
        catch (const nlohmann::json::exception &) { error = "result_view cannot create collection path"; return false; }
        output["result_count"] = count;
        output["preview_omitted_items"] = count - preview.size();
    }
    if (truncated) output["view_truncated"] = true;
    if (invalid_fields) output["view_invalid_fields"] = invalid_fields;
    output_json = output.dump();
    if (output_json.size() > max_bytes) {
        error = "result_view metadata exceeds configured byte limit";
        return false;
    }
    error.clear();
    return true;
}
