#include "agent-clangd-projection.h"

#include <cstdint>
#include <exception>
#include <utility>

namespace {
using json = agent_clangd_json;

int hex_digit(char value) {
    if (value >= '0' && value <= '9') return value - '0';
    if (value >= 'a' && value <= 'f') return value - 'a' + 10;
    if (value >= 'A' && value <= 'F') return value - 'A' + 10;
    return -1;
}

bool local_uri_path(const std::string & uri, std::filesystem::path & path) {
    std::string encoded;
    if (uri.rfind("file:///", 0) == 0) encoded = uri.substr(7);
    else if (uri.rfind("file://localhost/", 0) == 0) encoded = uri.substr(16);
    else return false;
    std::string decoded;
    for (size_t i = 0; i < encoded.size(); ++i) {
        char byte = encoded[i];
        if (byte == '%') {
            if (i + 2 >= encoded.size()) return false;
            const auto high = hex_digit(encoded[i + 1]), low = hex_digit(encoded[i + 2]);
            if (high < 0 || low < 0) return false;
            byte = static_cast<char>(high * 16 + low);
            i += 2;
        } else if (byte == '?' || byte == '#') return false;
        if (byte == '\0') return false;
        decoded.push_back(byte);
    }
#ifdef _WIN32
    if (decoded.size() > 3 && decoded[0] == '/' && decoded[2] == ':') decoded.erase(0, 1);
#endif
    try {
        (void) json(decoded).dump();
        path = std::filesystem::u8path(decoded);
    } catch (const std::exception &) { return false; }
    return path.is_absolute();
}

bool coordinate(const json & value, const char * field, uint64_t & result) {
    if (!value.is_object() || !value.contains(field) || !value[field].is_number_integer()) return false;
    if (value[field].is_number_unsigned()) result = value[field].get<uint64_t>();
    else {
        const auto number = value[field].get<int64_t>();
        if (number < 0) return false;
        result = static_cast<uint64_t>(number);
    }
    return result < 1000000;
}

bool position(const json & range, const char * field, uint64_t & line, uint64_t & column) {
    return range.is_object() && range.contains(field) &&
        coordinate(range[field], "line", line) && coordinate(range[field], "character", column);
}
}

bool agent_clangd_repository_path(const std::string & repository_root, const std::filesystem::path & input,
        std::filesystem::path & output, std::string & error) {
    if (repository_root.empty() || input.generic_u8string().find('\0') != std::string::npos) {
        error = "semantic path requires a host repository and a valid path"; return false;
    }
    std::error_code fs_error;
    const auto root = std::filesystem::weakly_canonical(repository_root, fs_error);
    if (fs_error) { error = "repository root is unavailable"; return false; }
    const auto candidate = std::filesystem::weakly_canonical(input.is_absolute() ? input : root / input, fs_error);
    if (fs_error) { error = "semantic path could not be resolved"; return false; }
    const auto relative = candidate.lexically_relative(root);
    if ((relative.empty() && candidate != root) || relative.is_absolute() ||
            (!relative.empty() && *relative.begin() == "..")) {
        error = "semantic path is outside the host repository"; return false;
    }
    output = candidate;
    error.clear(); return true;
}

bool agent_clangd_project_location(const json & input, const std::string & repository_root,
        json & output, std::string & error) {
    output = nullptr;
    if (!input.is_object() || repository_root.empty()) { error = "invalid semantic location"; return false; }
    const auto uri_key = input.contains("uri") ? "uri" : "targetUri";
    const auto range_key = input.contains("selectionRange") ? "selectionRange" :
        input.contains("targetSelectionRange") ? "targetSelectionRange" :
        input.contains("range") ? "range" : "targetRange";
    uint64_t line = 0, column = 0;
    if (!input.contains(uri_key) || !input[uri_key].is_string() || !input.contains(range_key) ||
            !position(input[range_key], "start", line, column)) {
        error = "semantic location lacks a valid URI or source position"; return false;
    }
    std::filesystem::path decoded;
    if (!local_uri_path(input[uri_key].get_ref<const std::string &>(), decoded)) {
        error = "semantic location is not a local file URI"; return false;
    }
    std::error_code fs_error;
    const auto root = std::filesystem::weakly_canonical(repository_root, fs_error);
    if (fs_error) { error = "repository root is unavailable"; return false; }
    std::filesystem::path candidate;
    if (!agent_clangd_repository_path(repository_root, decoded, candidate, error)) return false;
    const auto relative = candidate.lexically_relative(root);
    if (relative.empty() || relative == ".") { error = "semantic location is not a source file"; return false; }
    output = {{"uri", agent_clangd_file_uri(candidate)}, {"path", relative.generic_u8string()},
        {"line", line + 1}, {"column", column + 1}, {"column_encoding", "utf-16"}};
    if (input.contains("name") && input["name"].is_string()) output["name"] = input["name"];
    if (input.contains("kind") && input["kind"].is_number_integer()) output["kind"] = input["kind"];
    error.clear(); return true;
}

bool agent_clangd_project_call(const json & input, const json & selected, bool incoming,
        const std::string & repository_root, json & output, std::string & error) {
    output = nullptr;
    const auto endpoint = incoming ? "from" : "to";
    if (!input.is_object() || !input.contains(endpoint)) { error = "call relation lacks its endpoint"; return false; }
    json from, to;
    if (!agent_clangd_project_location(incoming ? input[endpoint] : selected, repository_root, from, error) ||
            !agent_clangd_project_location(incoming ? selected : input[endpoint], repository_root, to, error)) return false;
    if (!input.contains("fromRanges") || !input["fromRanges"].is_array() || input["fromRanges"].empty() ||
            input["fromRanges"].size() > 64) { error = "call relation lacks bounded call-site ranges"; return false; }
    json sites = json::array();
    for (const auto & range : input["fromRanges"]) {
        uint64_t line = 0, column = 0, end_line = 0, end_column = 0;
        if (!position(range, "start", line, column) || !position(range, "end", end_line, end_column) ||
                end_line < line || (end_line == line && end_column < column)) {
            error = "call relation contains an invalid call-site range"; return false;
        }
        sites.push_back({{"path", from["path"]}, {"line", line + 1}, {"column", column + 1},
            {"end_line", end_line + 1}, {"end_column", end_column + 1}});
    }
    output = {{"from", from}, {"to", to}, {"fromRanges", input["fromRanges"]}, {"call_sites", sites}};
    error.clear(); return true;
}

agent_clangd_json agent_clangd_enclosing_function(const agent_clangd_json & symbols,
        int64_t line, int64_t column, const std::string & uri, const std::string & repository_root) {
    json selected = nullptr;
    auto visit = [&](auto && self, const json & entries, const std::string & container) -> void {
        if (!entries.is_array()) return;
        for (const auto & entry : entries) {
            if (!entry.is_object() || !entry.contains("range")) continue;
            uint64_t first_line, first_column, last_line, last_column;
            if (!position(entry["range"], "start", first_line, first_column) ||
                    !position(entry["range"], "end", last_line, last_column)) continue;
            const auto point = std::make_pair(line, column);
            if (point < std::make_pair(static_cast<int64_t>(first_line), static_cast<int64_t>(first_column)) ||
                    point >= std::make_pair(static_cast<int64_t>(last_line), static_cast<int64_t>(last_column))) continue;
            const auto name = entry.contains("name") && entry["name"].is_string() ? entry["name"].get<std::string>() : std::string{};
            const auto kind = entry.contains("kind") && entry["kind"].is_number_integer() && entry["kind"] >= 1 && entry["kind"] <= 26 ? entry["kind"].get<int>() : 0;
            if ((kind == 6 || kind == 9 || kind == 12) && !name.empty() && name.size() <= 1024 && entry.contains("selectionRange")) {
                std::string error;
                json projected;
                if (agent_clangd_project_location({{"uri", uri}, {"range", entry["selectionRange"]}}, repository_root, projected, error)) {
                    projected["name"] = name; projected["kind"] = kind;
                    if (!container.empty()) projected["container_name"] = container;
                    if (entry.contains("detail") && entry["detail"].is_string() && entry["detail"].get_ref<const std::string &>().size() <= 1024)
                        projected["detail"] = entry["detail"];
                    selected = std::move(projected);
                }
            }
            const auto next = name.empty() ? container : container.empty() ? name : container + "::" + name;
            if (entry.contains("children") && next.size() <= 1024) self(self, entry["children"], next);
        }
    };
    visit(visit, symbols, "");
    return selected;
}
