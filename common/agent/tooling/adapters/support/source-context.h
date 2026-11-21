#pragma once

#include <nlohmann/json.hpp>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

inline std::string common_source_preview(const std::string & text) {
    size_t end = std::min<size_t>(text.size(), 256);
    while (end < text.size() && end > 0 && (static_cast<unsigned char>(text[end]) & 0xc0) == 0x80) --end;
    return text.substr(0, end);
}

// The caller must authorize and resolve the file inside its repository root.
inline bool common_source_read_text(const std::filesystem::path & path, std::string & text) {
    std::ifstream input(path, std::ios::binary);
    if (!input) return false;
    text.clear();
    char buffer[4096];
    while (input && text.size() <= 512 * 1024) {
        input.read(buffer, std::min<size_t>(sizeof(buffer), 512 * 1024 + 1 - text.size()));
        text.append(buffer, static_cast<size_t>(input.gcount()));
    }
    if (input.bad() || text.size() > 512 * 1024 || text.find('\0') != std::string::npos) return false;
    return true;
}

inline bool common_source_read_lines(const std::filesystem::path & path, std::vector<std::string> & lines) {
    std::string text;
    if (!common_source_read_text(path, text)) return false;
    std::istringstream stream(text);
    std::string line;
    while (std::getline(stream, line)) lines.push_back(std::move(line));
    return true;
}

inline nlohmann::ordered_json common_source_context(const std::vector<std::string> & lines, size_t line, size_t radius) {
    if (!line || line > lines.size()) return nullptr;
    nlohmann::ordered_json result = {{"start_line", line > radius ? line - radius : 1}, {"lines", nlohmann::ordered_json::array()}};
    const auto first = line > radius ? line - radius : 1;
    const auto last = std::min(lines.size(), line + radius);
    bool truncated = false;
    for (size_t number = first; number <= last; ++number) {
        const auto text = common_source_preview(lines[number - 1]);
        const bool clipped = text.size() != lines[number - 1].size();
        result["lines"].push_back({{"line", number}, {"text", text}, {"truncated", clipped}});
        truncated = truncated || clipped;
    }
    result["end_line"] = last; result["truncated"] = truncated;
    return result;
}

class common_source_context_cache {
public:
    nlohmann::ordered_json get(const std::filesystem::path & path, size_t line) {
        auto found = files.find(path);
        if (found == files.end()) {
            if (files.size() >= 16) return nullptr;
            std::vector<std::string> lines;
            if (common_source_read_lines(path, lines)) {
                size_t bytes = 0;
                for (const auto & text : lines) bytes += text.size() + 1;
                if (bytes <= 4 * 1024 * 1024 - retained_bytes) retained_bytes += bytes;
                else lines.clear();
            }
            found = files.emplace(path, std::move(lines)).first;
        }
        return common_source_context(found->second, line, 2);
    }
private:
    size_t retained_bytes = 0;
    std::map<std::filesystem::path, std::vector<std::string>> files;
};

inline void common_source_limit_context(nlohmann::ordered_json & entries) {
    for (size_t i = entries.size(); i > 0 && entries.dump().size() > 48 * 1024; --i) {
        auto & entry = entries[i - 1];
        if (!entry.contains("context")) continue;
        entry["context"] = nullptr;
        entry["context_status"] = "omitted_output_budget";
        entry["enclosing_function"] = nullptr;
        entry["enclosing_function_status"] = "omitted_output_budget";
    }
}
