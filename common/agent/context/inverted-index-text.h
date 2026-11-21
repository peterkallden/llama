#pragma once

#include <cstdint>
#include <cstddef>
#include <map>
#include <string>
#include <nlohmann/json.hpp>

// Reconstruct a bounded text from an inverted word-position index.
template<class Json>
std::string common_agent_inverted_index_text(const Json & index, bool & truncated) {
    truncated = false;
    if (!index.is_object()) return {};
    std::map<size_t, std::string> words;
    for (const auto & entry : index.items()) {
        if (!entry.value().is_array() || entry.key().empty()) continue;
        for (const auto & position : entry.value()) {
            if (!position.is_number_integer() && !position.is_number_unsigned()) continue;
            if (position.is_number_integer() && position.template get<int64_t>() < 0) continue;
            const uint64_t offset = position.is_number_unsigned() ? position.template get<uint64_t>() :
                static_cast<uint64_t>(position.template get<int64_t>());
            if (offset > 65535 || words.size() >= 16384) { truncated = true; continue; }
            words.emplace(static_cast<size_t>(offset), entry.key());
        }
    }
    std::string text;
    for (const auto & word : words) {
        if (text.size() + word.second.size() + (text.empty() ? 0 : 1) > 32768) { truncated = true; break; }
        if (!text.empty()) text.push_back(' ');
        text += word.second;
    }
    return text;
}
