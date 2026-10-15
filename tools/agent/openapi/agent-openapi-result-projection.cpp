#include "agent-openapi-result-projection.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <nlohmann/json.hpp>

namespace {
using json = nlohmann::ordered_json;

bool add_model_views(json & value, bool openalex_result) {
    bool changed = false;
    if (value.is_array()) {
        for (auto & item : value) changed = add_model_views(item, openalex_result) || changed;
        return changed;
    }
    if (!value.is_object()) return false;

    const auto abstract = value.find("abstract_inverted_index");
    if (abstract != value.end() && abstract->is_object() && !value.contains("abstract_text")) {
        std::map<size_t, std::string> words;
        for (const auto & entry : abstract->items()) {
            if (!entry.value().is_array() || entry.key().empty()) continue;
            for (const auto & position : entry.value()) {
                if (!position.is_number_integer() && !position.is_number_unsigned()) continue;
                if (position.is_number_integer() && position.get<int64_t>() < 0) continue;
                const uint64_t index = position.is_number_unsigned()
                    ? position.get<uint64_t>() : static_cast<uint64_t>(position.get<int64_t>());
                if (index <= 65535 && words.size() < 16384) {
                    words.emplace(static_cast<size_t>(index), entry.key());
                }
            }
        }
        std::string text;
        for (const auto & word : words) {
            if (!text.empty()) text.push_back(' ');
            if (text.size() + word.second.size() > 32768) break;
            text += word.second;
        }
        if (!text.empty()) {
            value["abstract_text"] = std::move(text);
            changed = true;
        }
    }

    // OpenAlex may expose an open-access file in either location. This is a
    // host-derived model view: prefer the PDF and never substitute a landing
    // page or accept a URL supplied by the model.
    if (openalex_result && !value.contains("full_text_url")) {
        std::string url;
        std::string source;
        if (value.contains("best_oa_location") && value["best_oa_location"].is_object() &&
                value["best_oa_location"].contains("pdf_url") &&
                value["best_oa_location"]["pdf_url"].is_string() &&
                !value["best_oa_location"]["pdf_url"].get<std::string>().empty()) {
            url = value["best_oa_location"]["pdf_url"].get<std::string>();
            source = "best_oa_location.pdf_url";
        } else if (value.contains("content_url") && value["content_url"].is_string()) {
            url = value["content_url"].get<std::string>();
            source = "content_url";
        }
        if (!url.empty()) {
            value["full_text_url"] = std::move(url);
            value["full_text_url_source"] = std::move(source);
            changed = true;
        }
    }

    for (auto & entry : value.items()) {
        if (entry.key() != "abstract_inverted_index") {
            changed = add_model_views(entry.value(), openalex_result) || changed;
        }
    }
    return changed;
}
} // namespace

std::string agent_openapi_project_model_result(
        const std::string & raw_json, bool openalex_result) {
    auto value = json::parse(raw_json, nullptr, false);
    if (value.is_discarded() || !add_model_views(value, openalex_result)) return raw_json;
    return value.dump();
}
