#include "agent/runtime/model-profile.h"

#include <cmath>
#include <nlohmann/json.hpp>
#include <sstream>
#include <unordered_set>

using json = nlohmann::ordered_json;

namespace {

bool bounded(const std::string & value, size_t max = 512) {
    return !value.empty() && value.size() <= max;
}

bool json_string(const json & value, const char * name, std::string & output, std::string & error) {
    if (!value.contains(name) || !value.at(name).is_string()) {
        error = std::string("model profile requires string field: ") + name;
        return false;
    }
    output = value.at(name).get<std::string>();
    return true;
}

} // namespace

bool common_agent_validate_model_profile(
        const common_agent_model_profile & profile,
        std::string & error) {
    error.clear();
    if (profile.schema_version != 1) {
        error = "unsupported model profile schema";
        return false;
    }
    if (!bounded(profile.id) || !bounded(profile.base_model_id) ||
            !bounded(profile.base_model_fingerprint) ||
            !bounded(profile.tokenizer_fingerprint) ||
            !bounded(profile.chat_template_fingerprint)) {
        error = "model profile identity is incomplete";
        return false;
    }
    if (profile.context_size_tokens == 0 || profile.context_size_tokens > 1024 * 1024) {
        error = "model profile context size is outside bounds";
        return false;
    }
    if (profile.load_policy != "resident" && profile.load_policy != "lazy") {
        error = "model profile load policy is invalid";
        return false;
    }
    if (profile.adapters.size() > 8) {
        error = "model profile has too many adapter overlays";
        return false;
    }
    std::unordered_set<std::string> adapter_ids;
    for (const auto & adapter : profile.adapters) {
        if (!bounded(adapter.adapter_id) || !std::isfinite(adapter.scale) ||
                adapter.scale <= 0.0 || adapter.scale > 4.0) {
            error = "model profile adapter overlay is invalid";
            return false;
        }
        if (!adapter_ids.insert(adapter.adapter_id).second) {
            error = "model profile repeats an adapter overlay";
            return false;
        }
    }
    return true;
}

std::string common_agent_model_profile_cache_key(
        const common_agent_model_profile & profile) {
    std::ostringstream key;
    key << profile.base_model_id << '\n'
        << profile.base_model_fingerprint << '\n'
        << profile.tokenizer_fingerprint << '\n'
        << profile.chat_template_fingerprint << '\n'
        << profile.context_size_tokens << '\n';
    for (const auto & adapter : profile.adapters) {
        key << adapter.adapter_id << ':' << adapter.scale << '\n';
    }
    return key.str();
}

std::string common_agent_model_profile_to_json(
        const common_agent_model_profile & profile) {
    json adapters = json::array();
    for (const auto & adapter : profile.adapters) {
        adapters.push_back({{"adapter_id", adapter.adapter_id}, {"scale", adapter.scale}});
    }
    return json{
        {"schema_version", profile.schema_version},
        {"id", profile.id},
        {"base_model_id", profile.base_model_id},
        {"base_model_fingerprint", profile.base_model_fingerprint},
        {"tokenizer_fingerprint", profile.tokenizer_fingerprint},
        {"chat_template_fingerprint", profile.chat_template_fingerprint},
        {"context_size_tokens", profile.context_size_tokens},
        {"load_policy", profile.load_policy},
        {"adapters", adapters},
    }.dump();
}

bool common_agent_model_profile_from_json(
        const std::string & text,
        common_agent_model_profile & profile,
        std::string & error) {
    error.clear();
    try {
        const auto value = json::parse(text);
        if (!value.is_object() || !value.contains("schema_version") ||
                !value.at("schema_version").is_number_integer() ||
                !value.contains("context_size_tokens") ||
                !value.at("context_size_tokens").is_number_unsigned() ||
                !value.contains("adapters") || !value.at("adapters").is_array()) {
            error = "model profile JSON has invalid structural fields";
            return false;
        }
        profile = {};
        profile.schema_version = value.at("schema_version").get<int>();
        profile.context_size_tokens = value.at("context_size_tokens").get<size_t>();
        if (!json_string(value, "id", profile.id, error) ||
                !json_string(value, "base_model_id", profile.base_model_id, error) ||
                !json_string(value, "base_model_fingerprint", profile.base_model_fingerprint, error) ||
                !json_string(value, "tokenizer_fingerprint", profile.tokenizer_fingerprint, error) ||
                !json_string(value, "chat_template_fingerprint", profile.chat_template_fingerprint, error) ||
                !json_string(value, "load_policy", profile.load_policy, error)) {
            return false;
        }
        for (const auto & item : value.at("adapters")) {
            if (!item.is_object() || !item.contains("adapter_id") ||
                    !item.at("adapter_id").is_string() || !item.contains("scale") ||
                    !item.at("scale").is_number()) {
                error = "model profile adapter JSON is invalid";
                return false;
            }
            common_agent_adapter_overlay adapter;
            adapter.adapter_id = item.at("adapter_id").get<std::string>();
            adapter.scale = item.at("scale").get<double>();
            profile.adapters.push_back(std::move(adapter));
        }
        return common_agent_validate_model_profile(profile, error);
    } catch (const std::exception & exception) {
        error = std::string("invalid model profile JSON: ") + exception.what();
        return false;
    }
}
