#include "agent/runtime/model-catalog.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <sstream>
#include <unordered_set>

using json = nlohmann::ordered_json;

namespace {

bool bounded(const std::string & value, size_t max = 512) {
    return !value.empty() && value.size() <= max;
}

bool optional_bounded(const std::string & value, size_t max = 512) {
    return value.empty() || value.size() <= max;
}

bool valid_relative_model_path(const std::string & value) {
    if (value.empty()) return false;
    const std::filesystem::path path(value);
    if (path.is_absolute() || path.lexically_normal() != path) return false;
    for (const auto & component : path) {
        if (component == "..") return false;
    }
    return true;
}

bool parse_base(const std::string & id, const json & value,
        common_agent_model_base_spec & base, std::string & error) {
    if (!value.is_object()) { error = "model base must be an object: " + id; return false; }
    base = {};
    base.kind = value.value("kind", "generation");
    base.backend = value.value("backend", "server-context");
    base.path = value.value("path", "");
    base.mmproj = value.value("mmproj", "");
    base.load_policy = value.value("load", value.value("load_policy", "lazy"));
    if (!bounded(id) || (base.kind != "generation" && base.kind != "embedding") ||
            (base.backend != "cli" && base.backend != "server-context") ||
            !valid_relative_model_path(base.path) || !optional_bounded(base.mmproj) ||
            (!base.mmproj.empty() && !valid_relative_model_path(base.mmproj)) ||
            (base.load_policy != "resident" && base.load_policy != "lazy")) {
        error = "model base has invalid identity or load policy: " + id;
        return false;
    }
    return true;
}

bool parse_profile(const std::string & id, const json & value,
        common_agent_model_profile_spec & profile, std::string & error) {
    if (!value.is_object()) { error = "model profile must be an object: " + id; return false; }
    profile = {};
    profile.base_model_id = value.value("base", value.value("base_model_id", ""));
    profile.context_size_tokens = value.value("context_size", value.value("context_size_tokens", size_t{0}));
    profile.n_parallel = value.value("n_parallel", 1);
    profile.n_sequences = value.value("n_sequences", 1);
    profile.load_policy = value.value("load", value.value("load_policy", ""));
    const auto adapters = value.value("adapters", json::array());
    if (!adapters.is_array()) { error = "model profile adapters must be an array: " + id; return false; }
    for (const auto & item : adapters) {
        common_agent_adapter_overlay adapter;
        if (item.is_string()) {
            adapter.adapter_id = item.get<std::string>();
        } else if (item.is_object() && item.contains("adapter_id") && item.at("adapter_id").is_string()) {
            adapter.adapter_id = item.at("adapter_id").get<std::string>();
            adapter.scale = item.value("scale", 1.0);
        } else {
            error = "model profile adapter is invalid: " + id;
            return false;
        }
        profile.adapters.push_back(std::move(adapter));
    }
    const auto sidebands = value.value("sidebands", json::array());
    if (!sidebands.is_array()) { error = "model profile sidebands must be an array: " + id; return false; }
    for (const auto & item : sidebands) {
        common_agent_flydelta_sideband_overlay sideband;
        if (item.is_string()) {
            sideband.sideband_id = item.get<std::string>();
        } else if (item.is_object() &&
                ((item.contains("sideband_id") && item.at("sideband_id").is_string()) ||
                 (item.contains("binding_key") && item.at("binding_key").is_string()))) {
            sideband.sideband_id = item.value("sideband_id", "");
            sideband.binding_key = item.value("binding_key", "");
            sideband.scale = item.value("scale", 1.0);
        } else {
            error = "model profile FlyDelta sideband is invalid: " + id;
            return false;
        }
        profile.sidebands.push_back(std::move(sideband));
    }
    if (!bounded(id) || !bounded(profile.base_model_id) ||
            profile.context_size_tokens == 0 ||
            profile.context_size_tokens > 1024 * 1024 ||
            (!profile.load_policy.empty() && profile.load_policy != "resident" && profile.load_policy != "lazy") ||
            profile.adapters.size() > 8 ||
            profile.n_parallel < 1 || profile.n_parallel > 256 ||
            profile.n_sequences < 1 || profile.n_sequences > 256) {
        error = "model profile has invalid identity or bounds: " + id;
        return false;
    }
    std::unordered_set<std::string> ids;
    for (const auto & adapter : profile.adapters) {
        if (!bounded(adapter.adapter_id) || !std::isfinite(adapter.scale) ||
                adapter.scale <= 0.0 || adapter.scale > 4.0 ||
                !ids.insert(adapter.adapter_id).second) {
            error = "model profile adapter overlay is invalid or repeated: " + id;
            return false;
        }
    }
    if (profile.sidebands.size() > 4) {
        error = "model profile has too many FlyDelta sidebands: " + id;
        return false;
    }
    ids.clear();
    for (const auto & sideband : profile.sidebands) {
        if ((!sideband.sideband_id.empty() && !bounded(sideband.sideband_id)) ||
                (!sideband.binding_key.empty() && !bounded(sideband.binding_key)) ||
                (sideband.sideband_id.empty() && sideband.binding_key.empty()) ||
                !std::isfinite(sideband.scale) ||
                sideband.scale <= 0.0 || sideband.scale > 4.0 ||
                (!sideband.sideband_id.empty() && !ids.insert(sideband.sideband_id).second)) {
            error = "model profile FlyDelta sideband is invalid or repeated: " + id;
            return false;
        }
        if (!sideband.binding_key.empty() && !ids.insert("binding:" + sideband.binding_key).second) {
            error = "model profile FlyDelta binding is invalid or repeated: " + id;
            return false;
        }
    }
    return true;
}

} // namespace

bool common_agent_validate_model_catalog(
        const common_agent_model_catalog & catalog,
        std::string & error) {
    error.clear();
    if (catalog.schema_version != 1) { error = "unsupported model catalog schema"; return false; }
    if (!optional_bounded(catalog.directory) || catalog.max_loaded_generation_models == 0 ||
            catalog.max_loaded_generation_models > 256 || catalog.model_eviction != "lru") {
        error = "model catalog has invalid limits or directory";
        return false;
    }
    if (catalog.bases.empty()) {
        if (!catalog.profiles.empty() || !catalog.embedding_model_id.empty()) {
            error = "model catalog references bases but contains no model bases";
            return false;
        }
        return true;
    }
    if (catalog.directory.empty()) { error = "model catalog directory is required when bases are configured"; return false; }
    size_t generation_count = 0;
    for (const auto & entry : catalog.bases) {
        if (entry.second.kind == "generation") ++generation_count;
        if (entry.second.kind == "embedding" && !entry.second.mmproj.empty()) {
            error = "embedding model base must not declare mmproj: " + entry.first;
            return false;
        }
    }
    if (generation_count == 0 && !catalog.profiles.empty()) {
        error = "model catalog profiles require a generation base";
        return false;
    }
    if (!catalog.embedding_model_id.empty()) {
        const auto embedding = catalog.bases.find(catalog.embedding_model_id);
        if (embedding == catalog.bases.end() || embedding->second.kind != "embedding") {
            error = "model catalog embedding_model does not reference an embedding base";
            return false;
        }
    }
    for (const auto & entry : catalog.profiles) {
        const auto base = catalog.bases.find(entry.second.base_model_id);
        if (base == catalog.bases.end() || base->second.kind != "generation") {
            error = "model profile does not reference a generation base: " + entry.first;
            return false;
        }
    }
    if (!catalog.profiles.empty() && catalog.profiles.find(catalog.default_profile) == catalog.profiles.end()) {
        error = "model catalog default profile is unavailable: " + catalog.default_profile;
        return false;
    }
    return true;
}

std::string common_agent_model_catalog_to_json(
        const common_agent_model_catalog & catalog) {
    json bases = json::object();
    for (const auto & entry : catalog.bases) {
        bases[entry.first] = {
            {"kind", entry.second.kind},
            {"backend", entry.second.backend},
            {"path", entry.second.path},
            {"mmproj", entry.second.mmproj},
            {"load", entry.second.load_policy},
        };
    }
    json profiles = json::object();
    for (const auto & entry : catalog.profiles) {
        json adapters = json::array();
        for (const auto & adapter : entry.second.adapters) {
            adapters.push_back({{"adapter_id", adapter.adapter_id}, {"scale", adapter.scale}});
        }
        json sidebands = json::array();
        for (const auto & sideband : entry.second.sidebands) {
            sidebands.push_back({{"sideband_id", sideband.sideband_id},
                {"binding_key", sideband.binding_key}, {"scale", sideband.scale}});
        }
        profiles[entry.first] = {
            {"base", entry.second.base_model_id},
            {"adapters", adapters},
            {"sidebands", sidebands},
            {"context_size", entry.second.context_size_tokens},
            {"n_parallel", entry.second.n_parallel},
            {"n_sequences", entry.second.n_sequences},
            {"load", entry.second.load_policy},
        };
    }
    return json{
        {"schema_version", catalog.schema_version},
        {"directory", catalog.directory},
        {"bases", bases},
        {"profiles", profiles},
        {"routing", {
            {"default_profile", catalog.default_profile},
            {"embedding_model", catalog.embedding_model_id},
        }},
        {"limits", {
            {"max_loaded_generation_models", catalog.max_loaded_generation_models},
            {"model_eviction", catalog.model_eviction},
        }},
    }.dump();
}

bool common_agent_model_catalog_from_json(
        const std::string & text,
        common_agent_model_catalog & catalog,
        std::string & error) {
    error.clear();
    try {
        const auto value = json::parse(text);
        if (!value.is_object()) { error = "model catalog must be an object"; return false; }
        catalog = {};
        catalog.schema_version = value.value("schema_version", 0);
        catalog.directory = value.value("directory", "");
        const auto bases = value.value("bases", json::object());
        const auto profiles = value.value("profiles", json::object());
        const auto routing = value.value("routing", json::object());
        const auto limits = value.value("limits", json::object());
        if (!bases.is_object() || !profiles.is_object() || !routing.is_object() || !limits.is_object()) {
            error = "model catalog sections are invalid";
            return false;
        }
        catalog.default_profile = routing.value("default_profile", "agent-default");
        catalog.embedding_model_id = routing.value("embedding_model", "");
        catalog.max_loaded_generation_models = limits.value("max_loaded_generation_models", size_t{1});
        catalog.model_eviction = limits.value("model_eviction", "lru");
        for (const auto & entry : bases.items()) {
            common_agent_model_base_spec base;
            if (!parse_base(entry.key(), entry.value(), base, error)) return false;
            catalog.bases.emplace(entry.key(), std::move(base));
        }
        for (const auto & entry : profiles.items()) {
            common_agent_model_profile_spec profile;
            if (!parse_profile(entry.key(), entry.value(), profile, error)) return false;
            catalog.profiles.emplace(entry.key(), std::move(profile));
        }
        return common_agent_validate_model_catalog(catalog, error);
    } catch (const std::exception & exception) {
        error = std::string("invalid model catalog JSON: ") + exception.what();
        return false;
    }
}

bool common_agent_model_catalog_make_profile(
        const common_agent_model_catalog & catalog,
        const std::string & profile_id,
        const std::string & base_model_fingerprint,
        const std::string & tokenizer_fingerprint,
        const std::string & chat_template_fingerprint,
        common_agent_model_profile & profile,
        std::string & error) {
    if (!common_agent_validate_model_catalog(catalog, error)) return false;
    const auto selected_id = profile_id.empty() ? catalog.default_profile : profile_id;
    const auto selected = catalog.profiles.find(selected_id);
    if (selected == catalog.profiles.end()) { error = "model profile is unavailable: " + selected_id; return false; }
    const auto base = catalog.bases.find(selected->second.base_model_id);
    if (base == catalog.bases.end()) { error = "model profile base is unavailable"; return false; }
    profile = {};
    profile.id = selected_id;
    profile.base_model_id = selected->second.base_model_id;
    profile.base_model_fingerprint = base_model_fingerprint;
    profile.tokenizer_fingerprint = tokenizer_fingerprint;
    profile.chat_template_fingerprint = chat_template_fingerprint;
    profile.context_size_tokens = selected->second.context_size_tokens;
    profile.n_parallel = selected->second.n_parallel;
    profile.n_sequences = selected->second.n_sequences;
    profile.load_policy = selected->second.load_policy.empty()
        ? base->second.load_policy : selected->second.load_policy;
    profile.adapters = selected->second.adapters;
    profile.sidebands = selected->second.sidebands;
    if (!common_agent_validate_model_profile(profile, error)) return false;
    return true;
}

bool common_agent_model_catalog_resolve_profile(
        const common_agent_model_catalog & catalog,
        const std::string & profile_id,
        common_agent_model_selection & selection,
        std::string & error) {
    if (!common_agent_validate_model_catalog(catalog, error)) return false;
    const auto selected_id = profile_id.empty() ? catalog.default_profile : profile_id;
    const auto selected = catalog.profiles.find(selected_id);
    if (selected == catalog.profiles.end()) {
        error = "model profile is unavailable: " + selected_id;
        return false;
    }
    const auto base = catalog.bases.find(selected->second.base_model_id);
    if (base == catalog.bases.end() || base->second.kind != "generation") {
        error = "model profile generation base is unavailable: " + selected->second.base_model_id;
        return false;
    }
    selection = {};
    selection.profile_id = selected_id;
    selection.base_model_id = base->first;
    selection.backend = base->second.backend;
    selection.path = (std::filesystem::path(catalog.directory) / base->second.path).lexically_normal().string();
    if (!base->second.mmproj.empty()) {
        selection.mmproj = (std::filesystem::path(catalog.directory) / base->second.mmproj).lexically_normal().string();
    }
    selection.context_size_tokens = selected->second.context_size_tokens;
    selection.n_parallel = selected->second.n_parallel;
    selection.n_sequences = selected->second.n_sequences;
    selection.load_policy = selected->second.load_policy.empty()
        ? base->second.load_policy : selected->second.load_policy;
    selection.adapters = selected->second.adapters;
    selection.sidebands = selected->second.sidebands;
    error.clear();
    return true;
}

std::string common_agent_model_selection_cache_key(
        const common_agent_model_selection & selection) {
    std::ostringstream key;
    key << selection.profile_id << '\n'
        << selection.base_model_id << '\n'
        << selection.backend << '\n'
        << selection.path << '\n'
        << selection.mmproj << '\n'
        << selection.context_size_tokens << '\n'
        << selection.n_parallel << '\n'
        << selection.n_sequences << '\n'
        << selection.load_policy << '\n';
    for (const auto & adapter : selection.adapters) {
        key << adapter.adapter_id << ':' << adapter.scale << '\n';
    }
    for (const auto & sideband : selection.sidebands) {
        key << "flydelta:" << sideband.sideband_id << ':' << sideband.scale << '\n';
    }
    return key.str();
}

common_agent_model_router::common_agent_model_router(
        const common_agent_model_catalog & value)
    : catalog(value), profile_cache(value.max_loaded_generation_models) {}

bool common_agent_model_router::begin_turn(
        const std::string & profile_id,
        common_agent_model_route & route,
        std::string & error) {
    route = {};
    if (!common_agent_model_catalog_resolve_profile(catalog, profile_id, route.selection, error)) {
        return false;
    }
    route.cache_key = common_agent_model_selection_cache_key(route.selection);
    route.cache_reused = std::any_of(profile_cache.list().begin(), profile_cache.list().end(),
        [&](const auto & entry) { return entry.key == route.cache_key; });
    if (!profile_cache.begin_turn(
            route.cache_key,
            route.selection.profile_id,
            route.selection.load_policy,
            route.evicted_cache_key,
            error)) {
        route = {};
        return false;
    }
    error.clear();
    return true;
}

bool common_agent_model_router::end_turn(
        const common_agent_model_route & route,
        std::string & error) {
    if (route.cache_key.empty()) {
        error = "model route has no cache key";
        return false;
    }
    return profile_cache.end_turn(route.cache_key, error);
}
