#include "agent/runtime/model-catalog.h"

#include <filesystem>
#include <string>

namespace {

const char * kCatalog = R"json({
  "schema_version": 1,
  "directory": "/models",
  "bases": {
    "small": {
      "kind": "generation",
      "backend": "server-context",
      "path": "qwen.gguf",
      "load": "resident"
    },
    "nomic": {
      "kind": "embedding",
      "backend": "server-context",
      "path": "nomic.gguf",
      "load": "resident"
    }
  },
  "profiles": {
    "agent-default": {
      "base": "small",
      "adapters": [
        {"adapter_id": "agent-correction-v1", "scale": 0.75}
      ],
      "context_size": 4096,
      "load": "resident"
    }
  },
  "routing": {
    "default_profile": "agent-default",
    "embedding_model": "nomic"
  },
  "limits": {
    "max_loaded_generation_models": 2,
    "model_eviction": "lru"
  }
})json";

bool test_parse_and_resolve() {
    common_agent_model_catalog catalog;
    std::string error;
    if (!common_agent_model_catalog_from_json(kCatalog, catalog, error)) return false;
    if (!error.empty() || catalog.directory != "/models" ||
            catalog.bases.at("nomic").kind != "embedding" ||
            catalog.profiles.at("agent-default").context_size_tokens != 4096) return false;

    const auto roundtrip = common_agent_model_catalog_to_json(catalog);
    common_agent_model_catalog decoded;
    if (!common_agent_model_catalog_from_json(roundtrip, decoded, error)) return false;

    common_agent_model_profile profile;
    if (!common_agent_model_catalog_make_profile(
        catalog, {}, "sha256:base", "sha256:tokenizer", "sha256:template",
        profile, error)) return false;
    if (!(profile.id == "agent-default" && profile.base_model_id == "small" &&
        profile.context_size_tokens == 4096 && profile.adapters.size() == 1 &&
        profile.load_policy == "resident")) return false;

    common_agent_model_selection selection;
    if (!common_agent_model_catalog_resolve_profile(catalog, {}, selection, error)) return false;
    return selection.profile_id == "agent-default" &&
        selection.base_model_id == "small" &&
        selection.path == (std::filesystem::path("/models") / "qwen.gguf").lexically_normal().string() &&
        selection.mmproj.empty() && selection.context_size_tokens == 4096 &&
        selection.adapters.size() == 1;
}

bool test_rejects_invalid_catalogs() {
    std::string error;
    common_agent_model_catalog catalog;

    if (common_agent_model_catalog_from_json(R"json({
      "schema_version": 1,
      "directory": "/models",
      "bases": {"bad": {"path": "/etc/passwd"}},
      "profiles": {}, "routing": {}, "limits": {}
    })json", catalog, error)) return false;

    if (common_agent_model_catalog_from_json(R"json({
      "schema_version": 1,
      "directory": "/models",
      "bases": {"embedding": {"kind": "embedding", "path": "nomic.gguf"}},
      "profiles": {"agent-default": {"base": "embedding", "context_size": 4096}},
      "routing": {"default_profile": "agent-default"}, "limits": {}
    })json", catalog, error)) return false;

    if (common_agent_model_catalog_from_json(R"json({
      "schema_version": 1,
      "directory": "/models",
      "bases": {"small": {"path": "qwen.gguf"}},
      "profiles": {"agent-default": {"base": "small"}},
      "routing": {"default_profile": "agent-default"}, "limits": {}
    })json", catalog, error)) return false;

    if (common_agent_model_catalog_from_json(R"json({
      "schema_version": 1,
      "directory": "/models",
      "bases": {"small": {"path": "../qwen.gguf"}},
      "profiles": {}, "routing": {}, "limits": {}
    })json", catalog, error)) return false;
    return true;
}

bool test_profile_router() {
    common_agent_model_catalog catalog;
    std::string error;
    if (!common_agent_model_catalog_from_json(R"json({
      "schema_version": 1,
      "directory": "/models",
      "bases": {
        "small": {"kind": "generation", "backend": "cli", "path": "qwen.gguf"},
        "research": {"kind": "generation", "backend": "cli", "path": "research.gguf"}
      },
      "profiles": {
        "agent-default": {"base": "small", "context_size": 4096},
        "research": {"base": "research", "context_size": 8192}
      },
      "routing": {"default_profile": "agent-default"},
      "limits": {"max_loaded_generation_models": 2, "model_eviction": "lru"}
    })json", catalog, error)) return false;

    common_agent_model_router router(catalog);
    common_agent_model_route baseline;
    if (!router.begin_turn({}, baseline, error) || baseline.cache_reused ||
            baseline.selection.path != (std::filesystem::path("/models") / "qwen.gguf").lexically_normal().string()) return false;
    common_agent_model_route same;
    if (!router.begin_turn("agent-default", same, error) || !same.cache_reused ||
            same.cache_key != baseline.cache_key) return false;
    common_agent_model_route research;
    if (!router.begin_turn("research", research, error) || research.cache_reused ||
            research.selection.path != (std::filesystem::path("/models") / "research.gguf").lexically_normal().string() ||
            router.cache().size() != 2) return false;
    if (!router.end_turn(same, error) || !router.end_turn(baseline, error) ||
            !router.end_turn(research, error)) return false;
    if (router.cache().list().size() != 2 || !error.empty()) return false;

    common_agent_model_catalog limited_catalog = catalog;
    limited_catalog.max_loaded_generation_models = 1;
    common_agent_model_router limited_router(limited_catalog);
    common_agent_model_route limited_baseline;
    if (!limited_router.begin_turn("agent-default", limited_baseline, error)) return false;
    if (!limited_router.end_turn(limited_baseline, error)) return false;
    common_agent_model_route limited_research;
    if (!limited_router.begin_turn("research", limited_research, error) ||
            limited_research.evicted_cache_key != limited_baseline.cache_key ||
            limited_router.cache().list().size() != 1) return false;
    return limited_router.end_turn(limited_research, error) && error.empty();
}

} // namespace

int main() {
    return test_parse_and_resolve() && test_rejects_invalid_catalogs() &&
        test_profile_router() ? 0 : 1;
}
