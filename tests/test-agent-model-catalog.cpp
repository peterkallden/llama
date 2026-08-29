#include "agent/runtime/model-catalog.h"

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
        selection.base_model_id == "small" && selection.path == "/models/qwen.gguf" &&
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

} // namespace

int main() {
    return test_parse_and_resolve() && test_rejects_invalid_catalogs() ? 0 : 1;
}
