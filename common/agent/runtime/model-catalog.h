#pragma once

#include "agent/runtime/model-profile.h"

#include <cstddef>
#include <map>
#include <string>
#include <vector>

// Configuration-level model kind. Embedding models are catalogued here for
// discovery, but are never valid generation bases for an agent profile.
struct common_agent_model_base_spec {
    std::string kind = "generation";
    std::string backend = "server-context";
    std::string path;
    std::string mmproj;
    std::string load_policy = "lazy";
};

struct common_agent_model_profile_spec {
    std::string base_model_id;
    std::vector<common_agent_adapter_overlay> adapters;
    size_t context_size_tokens = 0;
    // Empty means inherit the base model's load policy.
    std::string load_policy;
};

struct common_agent_model_catalog {
    int schema_version = 1;
    std::string directory;
    std::map<std::string, common_agent_model_base_spec> bases;
    std::map<std::string, common_agent_model_profile_spec> profiles;
    std::string default_profile = "agent-default";
    std::string embedding_model_id;
    size_t max_loaded_generation_models = 1;
    std::string model_eviction = "lru";
};

// Host-facing result of selecting one named generation profile. This resolves
// catalog-relative paths but does not load a model or apply adapter weights.
struct common_agent_model_selection {
    std::string profile_id;
    std::string base_model_id;
    std::string backend;
    std::string path;
    std::string mmproj;
    size_t context_size_tokens = 0;
    std::string load_policy;
    std::vector<common_agent_adapter_overlay> adapters;
};

bool common_agent_validate_model_catalog(
        const common_agent_model_catalog & catalog,
        std::string & error);

std::string common_agent_model_catalog_to_json(
        const common_agent_model_catalog & catalog);

bool common_agent_model_catalog_from_json(
        const std::string & text,
        common_agent_model_catalog & catalog,
        std::string & error);

// Resolve a configured profile into the host-neutral identity contract. The
// loader supplies fingerprints later; configuration cannot invent them.
bool common_agent_model_catalog_make_profile(
        const common_agent_model_catalog & catalog,
        const std::string & profile_id,
        const std::string & base_model_fingerprint,
        const std::string & tokenizer_fingerprint,
        const std::string & chat_template_fingerprint,
        common_agent_model_profile & profile,
        std::string & error);

bool common_agent_model_catalog_resolve_profile(
        const common_agent_model_catalog & catalog,
        const std::string & profile_id,
        common_agent_model_selection & selection,
        std::string & error);

// The key is safe to use for residency/KV admission before a model is loaded.
// It includes every selection field that can change serving behavior.
std::string common_agent_model_selection_cache_key(
        const common_agent_model_selection & selection);
