// Host-neutral identity for a runnable generation profile.
//
// A profile names a base model and approved adapter overlays.  It is a
// description used by a host/model loader; it does not load files or permit a
// model to select itself.
#pragma once

#include <cstddef>
#include <string>
#include <vector>

struct common_agent_adapter_overlay {
    std::string adapter_id;
    double scale = 1.0;
};

struct common_agent_model_profile {
    int schema_version = 1;
    std::string id;
    std::string base_model_id;
    std::string base_model_fingerprint;
    std::string tokenizer_fingerprint;
    std::string chat_template_fingerprint;
    size_t context_size_tokens = 0;
    std::string load_policy = "lazy";
    std::vector<common_agent_adapter_overlay> adapters;
};

bool common_agent_validate_model_profile(
        const common_agent_model_profile & profile,
        std::string & error);

// The key is used to prevent reuse of resident/KV state across profiles with
// different model, tokenizer, template, context, or adapter identity.
std::string common_agent_model_profile_cache_key(
        const common_agent_model_profile & profile);

std::string common_agent_model_profile_to_json(
        const common_agent_model_profile & profile);
bool common_agent_model_profile_from_json(
        const std::string & text,
        common_agent_model_profile & profile,
        std::string & error);
