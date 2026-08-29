#include "agent/runtime/model-profile.h"

#include <cassert>

static common_agent_model_profile profile() {
    common_agent_model_profile value;
    value.id = "agent-candidate";
    value.base_model_id = "qwen-small";
    value.base_model_fingerprint = "base:qwen-small:v1";
    value.tokenizer_fingerprint = "tokenizer:v1";
    value.chat_template_fingerprint = "chat:v1";
    value.context_size_tokens = 4096;
    value.load_policy = "resident";
    value.adapters.push_back({"adapter-v1", 0.75});
    return value;
}

int main() {
    std::string error;
    auto value = profile();
    assert(common_agent_validate_model_profile(value, error));
    const auto key = common_agent_model_profile_cache_key(value);
    const auto text = common_agent_model_profile_to_json(value);
    common_agent_model_profile parsed;
    assert(common_agent_model_profile_from_json(text, parsed, error));
    assert(common_agent_model_profile_cache_key(parsed) == key);

    value.adapters.push_back({"adapter-v1", 1.0});
    assert(!common_agent_validate_model_profile(value, error));
    assert(error.find("repeats") != std::string::npos);
    value = profile();
    value.context_size_tokens = 0;
    assert(!common_agent_validate_model_profile(value, error));
    assert(error.find("context") != std::string::npos);
    value = profile();
    value.adapters.front().scale = 5.0;
    assert(!common_agent_validate_model_profile(value, error));
    assert(error.find("overlay") != std::string::npos);
    return 0;
}
