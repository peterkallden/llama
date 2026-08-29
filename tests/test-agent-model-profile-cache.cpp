#include "agent/runtime/model-catalog.h"
#include "agent/runtime/model-profile-cache.h"

#include <string>

namespace {

bool test_cache_key_is_profile_specific() {
    common_agent_model_selection first;
    first.profile_id = "agent-default";
    first.base_model_id = "small";
    first.backend = "server-context";
    first.path = "/models/small.gguf";
    first.context_size_tokens = 4096;
    first.load_policy = "resident";
    first.adapters.push_back({"agent-correction-v1", 1.0});
    auto second = first;
    second.adapters.front().scale = 0.75;
    return common_agent_model_selection_cache_key(first) !=
        common_agent_model_selection_cache_key(second);
}

bool test_pinning_and_lru_eviction() {
    common_agent_model_profile_cache cache(2);
    std::string evicted;
    std::string error;
    if (!cache.begin_turn("a", "small", "resident", evicted, error) ||
            !cache.begin_turn("b", "research", "lazy", evicted, error)) return false;
    if (cache.begin_turn("c", "coding", "lazy", evicted, error) ||
            error.find("pinned") == std::string::npos) return false;
    if (!cache.end_turn("a", error)) return false;
    if (!cache.begin_turn("c", "coding", "lazy", evicted, error) || evicted != "a") return false;
    if (!cache.end_turn("b", error) || !cache.end_turn("c", error)) return false;
    return cache.size() == 2;
}

bool test_reentrant_turns_and_invalid_release() {
    common_agent_model_profile_cache cache(1);
    std::string evicted;
    std::string error;
    if (!cache.begin_turn("a", "small", "resident", evicted, error) ||
            !cache.begin_turn("a", "small", "resident", evicted, error) ||
            cache.list().front().active_turns != 2) return false;
    if (!cache.end_turn("a", error) || !cache.end_turn("a", error)) return false;
    if (cache.end_turn("a", error) || error.find("not active") == std::string::npos) return false;
    if (cache.begin_turn("b", "small", "invalid", evicted, error)) return false;
    return error.find("load policy") != std::string::npos;
}

} // namespace

int main() {
    return test_cache_key_is_profile_specific() &&
        test_pinning_and_lru_eviction() &&
        test_reentrant_turns_and_invalid_release() ? 0 : 1;
}
