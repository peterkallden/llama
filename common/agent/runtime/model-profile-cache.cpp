#include "agent/runtime/model-profile-cache.h"

#include <algorithm>

namespace {

bool valid_load_policy(const std::string & value) {
    return value == "resident" || value == "lazy";
}

} // namespace

common_agent_model_profile_cache::common_agent_model_profile_cache(size_t capacity)
    : max_entries(capacity) {}

bool common_agent_model_profile_cache::begin_turn(
        const std::string & key,
        const std::string & profile_id,
        const std::string & load_policy,
        std::string & evicted_key,
        std::string & error) {
    evicted_key.clear();
    error.clear();
    if (max_entries == 0) {
        error = "model profile cache capacity must be greater than zero";
        return false;
    }
    if (key.empty() || profile_id.empty() || !valid_load_policy(load_policy)) {
        error = "model profile cache admission identity or load policy is invalid";
        return false;
    }
    ++clock;
    for (auto & entry : entries) {
        if (entry.key == key) {
            if (entry.profile_id != profile_id || entry.load_policy != load_policy) {
                error = "model profile cache key conflicts with existing identity";
                return false;
            }
            ++entry.active_turns;
            entry.last_used = clock;
            return true;
        }
    }
    if (entries.size() >= max_entries) {
        auto candidate = std::min_element(entries.begin(), entries.end(),
            [](const auto & lhs, const auto & rhs) {
                if (lhs.active_turns != rhs.active_turns) return lhs.active_turns < rhs.active_turns;
                return lhs.last_used < rhs.last_used;
            });
        if (candidate == entries.end() || candidate->active_turns != 0) {
            error = "all model profiles are pinned by active turns";
            return false;
        }
        evicted_key = candidate->key;
        *candidate = {};
        entries.erase(candidate);
    }
    entries.push_back({key, profile_id, load_policy, 1, clock});
    return true;
}

bool common_agent_model_profile_cache::end_turn(
        const std::string & key,
        std::string & error) {
    for (auto & entry : entries) {
        if (entry.key == key) {
            if (entry.active_turns == 0) {
                error = "model profile cache turn is not active";
                return false;
            }
            --entry.active_turns;
            error.clear();
            return true;
        }
    }
    error = "model profile cache key is not loaded";
    return false;
}
