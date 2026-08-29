#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct common_agent_model_cache_entry {
    std::string key;
    std::string profile_id;
    std::string load_policy;
    size_t active_turns = 0;
    uint64_t last_used = 0;
};

// Host-neutral admission control for resident model/KV state. It owns no
// llama_model pointer; the loader uses the eviction result to release state.
class common_agent_model_profile_cache {
public:
    explicit common_agent_model_profile_cache(size_t capacity);

    bool begin_turn(
            const std::string & key,
            const std::string & profile_id,
            const std::string & load_policy,
            std::string & evicted_key,
            std::string & error);
    bool end_turn(const std::string & key, std::string & error);
    bool abort_turn(const std::string & key, std::string & error);

    size_t capacity() const { return max_entries; }
    size_t size() const { return entries.size(); }
    const std::vector<common_agent_model_cache_entry> & list() const { return entries; }

private:
    size_t max_entries;
    uint64_t clock = 0;
    std::vector<common_agent_model_cache_entry> entries;
};
