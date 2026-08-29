#pragma once

#include "agent/runtime/model-catalog.h"
#include "agent/runtime/model-profile-cache.h"

#include <condition_variable>
#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// The backend loader owns the actual llama.cpp/server-context resources.  The
// residency manager owns the lifetime of the returned shared state and never
// needs to know which backend-specific object it contains.
struct common_agent_runtime_resident_model {
    virtual ~common_agent_runtime_resident_model() = default;
};

class common_agent_runtime_model_loader {
public:
    virtual ~common_agent_runtime_model_loader() = default;

    virtual bool load(
            const common_agent_model_selection & selection,
            std::shared_ptr<common_agent_runtime_resident_model> & model,
            std::string & error) = 0;
};

struct common_agent_runtime_model_resident_handle {
    common_agent_model_selection selection;
    std::string profile_id;
    std::string cache_key;
    std::shared_ptr<common_agent_runtime_resident_model> model;

    bool valid() const { return !cache_key.empty() && model != nullptr; }
};

struct common_agent_runtime_model_residency_entry {
    common_agent_model_cache_entry cache;
    bool loading = false;
};

// Process-wide profile residency.  It coordinates admission and eviction but
// deliberately leaves session contexts/KV state outside the manager.
class common_agent_runtime_model_residency {
public:
    common_agent_runtime_model_residency(
            common_agent_model_catalog catalog,
            std::unordered_map<std::string,
                std::shared_ptr<common_agent_runtime_model_loader>> loaders);

    bool acquire(
            const std::string & profile_id,
            common_agent_runtime_model_resident_handle & handle,
            std::string & error);

    bool release(
            const common_agent_runtime_model_resident_handle & handle,
            std::string & error);

    std::vector<common_agent_runtime_model_residency_entry> list() const;
    const common_agent_model_catalog & catalog() const { return catalog_; }

private:
    struct resident_entry {
        common_agent_model_cache_entry cache;
        std::shared_ptr<common_agent_runtime_resident_model> model;
        bool loading = false;
    };

    common_agent_model_catalog catalog_;
    std::unordered_map<std::string,
        std::shared_ptr<common_agent_runtime_model_loader>> loaders_;
    common_agent_model_profile_cache cache_;
    mutable std::mutex mutex_;
    std::condition_variable condition_;
    std::unordered_set<std::string> loading_keys_;
    std::unordered_map<std::string, resident_entry> entries_;
};
