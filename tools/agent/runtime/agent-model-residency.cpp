#include "agent-model-residency.h"

#include <algorithm>

namespace {

bool find_cache_entry(
        const common_agent_model_profile_cache & cache,
        const std::string & key,
        common_agent_model_cache_entry & result) {
    const auto it = std::find_if(cache.list().begin(), cache.list().end(),
        [&](const auto & entry) { return entry.key == key; });
    if (it == cache.list().end()) return false;
    result = *it;
    return true;
}

} // namespace

common_agent_runtime_model_residency::common_agent_runtime_model_residency(
        common_agent_model_catalog catalog,
        std::unordered_map<std::string,
            std::shared_ptr<common_agent_runtime_model_loader>> loaders)
    : catalog_(std::move(catalog)),
      loaders_(std::move(loaders)),
      cache_(catalog_.max_loaded_generation_models) {}

bool common_agent_runtime_model_residency::acquire(
        const std::string & profile_id,
        common_agent_runtime_model_resident_handle & handle,
        std::string & error) {
    handle = {};
    common_agent_model_selection selection;
    if (!common_agent_model_catalog_resolve_profile(
            catalog_, profile_id, selection, error)) {
        return false;
    }
    const std::string cache_key = common_agent_model_selection_cache_key(selection);

    std::shared_ptr<common_agent_runtime_model_loader> loader;
    std::shared_ptr<common_agent_runtime_resident_model> evicted_model;
    {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [&]() { return loading_keys_.count(cache_key) == 0; });

        const auto existing = entries_.find(cache_key);
        if (existing != entries_.end() && existing->second.model != nullptr) {
            std::string cache_error;
            std::string evicted_key;
            if (!cache_.begin_turn(cache_key, selection.profile_id,
                    selection.load_policy, evicted_key, cache_error)) {
                error = cache_error;
                return false;
            }
            if (!find_cache_entry(cache_, cache_key, existing->second.cache)) {
                error = "model residency cache entry disappeared";
                return false;
            }
            handle = {selection, selection.profile_id, cache_key, existing->second.model};
            error.clear();
            return true;
        }

        loader = loaders_.find(selection.backend) == loaders_.end()
            ? nullptr : loaders_.at(selection.backend);
        if (!loader) {
            error = "no model loader is registered for backend: " + selection.backend;
            return false;
        }

        std::string evicted_key;
        if (!cache_.begin_turn(cache_key, selection.profile_id,
                selection.load_policy, evicted_key, error)) {
            return false;
        }
        if (!evicted_key.empty()) {
            const auto evicted = entries_.find(evicted_key);
            if (evicted != entries_.end()) {
                evicted_model = std::move(evicted->second.model);
                entries_.erase(evicted);
            }
        }
        entries_[cache_key] = {cache_.list().back(), {}, true};
        loading_keys_.insert(cache_key);
    }

    // Model I/O happens outside the manager mutex.  Other profiles can still
    // be released or inspected while a large model is loading.
    evicted_model.reset();
    std::shared_ptr<common_agent_runtime_resident_model> loaded_model;
    const bool loaded = loader->load(selection, loaded_model, error);

    {
        std::lock_guard<std::mutex> lock(mutex_);
        loading_keys_.erase(cache_key);
        const auto current = entries_.find(cache_key);
        if (!loaded || loaded_model == nullptr) {
            if (current != entries_.end()) entries_.erase(current);
            std::string ignored;
            cache_.abort_turn(cache_key, ignored);
            if (loaded && loaded_model == nullptr) {
                error = "model loader returned an empty resident model";
            }
            condition_.notify_all();
            return false;
        }
        if (current == entries_.end()) {
            error = "model residency reservation disappeared during load";
            condition_.notify_all();
            return false;
        }
        current->second.model = std::move(loaded_model);
        current->second.loading = false;
        handle = {selection, selection.profile_id, cache_key, current->second.model};
    }
    condition_.notify_all();
    error.clear();
    return true;
}

bool common_agent_runtime_model_residency::release(
        const common_agent_runtime_model_resident_handle & handle,
        std::string & error) {
    if (!handle.valid()) {
        error = "model residency handle is invalid";
        return false;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = entries_.find(handle.cache_key);
    if (it == entries_.end() || it->second.model != handle.model) {
        error = "model residency handle is not active";
        return false;
    }
    if (!cache_.end_turn(handle.cache_key, error)) return false;
    if (!find_cache_entry(cache_, handle.cache_key, it->second.cache)) {
        error = "model residency cache entry disappeared during release";
        return false;
    }
    error.clear();
    return true;
}

std::vector<common_agent_runtime_model_residency_entry>
common_agent_runtime_model_residency::list() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<common_agent_runtime_model_residency_entry> result;
    result.reserve(entries_.size());
    for (const auto & item : entries_) {
        result.push_back({item.second.cache, item.second.loading});
    }
    std::sort(result.begin(), result.end(), [](const auto & lhs, const auto & rhs) {
        return lhs.cache.key < rhs.cache.key;
    });
    return result;
}
