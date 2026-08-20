#include "agent-resource-catalog.h"

#include <mutex>

bool agent_in_memory_resource_catalog::next_resource_id(
    std::string & out,
    std::string & error) {
    std::lock_guard<std::mutex> lock(mutex_);
    out = "resource-" + std::to_string(next_id_++);
    error.clear();
    return true;
}

bool agent_in_memory_resource_catalog::put_descriptor(
    const agent_resource_descriptor & descriptor,
    std::string & error) {
    std::lock_guard<std::mutex> lock(mutex_);
    resources_[descriptor.uri] = descriptor;
    error.clear();
    return true;
}

bool agent_in_memory_resource_catalog::find_descriptor(
    const std::string & uri,
    agent_resource_descriptor & out,
    std::string & error) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = resources_.find(uri);
    if (it == resources_.end()) {
        error = "resource was not found";
        return false;
    }
    out = it->second;
    error.clear();
    return true;
}

bool agent_in_memory_resource_catalog::list_descriptors(
    std::vector<agent_resource_descriptor> & out,
    std::string & error) const {
    std::lock_guard<std::mutex> lock(mutex_);
    out.clear();
    out.reserve(resources_.size());
    for (const auto & entry : resources_) {
        out.push_back(entry.second);
    }
    error.clear();
    return true;
}
