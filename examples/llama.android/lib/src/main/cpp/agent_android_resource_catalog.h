#pragma once

#include "tools/agent/resource/agent-resource-catalog.h"
#include "agent/storage/sqlite/sqlite-database.h"

// Persistent metadata for the existing resource store and its filesystem blobs.
class android_resource_catalog final : public agent_resource_catalog {
public:
    bool open(const std::string & path, std::string & error);
    bool next_resource_id(std::string & out, std::string & error) override;
    bool put_descriptor(const agent_resource_descriptor & value, std::string & error) override;
    bool find_descriptor(const std::string & uri, agent_resource_descriptor & out, std::string & error) const override;
    bool list_descriptors(std::vector<agent_resource_descriptor> & out, std::string & error) const override;
private:
    mutable std::mutex mutex_;
    common_sqlite_database db_;
};
