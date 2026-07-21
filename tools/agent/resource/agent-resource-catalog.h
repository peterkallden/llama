#pragma once

#include "resource/resource-contract.h"

#include <memory>
#include <mutex>
#include <unordered_map>

class agent_resource_catalog {
public:
    virtual ~agent_resource_catalog() = default;

    virtual bool next_resource_id(
        std::string & out,
        std::string & error) = 0;

    virtual bool put_descriptor(
        const agent_resource_descriptor & descriptor,
        std::string & error) = 0;

    virtual bool find_descriptor(
        const std::string & uri,
        agent_resource_descriptor & out,
        std::string & error) const = 0;

    virtual bool list_descriptors(
        std::vector<agent_resource_descriptor> & out,
        std::string & error) const = 0;
};

class agent_in_memory_resource_catalog : public agent_resource_catalog {
public:
    bool next_resource_id(
        std::string & out,
        std::string & error) override;

    bool put_descriptor(
        const agent_resource_descriptor & descriptor,
        std::string & error) override;

    bool find_descriptor(
        const std::string & uri,
        agent_resource_descriptor & out,
        std::string & error) const override;

    bool list_descriptors(
        std::vector<agent_resource_descriptor> & out,
        std::string & error) const override;

private:
    mutable std::mutex mutex_;
    std::unordered_map<std::string, agent_resource_descriptor> resources_;
    uint64_t next_id_ = 1;
};

#ifdef LLAMA_MEMORY_USE_COZO
class agent_cozo_resource_catalog : public agent_resource_catalog {
public:
    agent_cozo_resource_catalog() = default;
    ~agent_cozo_resource_catalog() override;

    bool open(const std::string & path, std::string & error);
    void close();

    bool next_resource_id(
        std::string & out,
        std::string & error) override;

    bool put_descriptor(
        const agent_resource_descriptor & descriptor,
        std::string & error) override;

    bool find_descriptor(
        const std::string & uri,
        agent_resource_descriptor & out,
        std::string & error) const override;

    bool list_descriptors(
        std::vector<agent_resource_descriptor> & out,
        std::string & error) const override;

private:
    bool run(
        const std::string & script,
        const std::string & params_json,
        std::string & result_json,
        std::string & error) const;

    int32_t db_id_ = -1;
    uint64_t next_id_ = 1;
};
#endif
