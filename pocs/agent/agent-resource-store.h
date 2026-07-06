#pragma once

#include "runtime-resource.h"

#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <unordered_map>

bool validate_agent_resource_store_config(
    const agent_resource_store_config & config,
    std::string & error);

class agent_in_memory_blob_store : public agent_blob_store {
public:
    bool put_bytes(
        const std::string & bytes,
        agent_blob_descriptor & out,
        std::string & error) override;

    bool get_bytes(
        const std::string & sha256,
        size_t max_bytes,
        std::string & out,
        std::string & error) const override;

    bool exists_sha256(const std::string & sha256) const override;

private:
    mutable std::mutex mutex_;
    std::unordered_map<std::string, std::string> blobs_;
};

class agent_filesystem_blob_store : public agent_blob_store {
public:
    explicit agent_filesystem_blob_store(std::string root);

    bool put_bytes(
        const std::string & bytes,
        agent_blob_descriptor & out,
        std::string & error) override;

    bool get_bytes(
        const std::string & sha256,
        size_t max_bytes,
        std::string & out,
        std::string & error) const override;

    bool exists_sha256(const std::string & sha256) const override;

private:
    std::filesystem::path blob_path_for_sha256(const std::string & sha256) const;

    std::filesystem::path root_;
};

class agent_in_memory_resource_store : public agent_resource_store {
public:
    explicit agent_in_memory_resource_store(std::shared_ptr<agent_blob_store> blob_store = nullptr);

    bool put_text(
        const agent_resource_put_request & request,
        agent_resource_descriptor & out,
        std::string & error) override;

    bool read_text(
        const std::string & uri,
        const agent_resource_read_authority & authority,
        size_t max_bytes,
        std::string & out,
        std::string & error) const override;

    bool stat(
        const std::string & uri,
        const agent_resource_read_authority & authority,
        agent_resource_descriptor & out,
        std::string & error) const override;

private:
    bool resolve_descriptor(
        const std::string & uri,
        const agent_resource_read_authority & authority,
        agent_resource_descriptor & out,
        std::string & error) const;

    std::shared_ptr<agent_blob_store> blob_store_;
    mutable std::mutex mutex_;
    std::unordered_map<std::string, agent_resource_descriptor> resources_;
    uint64_t next_id_ = 1;
};

#ifdef LLAMA_MEMORY_USE_COZO
class agent_cozo_resource_store : public agent_resource_store {
public:
    explicit agent_cozo_resource_store(std::shared_ptr<agent_blob_store> blob_store = nullptr);
    ~agent_cozo_resource_store() override;

    bool open(const std::string & path, std::string & error);
    void close();

    bool put_text(
        const agent_resource_put_request & request,
        agent_resource_descriptor & out,
        std::string & error) override;

    bool read_text(
        const std::string & uri,
        const agent_resource_read_authority & authority,
        size_t max_bytes,
        std::string & out,
        std::string & error) const override;

    bool stat(
        const std::string & uri,
        const agent_resource_read_authority & authority,
        agent_resource_descriptor & out,
        std::string & error) const override;

private:
    bool run(
        const std::string & script,
        const std::string & params_json,
        std::string & result_json,
        std::string & error) const;

    bool get_descriptor(
        const std::string & uri,
        agent_resource_descriptor & out,
        std::string & error) const;

    std::shared_ptr<agent_blob_store> blob_store_;
    int32_t db_id_ = -1;
    uint64_t next_id_ = 1;
};
#endif

std::shared_ptr<agent_blob_store> make_agent_blob_store(
    const agent_resource_store_config & config,
    std::string & error);

std::unique_ptr<agent_resource_store> make_agent_resource_store(
    const agent_resource_store_config & config,
    std::string & error);
