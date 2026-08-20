#pragma once

#include "agent-resource-catalog.h"
#include "resource/resource-contract.h"

#include <filesystem>
#include <fstream>
#include <memory>

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

    bool get_bytes_range(
        const std::string & sha256,
        size_t offset,
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

    bool get_bytes_range(
        const std::string & sha256,
        size_t offset,
        size_t max_bytes,
        std::string & out,
        std::string & error) const override;

    bool exists_sha256(const std::string & sha256) const override;

private:
    std::filesystem::path blob_path_for_sha256(const std::string & sha256) const;

    std::filesystem::path root_;
};

class agent_catalogued_resource_store : public agent_resource_store {
public:
    agent_catalogued_resource_store(
        std::shared_ptr<agent_blob_store> blob_store,
        std::unique_ptr<agent_resource_catalog> catalog);

    bool put_bytes(
        const agent_resource_put_request & request,
        agent_resource_descriptor & out,
        std::string & error) override;

    bool read_bytes(
        const std::string & uri,
        const agent_resource_read_authority & authority,
        size_t max_bytes,
        std::string & out,
        std::string & error) const override;

    bool read_bytes_range(
        const std::string & uri,
        const agent_resource_read_authority & authority,
        size_t offset,
        size_t max_bytes,
        std::string & out,
        std::string & error) const override;

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

    bool read_text_range(
        const std::string & uri,
        const agent_resource_read_authority & authority,
        size_t offset,
        size_t max_bytes,
        std::string & out,
        std::string & error) const override;

    bool stat(
        const std::string & uri,
        const agent_resource_read_authority & authority,
        agent_resource_descriptor & out,
        std::string & error) const override;

    bool list(
        const agent_resource_read_authority & authority,
        std::vector<agent_resource_descriptor> & out,
        std::string & error) const override;

private:
    std::shared_ptr<agent_blob_store> blob_store_;
    std::unique_ptr<agent_resource_catalog> catalog_;
};

std::shared_ptr<agent_blob_store> make_agent_blob_store(
    const agent_resource_store_config & config,
    std::string & error);

std::unique_ptr<agent_resource_store> make_agent_resource_store(
    const agent_resource_store_config & config,
    std::string & error);
