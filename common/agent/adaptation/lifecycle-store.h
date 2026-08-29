#pragma once

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

// The lifecycle journal is an append-only host record for the offline
// adaptation pipeline. A status transition appends a new record; it never
// rewrites the historical event that caused it.
enum class common_learning_lifecycle_kind {
    candidate,
    corpus_revision,
    training_job,
    training_result,
    adapter,
};

enum class common_learning_lifecycle_status {
    observed,
    eligible,
    approved,
    rejected,
    revoked,
    queued,
    running,
    succeeded,
    failed,
    cancelled,
    canary,
    active,
    retired,
};

const char * common_learning_lifecycle_kind_name(common_learning_lifecycle_kind kind);
const char * common_learning_lifecycle_status_name(common_learning_lifecycle_status status);
bool parse_common_learning_lifecycle_kind(
        const std::string & value,
        common_learning_lifecycle_kind & kind,
        std::string & error);
bool parse_common_learning_lifecycle_status(
        const std::string & value,
        common_learning_lifecycle_status & status,
        std::string & error);

struct common_learning_lifecycle_record {
    int schema_version = 1;
    std::string event_id;
    std::string subject_id;
    common_learning_lifecycle_kind kind = common_learning_lifecycle_kind::candidate;
    common_learning_lifecycle_status status = common_learning_lifecycle_status::observed;
    std::string idempotency_key;
    std::string source_id;
    std::string namespace_id = "local";
    std::string project_id;
    std::string session_id;
    std::string content_hash;
    std::string created_at;
    // Canonical JSON for the typed candidate/corpus/job/result/adapter
    // payload. The lifecycle store indexes the envelope, not its payload.
    std::string payload_json;
};

bool common_learning_lifecycle_validate(
        const common_learning_lifecycle_record & record,
        size_t max_payload_bytes,
        std::string & error);

std::string common_learning_lifecycle_to_json(
        const common_learning_lifecycle_record & record);
bool common_learning_lifecycle_from_json(
        const std::string & text,
        common_learning_lifecycle_record & record,
        std::string & error);

class common_learning_lifecycle_store {
public:
    virtual ~common_learning_lifecycle_store() = default;
    virtual bool append(const common_learning_lifecycle_record & record, std::string & error) = 0;
    virtual bool contains_idempotency(const std::string & key, bool & contains, std::string & error) const = 0;
    virtual std::vector<common_learning_lifecycle_record> list(std::string & error) const = 0;
};

class common_learning_in_memory_lifecycle_store final : public common_learning_lifecycle_store {
public:
    bool append(const common_learning_lifecycle_record & record, std::string & error) override;
    bool contains_idempotency(const std::string & key, bool & contains, std::string & error) const override;
    std::vector<common_learning_lifecycle_record> list(std::string & error) const override;

private:
    std::vector<common_learning_lifecycle_record> records;
};

class common_learning_jsonl_lifecycle_store final : public common_learning_lifecycle_store {
public:
    bool open(const std::filesystem::path & path, std::string & error);
    bool append(const common_learning_lifecycle_record & record, std::string & error) override;
    bool contains_idempotency(const std::string & key, bool & contains, std::string & error) const override;
    std::vector<common_learning_lifecycle_record> list(std::string & error) const override;

private:
    std::filesystem::path path;
};
