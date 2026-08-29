#include "agent/adaptation/lifecycle-store.h"

#include <cassert>
#include <filesystem>

static common_learning_lifecycle_record record(
        const std::string & event_id,
        const std::string & status,
        const std::string & idempotency) {
    common_learning_lifecycle_record value;
    value.event_id = event_id;
    value.subject_id = "learning://candidate/compact-binding";
    value.kind = common_learning_lifecycle_kind::candidate;
    value.idempotency_key = idempotency;
    value.source_id = "learning://observation/1";
    value.namespace_id = "local";
    value.project_id = "agent-tests";
    value.session_id = "session-1";
    value.content_hash = "identity:fnv1a64:0123456789abcdef";
    value.created_at = "2026-08-29T00:00:00Z";
    value.payload_json = "{\"approved_prompt\":\"compact\",\"status\":\"" + status + "\"}";
    if (status == "approved") value.status = common_learning_lifecycle_status::approved;
    else if (status == "canary") value.status = common_learning_lifecycle_status::canary;
    return value;
}

static void assert_round_trip(common_learning_lifecycle_store & store) {
    std::string error;
    const auto observed = record("event-1", "observed", "idempotency-1");
    assert(store.append(observed, error));
    assert(store.append(observed, error));
    auto values = store.list(error);
    assert(error.empty() && values.size() == 1);
    assert(values.front().subject_id == observed.subject_id);

    const auto approved = record("event-2", "approved", "idempotency-2");
    assert(store.append(approved, error));
    values = store.list(error);
    assert(error.empty() && values.size() == 2);

    const auto canary = record("event-3", "canary", "idempotency-3");
    assert(store.append(canary, error));
    values = store.list(error);
    assert(error.empty() && values.size() == 3);

    auto conflict = observed;
    conflict.payload_json = "{\"different\":true}";
    assert(!store.append(conflict, error));
    assert(error.find("conflicts") != std::string::npos);
}

int main() {
    std::string error;
    common_learning_in_memory_lifecycle_store memory;
    assert_round_trip(memory);

    auto invalid = record("event-invalid", "observed", "idempotency-invalid");
    invalid.payload_json = "[]";
    assert(!memory.append(invalid, error));
    assert(error.find("JSON object") != std::string::npos);

    const auto path = std::filesystem::temp_directory_path() / "llama-agent-learning-lifecycle-test.jsonl";
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
    {
        common_learning_jsonl_lifecycle_store jsonl;
        assert(jsonl.open(path, error));
        assert_round_trip(jsonl);
    }
    {
        common_learning_jsonl_lifecycle_store reopened;
        assert(reopened.open(path, error));
        const auto values = reopened.list(error);
        assert(error.empty() && values.size() == 3);
        assert(values.back().status == common_learning_lifecycle_status::canary);
    }
    std::filesystem::remove(path, ignored);
    return 0;
}
