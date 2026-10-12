#include "agent-learning-transaction-store.h"
#include "agent-learning-lifecycle-store.h"

#if defined(LLAMA_AGENT_ADAPTATION_TEST_COZO)
#include "agent-learning-transaction-store-cozo.h"
#endif
#if defined(LLAMA_AGENT_ADAPTATION_TEST_SQLITE)
#include "agent-learning-transaction-store-sqlite.h"
#endif

#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <iostream>

static size_t require_index = 0;
static void require(bool condition) {
    ++require_index;
    if (!condition) {
        std::cerr << "backend contract check failed #" << require_index << '\n';
        std::abort();
    }
}

static common_learning_transaction transaction() {
    common_learning_transaction value;
    value.id = "learning://observation/backend-test";
    value.created_at = "2026-08-29T00:00:00Z";
    value.observation.id = value.id;
    value.observation.scope.namespace_id = "local";
    value.observation.scope.session_id = "session-1";
    value.observation.scope.project_id = "project-1";
    value.observation.scope.turn_id = "turn-1";
    value.observation.idempotency_key = "backend-test-key";
    value.observation.content_hash = "hash";
    return value;
}

static void assert_round_trip(common_learning_transaction_store & store) {
    std::string error;
    const auto value = transaction();
    require(store.append(value, error));
    require(store.append(value, error));
    const auto values = store.list(error);
    require(error.empty());
    require(values.size() == 1);
    require(values.front().id == value.id);
    bool contains = false;
    require(store.contains_idempotency(value.observation.idempotency_key, contains, error));
    require(contains);
}

static common_learning_lifecycle_record lifecycle_record() {
    common_learning_lifecycle_record value;
    value.event_id = "lifecycle-event-1";
    value.subject_id = "learning://candidate/backend-test";
    value.kind = common_learning_lifecycle_kind::candidate;
    value.status = common_learning_lifecycle_status::observed;
    value.idempotency_key = "lifecycle-backend-test-key";
    value.source_id = "learning://observation/backend-test";
    value.namespace_id = "local";
    value.project_id = "project-1";
    value.session_id = "session-1";
    value.content_hash = "identity:fnv1a64:0123456789abcdef";
    value.created_at = "2026-08-29T00:00:00Z";
    value.payload_json = "{\"kind\":\"candidate\"}";
    return value;
}

static void assert_lifecycle_round_trip(common_learning_lifecycle_store & store) {
    std::string error;
    const auto value = lifecycle_record();
    require(store.append(value, error));
    if (!store.append(value, error)) {
        std::cerr << "lifecycle duplicate append error: " << error << '\n';
        require(false);
    }
    const auto values = store.list(error);
    require(error.empty());
    require(values.size() == 1);
    require(values.front().subject_id == value.subject_id);
    bool contains = false;
    require(store.contains_idempotency(value.idempotency_key, contains, error));
    require(contains);
}

int main() {
    std::string error;
    common_learning_transaction_backend parsed_backend;
    require(parse_common_learning_transaction_backend("memory", parsed_backend));
    require(parsed_backend == common_learning_transaction_backend::in_memory);

    auto memory = make_agent_learning_transaction_store("auto", {}, error);
    require(memory && error.empty());
    assert_round_trip(*memory);

    auto invalid_memory = make_agent_learning_transaction_store("in-memory", "not-a-path", error);
    require(!invalid_memory);
    require(error.find("does not accept a path") != std::string::npos);

    const auto jsonl_path = std::filesystem::temp_directory_path() / "llama-agent-learning-backend-test.jsonl";
    std::error_code ignored;
    std::filesystem::remove(jsonl_path, ignored);
    auto jsonl = make_agent_learning_transaction_store("jsonl", jsonl_path.string(), error);
    require(jsonl && error.empty());
    assert_round_trip(*jsonl);
    std::filesystem::remove(jsonl_path, ignored);

    auto lifecycle_memory = make_agent_learning_lifecycle_store("auto", {}, error);
    require(lifecycle_memory && error.empty());
    assert_lifecycle_round_trip(*lifecycle_memory);
    auto lifecycle_jsonl = make_agent_learning_lifecycle_store("jsonl", jsonl_path.string(), error);
    require(lifecycle_jsonl && error.empty());
    assert_lifecycle_round_trip(*lifecycle_jsonl);
    std::filesystem::remove(jsonl_path, ignored);

    const auto persistent_path = std::filesystem::temp_directory_path() / "llama-agent-learning-backend-test.db";
    std::filesystem::remove(persistent_path, ignored);
    auto automatic_persistent = make_agent_learning_transaction_store("auto", persistent_path.string(), error);
#if defined(LLAMA_AGENT_ADAPTATION_TEST_COZO) || defined(LLAMA_AGENT_ADAPTATION_TEST_SQLITE)
    require(automatic_persistent && error.empty());
#if defined(LLAMA_AGENT_ADAPTATION_TEST_COZO)
    require(dynamic_cast<common_agent_cozo_learning_transaction_store *>(automatic_persistent.get()) != nullptr);
#endif
    assert_round_trip(*automatic_persistent);
#else
    require(!automatic_persistent);
    require(error.find("persistent adaptation transaction backend") != std::string::npos);
#endif
    std::filesystem::remove(persistent_path, ignored);

#if defined(LLAMA_AGENT_ADAPTATION_TEST_COZO)
    const auto cozo_path = std::filesystem::temp_directory_path() / "llama-agent-learning-backend-cozo-test.db";
    std::filesystem::remove(cozo_path, ignored);
    auto cozo = make_agent_learning_transaction_store("cozo", cozo_path.string(), error);
    require(cozo && error.empty());
    assert_round_trip(*cozo);
    std::filesystem::remove(cozo_path, ignored);

    const auto lifecycle_cozo_path = std::filesystem::temp_directory_path() / "llama-agent-learning-lifecycle-cozo-test.db";
    std::filesystem::remove(lifecycle_cozo_path, ignored);
    auto lifecycle_cozo = make_agent_learning_lifecycle_store("cozo", lifecycle_cozo_path.string(), error);
    require(lifecycle_cozo && error.empty());
    assert_lifecycle_round_trip(*lifecycle_cozo);
    std::filesystem::remove(lifecycle_cozo_path, ignored);
#endif
#if defined(LLAMA_AGENT_ADAPTATION_TEST_SQLITE)
    const auto sqlite_path = std::filesystem::temp_directory_path() / "llama-agent-learning-backend-sqlite-test.db";
    std::filesystem::remove(sqlite_path, ignored);
    auto sqlite = make_agent_learning_transaction_store("sqlite", sqlite_path.string(), error);
    require(sqlite && error.empty());
    assert_round_trip(*sqlite);
    std::filesystem::remove(sqlite_path, ignored);

    const auto lifecycle_sqlite_path = std::filesystem::temp_directory_path() / "llama-agent-learning-lifecycle-sqlite-test.db";
    std::filesystem::remove(lifecycle_sqlite_path, ignored);
    auto lifecycle_sqlite = make_agent_learning_lifecycle_store("sqlite", lifecycle_sqlite_path.string(), error);
    require(lifecycle_sqlite && error.empty());
    assert_lifecycle_round_trip(*lifecycle_sqlite);
    std::filesystem::remove(lifecycle_sqlite_path, ignored);
#endif
    return 0;
}
