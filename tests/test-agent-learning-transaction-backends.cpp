#include "agent-learning-transaction-store.h"

#if defined(LLAMA_AGENT_ADAPTATION_TEST_COZO)
#include "agent-learning-transaction-store-cozo.h"
#endif
#if defined(LLAMA_AGENT_ADAPTATION_TEST_SQLITE)
#include "agent-learning-transaction-store-sqlite.h"
#endif

#include <cassert>
#include <cstdlib>
#include <filesystem>

static void require(bool condition) {
    if (!condition) std::abort();
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
#endif
#if defined(LLAMA_AGENT_ADAPTATION_TEST_SQLITE)
    const auto sqlite_path = std::filesystem::temp_directory_path() / "llama-agent-learning-backend-sqlite-test.db";
    std::filesystem::remove(sqlite_path, ignored);
    auto sqlite = make_agent_learning_transaction_store("sqlite", sqlite_path.string(), error);
    require(sqlite && error.empty());
    assert_round_trip(*sqlite);
    std::filesystem::remove(sqlite_path, ignored);
#endif
    return 0;
}
