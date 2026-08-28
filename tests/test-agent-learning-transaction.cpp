#include "agent/adaptation/learning-transaction.h"

#include <cassert>
#include <filesystem>

static common_agent_request request() {
    common_agent_request value;
    value.turn_id = "turn-1";
    value.session_id = "session-1";
    value.project_id = "project-1";
    return value;
}

static common_plan_state plan() {
    common_plan_state value;
    value.id = "plan-1";
    return value;
}

static common_agent_result failure_result() {
    common_agent_result value;
    value.error = "tool failed";
    value.learning_signals.push_back({common_learning_signal_type::tool_failure, "plan-1", "step-1", "data.inspect", "evidence-1", "tool failed"});
    return value;
}

int main() {
    std::string error;
    common_learning_in_memory_transaction_store memory_store;
    common_learning_transaction_observer observer(memory_store, {true, 4});
    auto req = request();
    auto pl = plan();
    auto failed = failure_result();
    assert(observer.observe(req, pl, failed, error));
    assert(observer.observe(req, pl, failed, error));
    auto transactions = memory_store.list(error);
    assert(error.empty() && transactions.size() == 1);
    assert(transactions.front().observation.scope.project_id == "project-1");
    assert(transactions.front().observation.verification == common_learning_verification::unverified);

    auto second_failure = failed;
    second_failure.learning_signals.front().evidence_id = "evidence-2";
    assert(observer.observe(req, pl, second_failure, error));
    assert(memory_store.list(error).size() == 2);

    common_agent_result ordinary;
    ordinary.response = "hello";
    assert(observer.observe(req, pl, ordinary, error));
    assert(memory_store.list(error).size() == 1);

    common_learning_in_memory_transaction_store denied_store;
    common_learning_transaction_observer denied(denied_store, {false, 4});
    assert(denied.observe(req, pl, failed, error));
    assert(denied_store.list(error).empty());

    const auto path = std::filesystem::temp_directory_path() / "llama-agent-learning-test.jsonl";
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
    common_learning_jsonl_transaction_store file_store;
    assert(file_store.open(path, error));
    assert(observer.observe(req, pl, failed, error));
    const auto & transaction = transactions.front();
    assert(file_store.append(transaction, error));
    assert(file_store.append(transaction, error));
    assert(file_store.list(error).size() == 1);
    std::filesystem::remove(path, ignored);
    return 0;
}
