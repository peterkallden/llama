#include "agent/adaptation/learning-transaction.h"

#include <filesystem>
#include <vector>

#define CHECK(condition) do { if (!(condition)) return 1; } while (false)

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
    value.failures.push_back({"tool.invalid_arguments", common_agent_failure_class::validation,
        "tool_execution", "dataset.inspect", "step-1", "evidence-1", false,
        "tool failed", "{}"});
    value.learning_signals.push_back({common_learning_signal_type::tool_failure, "plan-1", "step-1", "dataset.inspect", "evidence-1", "tool failed"});
    value.learning_signals.front().tool_family = "diagnostics";
    value.learning_signals.front().provider_kind = "openapi";
    return value;
}

int main() {
    std::string error;
    common_learning_in_memory_transaction_store memory_store;
    common_learning_transaction_observer_config observer_config;
    observer_config.collection_allowed = true;
    observer_config.max_evidence = 4;
    common_learning_transaction_observer observer(memory_store, observer_config);
    auto req = request();
    auto pl = plan();
    auto failed = failure_result();
    CHECK(observer.observe(req, pl, failed, error));
    CHECK(observer.observe(req, pl, failed, error));
    auto transactions = memory_store.list(error);
    CHECK(error.empty() && transactions.size() == 1);
    CHECK(transactions.front().observation.scope.project_id == "project-1");
    CHECK(transactions.front().observation.cause == common_learning_cause::host_contract);
    CHECK(transactions.front().observation.signals.front().tool_family == "diagnostics");
    CHECK(transactions.front().observation.signals.front().provider_kind == "openapi");
    CHECK(transactions.front().observation.recovery_of_signal_id.empty());
    CHECK(transactions.front().observation.verification == common_learning_verification::unverified);

    common_agent_result generic_sources;
    generic_sources.learning_signals.push_back({common_learning_signal_type::planning_revision,
        "plan-1", "step-1", {}, "planning-evidence", "plan changed"});
    generic_sources.learning_signals.push_back({common_learning_signal_type::research_verification,
        "plan-1", "step-1", {}, "research-evidence", "research verified"});
    common_learning_in_memory_transaction_store generic_store;
    common_learning_transaction_observer_config generic_config;
    generic_config.collection_allowed = true;
    generic_config.domain_policy.configured = true;
    generic_config.domain_policy.planning = true;
    generic_config.domain_policy.research = true;
    common_learning_transaction_observer generic_observer(generic_store, generic_config);
    CHECK(generic_observer.observe(req, pl, generic_sources, error));
    const auto generic_transactions = generic_store.list(error);
    CHECK(generic_transactions.size() == 1);
    CHECK(generic_transactions.front().observation.signals.size() == 2);

    common_learning_in_memory_transaction_store relation_store;
    common_learning_transaction_observer_config relation_config;
    relation_config.collection_allowed = true;
    relation_config.domain_policy.configured = true;
    relation_config.domain_policy.procedure_learning = true;
    size_t relation_callbacks = 0;
    bool relation_callback_valid = true;
    relation_config.host_relation_observer = [&](const auto & callback_request,
            const auto & callback_plan, const auto & callback_result,
            const auto & transaction, std::string &) {
        relation_callback_valid = callback_request.turn_id == "turn-1" &&
            callback_plan.id == "plan-1" && callback_result.learning_signals.size() == 1 &&
            transaction.observation.verification == common_learning_verification::host_verified;
        ++relation_callbacks;
        return relation_callback_valid;
    };
    common_learning_transaction_observer relation_observer(relation_store, relation_config);
    common_agent_result procedure_result;
    procedure_result.learning_signals.push_back({common_learning_signal_type::procedure_verification,
        "plan-1", {}, {}, "procedure-memory-1", "host verified procedure"});
    CHECK(relation_observer.observe(req, pl, procedure_result, error));
    CHECK(relation_callbacks == 1);
    CHECK(relation_callback_valid);
    CHECK(relation_store.list(error).size() == 1);

    common_learning_transaction_query query;
    query.scope.session_id = "session-1";
    query.tool_family = "diagnostics";
    query.provider_kind = "openapi";
    query.max_results = 4;
    query.max_scan = 4;
    const auto queried = common_learning_query_transactions(memory_store, query, error);
    CHECK(error.empty() && queried.transactions.size() == 1 && !queried.truncated);
    const auto replay_ids = common_learning_select_replay_transaction_ids(memory_store, query, error);
    CHECK(error.empty() && replay_ids.size() == 1 && replay_ids[0] == transactions.front().id);
    query.provider_kind = "mcp";
    CHECK(common_learning_query_transactions(memory_store, query, error).transactions.empty());

    auto second_failure = failed;
    second_failure.learning_signals.front().evidence_id = "evidence-2";
    CHECK(observer.observe(req, pl, second_failure, error));
    CHECK(memory_store.list(error).size() == 2);

    common_agent_result ordinary;
    ordinary.response = "hello";
    CHECK(observer.observe(req, pl, ordinary, error));
    CHECK(memory_store.list(error).size() == 2);

    common_learning_in_memory_transaction_store denied_store;
    common_learning_transaction_observer_config denied_config;
    denied_config.collection_allowed = false;
    denied_config.max_evidence = 4;
    common_learning_transaction_observer denied(denied_store, denied_config);
    CHECK(denied.observe(req, pl, failed, error));
    CHECK(denied_store.list(error).empty());

    common_learning_in_memory_transaction_store routed_store;
    common_learning_transaction_observer_config routed_config;
    routed_config.collection_allowed = true;
    routed_config.max_evidence = 4;
    std::vector<common_adaptation_evidence_source_match> source_matches;
    routed_config.source_observer = [&](const auto & match,
            const common_learning_transaction &, std::string &) {
        source_matches.push_back(match);
        return true;
    };
    common_learning_transaction_observer routed(routed_store, routed_config);
    auto recovered = failure_result();
    recovered.learning_signals.push_back({common_learning_signal_type::successful_recovery,
        "plan-1", "step-1", "dataset.inspect", "evidence-1", "recovered"});
    CHECK(routed.observe(req, pl, recovered, error));
    CHECK(source_matches.size() == 1);
    CHECK(source_matches.front().source == common_adaptation_evidence_source::tool_repair);
    CHECK(source_matches.front().candidate_ready);

    common_learning_in_memory_transaction_store best_effort_store;
    common_learning_transaction_observer_config best_effort_config;
    best_effort_config.collection_allowed = true;
    best_effort_config.max_evidence = 4;
    best_effort_config.source_observer = [](const auto &, const auto &, std::string & callback_error) {
        callback_error = "collector unavailable";
        return false;
    };
    common_learning_transaction_observer best_effort(best_effort_store, best_effort_config);
    CHECK(best_effort.observe(req, pl, recovered, error));
    CHECK(best_effort_store.list(error).size() == 1);

    const auto path = std::filesystem::temp_directory_path() / "llama-agent-learning-test.jsonl";
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
    common_learning_jsonl_transaction_store file_store;
    CHECK(file_store.open(path, error));
    CHECK(observer.observe(req, pl, failed, error));
    const auto transaction = transactions.front();
    if (!file_store.append(transaction, error) ||
            !file_store.append(transaction, error) ||
            file_store.list(error).size() != 1) return 1;
    std::filesystem::remove(path, ignored);
    return 0;
}
