#include "agent/adaptation/learning-transaction.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <ctime>
#include <fstream>
#include <set>
#include <sstream>

using json = nlohmann::ordered_json;

const char * common_learning_transaction_backend_name(
        common_learning_transaction_backend backend) {
    switch (backend) {
        case common_learning_transaction_backend::automatic: return "auto";
        case common_learning_transaction_backend::in_memory: return "in-memory";
        case common_learning_transaction_backend::cozo: return "cozo";
        case common_learning_transaction_backend::sqlite: return "sqlite";
        case common_learning_transaction_backend::jsonl: return "jsonl";
    }
    return "auto";
}

bool parse_common_learning_transaction_backend(
        const std::string & value,
        common_learning_transaction_backend & backend) {
    if (value == "auto") backend = common_learning_transaction_backend::automatic;
    else if (value == "in-memory" || value == "memory") backend = common_learning_transaction_backend::in_memory;
    else if (value == "cozo") backend = common_learning_transaction_backend::cozo;
    else if (value == "sqlite") backend = common_learning_transaction_backend::sqlite;
    else if (value == "jsonl") backend = common_learning_transaction_backend::jsonl;
    else return false;
    return true;
}

static std::string now_iso8601() {
    const auto now = std::chrono::system_clock::now();
    const auto seconds = std::chrono::time_point_cast<std::chrono::seconds>(now);
    const auto value = std::chrono::system_clock::to_time_t(seconds);
    char buffer[32]{};
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&value));
    return buffer;
}

static json signal_to_json(const common_learning_signal & signal) {
    return {
        {"type", common_learning_signal_type_name(signal.type)},
        {"plan_id", signal.plan_id},
        {"step_id", signal.step_id},
        {"tool_name", signal.tool_name},
        {"evidence_id", signal.evidence_id},
        {"summary", signal.summary},
    };
}

static bool signal_from_json(const json & value, common_learning_signal & signal, std::string & error) {
    if (!value.is_object() || !value.contains("type") || !value["type"].is_string()) {
        error = "learning transaction contains invalid signal";
        return false;
    }
    const auto type = value.value("type", "");
    if (type == "tool_failure") signal.type = common_learning_signal_type::tool_failure;
    else if (type == "successful_recovery") signal.type = common_learning_signal_type::successful_recovery;
    else if (type == "reflection_hint") signal.type = common_learning_signal_type::reflection_hint;
    else if (type == "user_correction") signal.type = common_learning_signal_type::user_correction;
    else { error = "learning transaction contains unknown signal type"; return false; }
    signal.plan_id = value.value("plan_id", "");
    signal.step_id = value.value("step_id", "");
    signal.tool_name = value.value("tool_name", "");
    signal.evidence_id = value.value("evidence_id", "");
    signal.summary = value.value("summary", "");
    return true;
}

static json observation_to_json(const common_learning_observation & observation) {
    json signals = json::array();
    for (const auto & signal : observation.signals) signals.push_back(signal_to_json(signal));
    return {
        {"schema_version", observation.schema_version},
        {"id", observation.id},
        {"scope", {
            {"namespace_id", observation.scope.namespace_id},
            {"session_id", observation.scope.session_id},
            {"project_id", observation.scope.project_id},
            {"turn_id", observation.scope.turn_id},
        }},
        {"source_turn_id", observation.source_turn_id},
        {"source_plan_id", observation.source_plan_id},
        {"signals", std::move(signals)},
        {"evidence_ids", observation.evidence_ids},
        {"recovery_of_signal_id", observation.recovery_of_signal_id},
        {"cause", common_learning_cause_name(observation.cause)},
        {"verification", common_learning_verification_name(observation.verification)},
        {"idempotency_key", observation.idempotency_key},
        {"content_hash", observation.content_hash},
        {"collection_allowed", observation.collection_allowed},
    };
}

static bool observation_from_json(const json & value, common_learning_observation & observation, std::string & error) {
    if (!value.is_object()) { error = "learning transaction observation is not an object"; return false; }
    observation.schema_version = value.value("schema_version", 0);
    observation.id = value.value("id", "");
    const auto scope = value.value("scope", json::object());
    if (!scope.is_object()) { error = "learning transaction scope is invalid"; return false; }
    observation.scope.namespace_id = scope.value("namespace_id", "local");
    observation.scope.session_id = scope.value("session_id", "default");
    observation.scope.project_id = scope.value("project_id", "");
    observation.scope.turn_id = scope.value("turn_id", "");
    observation.source_turn_id = value.value("source_turn_id", "");
    observation.source_plan_id = value.value("source_plan_id", "");
    const auto signals = value.value("signals", json::array());
    if (!signals.is_array()) { error = "learning transaction signals are invalid"; return false; }
    for (const auto & item : signals) {
        common_learning_signal signal;
        if (!signal_from_json(item, signal, error)) return false;
        observation.signals.push_back(std::move(signal));
    }
    observation.evidence_ids = value.value("evidence_ids", std::vector<std::string>{});
    observation.recovery_of_signal_id = value.value("recovery_of_signal_id", "");
    const auto cause = value.value("cause", "unknown");
    if (cause == "model_behavior") observation.cause = common_learning_cause::model_behavior;
    else if (cause == "host_contract") observation.cause = common_learning_cause::host_contract;
    else if (cause == "policy") observation.cause = common_learning_cause::policy;
    else if (cause == "missing_evidence") observation.cause = common_learning_cause::missing_evidence;
    else if (cause == "project_knowledge") observation.cause = common_learning_cause::project_knowledge;
    else observation.cause = common_learning_cause::unknown;
    const auto verification = value.value("verification", "unverified");
    if (verification == "host_verified") observation.verification = common_learning_verification::host_verified;
    else if (verification == "user_confirmed") observation.verification = common_learning_verification::user_confirmed;
    else observation.verification = common_learning_verification::unverified;
    observation.idempotency_key = value.value("idempotency_key", "");
    observation.content_hash = value.value("content_hash", "");
    observation.collection_allowed = value.value("collection_allowed", false);
    return true;
}

bool common_learning_transaction_validate(const common_learning_transaction & transaction, size_t max_evidence, std::string & error) {
    error.clear();
    if (transaction.schema_version != 1) { error = "unsupported learning transaction schema"; return false; }
    if (transaction.id.empty()) { error = "learning transaction requires id"; return false; }
    if (transaction.created_at.empty()) { error = "learning transaction requires created_at"; return false; }
    if (transaction.observation.id != transaction.id) { error = "transaction and observation ids must match"; return false; }
    return common_learning_observation_validate(transaction.observation, max_evidence, error);
}

std::string common_learning_transaction_to_json(const common_learning_transaction & transaction) {
    return json{
        {"schema_version", transaction.schema_version},
        {"id", transaction.id},
        {"created_at", transaction.created_at},
        {"observation", observation_to_json(transaction.observation)},
    }.dump();
}

bool common_learning_transaction_from_json(const std::string & text, common_learning_transaction & transaction, std::string & error) {
    try {
        const auto value = json::parse(text);
        if (!value.is_object()) { error = "learning transaction is not an object"; return false; }
        transaction.schema_version = value.value("schema_version", 0);
        transaction.id = value.value("id", "");
        transaction.created_at = value.value("created_at", "");
        if (!observation_from_json(value.value("observation", json::object()), transaction.observation, error)) return false;
        return true;
    } catch (const std::exception & exception) {
        error = std::string("invalid learning transaction JSON: ") + exception.what();
        return false;
    }
}

bool common_learning_in_memory_transaction_store::append(const common_learning_transaction & transaction, std::string & error) {
    error.clear();
    bool exists = false;
    if (!contains_idempotency(transaction.observation.idempotency_key, exists, error)) return false;
    if (exists) return true;
    transactions.push_back(transaction);
    return true;
}

bool common_learning_in_memory_transaction_store::contains_idempotency(const std::string & key, bool & contains, std::string & error) const {
    error.clear();
    contains = std::any_of(transactions.begin(), transactions.end(), [&](const auto & item) {
        return item.observation.idempotency_key == key;
    });
    return true;
}

std::vector<common_learning_transaction> common_learning_in_memory_transaction_store::list(std::string & error) const {
    error.clear();
    return transactions;
}

bool common_learning_jsonl_transaction_store::open(const std::filesystem::path & value, std::string & error) {
    error.clear();
    if (value.empty()) { error = "learning transaction store requires a path"; return false; }
    path = value;
    std::error_code filesystem_error;
    if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path(), filesystem_error);
    if (filesystem_error) { error = filesystem_error.message(); return false; }
    std::ofstream output(path, std::ios::app);
    if (!output) { error = "could not open learning transaction store"; return false; }
    return true;
}

std::vector<common_learning_transaction> common_learning_jsonl_transaction_store::list(std::string & error) const {
    error.clear();
    std::vector<common_learning_transaction> result;
    std::ifstream input(path);
    if (!input) { error = "could not read learning transaction store"; return result; }
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty()) continue;
        common_learning_transaction transaction;
        if (!common_learning_transaction_from_json(line, transaction, error)) return {};
        result.push_back(std::move(transaction));
    }
    return result;
}

bool common_learning_jsonl_transaction_store::contains_idempotency(const std::string & key, bool & contains, std::string & error) const {
    contains = false;
    const auto transactions = list(error);
    if (!error.empty()) return false;
    contains = std::any_of(transactions.begin(), transactions.end(), [&](const auto & item) {
        return item.observation.idempotency_key == key;
    });
    return true;
}

bool common_learning_jsonl_transaction_store::append(const common_learning_transaction & transaction, std::string & error) {
    error.clear();
    bool exists = false;
    if (!contains_idempotency(transaction.observation.idempotency_key, exists, error)) return false;
    if (exists) return true;
    std::ofstream output(path, std::ios::app);
    if (!output) { error = "could not append learning transaction"; return false; }
    output << common_learning_transaction_to_json(transaction) << '\n';
    if (!output) { error = "could not flush learning transaction"; return false; }
    return true;
}

common_learning_transaction_observer::common_learning_transaction_observer(
        common_learning_transaction_store & store,
        common_learning_transaction_observer_config config)
    : store(store), config(config) {}

bool common_learning_transaction_observer::observe(
        const common_agent_request & request,
        const common_plan_state & plan,
        const common_agent_result & result,
        std::string & error) {
    error.clear();
    if (result.learning_signals.empty()) return true;
    if (!common_learning_domain_policy_allows(
            config.domain_policy, request, plan, result)) return true;
    common_learning_observation observation;
    observation.scope = common_agent_scope_from_request(request);
    observation.source_turn_id = request.turn_id;
    observation.source_plan_id = plan.id;
    observation.signals = result.learning_signals;
    observation.collection_allowed = config.collection_allowed;
    std::set<std::string> evidence;
    for (const auto & signal : observation.signals) if (!signal.evidence_id.empty()) evidence.insert(signal.evidence_id);
    observation.evidence_ids.assign(evidence.begin(), evidence.end());
    if (observation.source_plan_id.empty() && !observation.signals.front().plan_id.empty()) observation.source_plan_id = observation.signals.front().plan_id;
    std::ostringstream idempotency;
    idempotency << observation.source_turn_id << ":" << observation.source_plan_id;
    for (const auto & signal : observation.signals) {
        idempotency << ":" << common_learning_signal_type_name(signal.type)
                    << ":" << signal.step_id
                    << ":" << signal.tool_name
                    << ":" << signal.evidence_id;
    }
    observation.idempotency_key = idempotency.str();
    observation.id = "learning://observation/" + observation.idempotency_key;
    observation.content_hash = common_learning_observation_hash(observation);
    observation.cause = common_learning_classify_result(result, config.cause_classifier);
    observation.recovery_of_signal_id = common_learning_recovery_reference(result);
    for (const auto & signal : observation.signals) {
        if (signal.type == common_learning_signal_type::successful_recovery) observation.verification = common_learning_verification::host_verified;
        if (signal.type == common_learning_signal_type::user_correction) observation.verification = common_learning_verification::user_confirmed;
    }
    if (!common_learning_observation_qualifies(observation)) return true;
    common_learning_transaction transaction;
    transaction.id = observation.id;
    transaction.observation = observation;
    transaction.created_at = now_iso8601();
    if (!common_learning_transaction_validate(transaction, config.max_evidence, error)) return false;
    return store.append(transaction, error);
}
