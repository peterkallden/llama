#include "agent-learning-transaction-store-cozo.h"

extern "C" {
#include <cozo_c.h>
}

#include <nlohmann/json.hpp>

#include <algorithm>

using json = nlohmann::ordered_json;

namespace {

const char * schema = R"COZO(
    {
        ?[transaction_id, idempotency_key, namespace_id, session_id, project_id, turn_id, created_at, transaction_json] <- [['__probe__', '__probe__', 'local', 'default', '', '', '', '{}']]
        :create agent_learning_transactions {
            transaction_id: String =>
            idempotency_key: String,
            namespace_id: String,
            session_id: String,
            project_id: String,
            turn_id: String,
            created_at: String,
            transaction_json: String
        }
    }
    {
        ?[transaction_id] <- [['__probe__']]
        :delete agent_learning_transactions { transaction_id }
    }
)COZO";

bool relation_exists(const json & relations) {
    if (!relations.is_object() || !relations.contains("rows") || !relations["rows"].is_array()) return false;
    for (const auto & row : relations["rows"]) {
        if (row.is_array() && !row.empty() && row[0].is_string() &&
                row[0].get<std::string>() == "agent_learning_transactions") return true;
    }
    return false;
}

bool parse_transaction_rows(const std::string & raw,
        std::vector<common_learning_transaction> & result, std::string & error) {
    const auto value = json::parse(raw, nullptr, false);
    if (!value.is_object() || !value.contains("rows") || !value["rows"].is_array()) {
        error = "Cozo learning transaction query returned invalid rows";
        return false;
    }
    for (const auto & row : value["rows"]) {
        if (!row.is_array() || row.size() < 8 || !row[7].is_string()) {
            error = "Cozo learning transaction row is invalid";
            return false;
        }
        common_learning_transaction transaction;
        if (!common_learning_transaction_from_json(row[7].get<std::string>(), transaction, error)) return false;
        result.push_back(std::move(transaction));
    }
    std::sort(result.begin(), result.end(), [](const auto & left, const auto & right) {
        return left.id < right.id;
    });
    return true;
}

} // namespace

common_agent_cozo_learning_transaction_store::~common_agent_cozo_learning_transaction_store() { close(); }

bool common_agent_cozo_learning_transaction_store::run(
        const std::string & script, const std::string & params_json,
        std::string & result_json, std::string & error) const {
    if (db_id_ < 0) { error = "Cozo learning transaction store is not open"; return false; }
    char * result = cozo_run_query(db_id_, script.c_str(), params_json.empty() ? "{}" : params_json.c_str(), false);
    if (!result) { error = "Cozo learning transaction query failed without diagnostic output"; return false; }
    result_json = result;
    cozo_free_str(result);
    const auto parsed = json::parse(result_json, nullptr, false);
    if (parsed.is_object() && parsed.value("ok", true) == false) {
        error = parsed.value("message", std::string("Cozo learning transaction query failed"));
        return false;
    }
    return true;
}

bool common_agent_cozo_learning_transaction_store::open(const std::string & path, std::string & error) {
    close();
    if (path.empty()) { error = "Cozo learning transaction store requires a path"; return false; }
    char * open_error = cozo_open_db("sqlite", path.c_str(), "{}", &db_id_);
    if (open_error) { error = open_error; cozo_free_str(open_error); db_id_ = -1; return false; }
    std::string relations;
    if (!run("::relations", "{}", relations, error)) { close(); return false; }
    if (!relation_exists(json::parse(relations, nullptr, false))) {
        std::string ignored;
        if (!run(schema, "{}", ignored, error)) { close(); return false; }
    }
    return true;
}

void common_agent_cozo_learning_transaction_store::close() {
    if (db_id_ >= 0) { cozo_close_db(db_id_); db_id_ = -1; }
}

bool common_agent_cozo_learning_transaction_store::append(
        const common_learning_transaction & transaction, std::string & error) {
    error.clear();
    bool exists = false;
    if (!contains_idempotency(transaction.observation.idempotency_key, exists, error)) return false;
    if (exists) return true;
    const json params = {
        {"transaction_id", transaction.id},
        {"idempotency_key", transaction.observation.idempotency_key},
        {"namespace_id", transaction.observation.scope.namespace_id},
        {"session_id", transaction.observation.scope.session_id},
        {"project_id", transaction.observation.scope.project_id},
        {"turn_id", transaction.observation.scope.turn_id},
        {"created_at", transaction.created_at},
        {"transaction_json", common_learning_transaction_to_json(transaction)},
    };
    std::string result;
    return run("?[transaction_id, idempotency_key, namespace_id, session_id, project_id, turn_id, created_at, transaction_json] <- [[$transaction_id, $idempotency_key, $namespace_id, $session_id, $project_id, $turn_id, $created_at, $transaction_json]] :put agent_learning_transactions { transaction_id => idempotency_key, namespace_id, session_id, project_id, turn_id, created_at, transaction_json }", params.dump(), result, error);
}

bool common_agent_cozo_learning_transaction_store::contains_idempotency(
        const std::string & key, bool & contains, std::string & error) const {
    contains = false;
    std::string result;
    const auto params = json({{"idempotency_key", key}}).dump();
    if (!run("?[idempotency_key] := *agent_learning_transactions[transaction_id, idempotency_key, namespace_id, session_id, project_id, turn_id, created_at, transaction_json], idempotency_key == $idempotency_key", params, result, error)) return false;
    const auto value = json::parse(result, nullptr, false);
    if (!value.is_object() || !value.contains("rows") || !value["rows"].is_array()) { error = "Cozo idempotency query returned invalid rows"; return false; }
    contains = !value["rows"].empty();
    return true;
}

std::vector<common_learning_transaction> common_agent_cozo_learning_transaction_store::list(std::string & error) const {
    error.clear();
    std::vector<common_learning_transaction> result;
    std::string raw;
    if (!run("?[transaction_id, idempotency_key, namespace_id, session_id, project_id, turn_id, created_at, transaction_json] := *agent_learning_transactions[transaction_id, idempotency_key, namespace_id, session_id, project_id, turn_id, created_at, transaction_json]", "{}", raw, error)) return result;
    if (!parse_transaction_rows(raw, result, error)) result.clear();
    return result;
}
