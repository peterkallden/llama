#include "plan-sqlite.h"

#include <nlohmann/json.hpp>

#include <map>

using json = nlohmann::ordered_json;

namespace {

const char * scope_name(common_plan_scope value) { return value == common_plan_scope::session ? "session" : value == common_plan_scope::project ? "project" : value == common_plan_scope::global ? "global" : "turn"; }
common_plan_scope parse_scope(const std::string & value) { return value == "session" ? common_plan_scope::session : value == "project" ? common_plan_scope::project : value == "global" ? common_plan_scope::global : common_plan_scope::turn; }
const char * kind_name(common_plan_kind value) { return value == common_plan_kind::blueprint ? "blueprint" : "task"; }
common_plan_kind parse_kind(const std::string & value) { return value == "blueprint" ? common_plan_kind::blueprint : common_plan_kind::task; }
const char * status_name(common_plan_status value) { switch (value) { case common_plan_status::proposed: return "proposed"; case common_plan_status::active: return "active"; case common_plan_status::completed: return "completed"; case common_plan_status::blocked: return "blocked"; case common_plan_status::failed: return "failed"; case common_plan_status::cancelled: return "cancelled"; } return "proposed"; }
common_plan_status parse_status(const std::string & value) { if (value == "active") return common_plan_status::active; if (value == "completed") return common_plan_status::completed; if (value == "blocked") return common_plan_status::blocked; if (value == "failed") return common_plan_status::failed; if (value == "cancelled") return common_plan_status::cancelled; return common_plan_status::proposed; }
const char * step_status_name(common_plan_step_status value) { switch (value) { case common_plan_step_status::pending: return "pending"; case common_plan_step_status::active: return "active"; case common_plan_step_status::completed: return "completed"; case common_plan_step_status::blocked: return "blocked"; case common_plan_step_status::skipped: return "skipped"; case common_plan_step_status::failed: return "failed"; } return "pending"; }
common_plan_step_status parse_step_status(const std::string & value) { if (value == "active") return common_plan_step_status::active; if (value == "completed") return common_plan_step_status::completed; if (value == "blocked") return common_plan_step_status::blocked; if (value == "skipped") return common_plan_step_status::skipped; if (value == "failed") return common_plan_step_status::failed; return common_plan_step_status::pending; }
const char * step_mode_name(common_plan_step_mode value) { return value == common_plan_step_mode::tool ? "tool" : value == common_plan_step_mode::reasoning ? "reasoning" : "final_response"; }
common_plan_step_mode parse_step_mode(const std::string & value) { return value == "tool" ? common_plan_step_mode::tool : value == "reasoning" ? common_plan_step_mode::reasoning : common_plan_step_mode::final_response; }
common_plan_operation_kind parse_operation_kind(const std::string & value) {
    const common_plan_operation_kind values[] = {
        common_plan_operation_kind::create_plan, common_plan_operation_kind::revise_goal,
        common_plan_operation_kind::add_step, common_plan_operation_kind::revise_step,
        common_plan_operation_kind::replace_step, common_plan_operation_kind::remove_step,
        common_plan_operation_kind::activate_step, common_plan_operation_kind::reset_step,
        common_plan_operation_kind::complete_step, common_plan_operation_kind::block_step,
        common_plan_operation_kind::unblock_step, common_plan_operation_kind::fail_step,
        common_plan_operation_kind::skip_step, common_plan_operation_kind::add_dependency,
        common_plan_operation_kind::remove_dependency, common_plan_operation_kind::add_constraint,
        common_plan_operation_kind::add_assumption, common_plan_operation_kind::invalidate_assumption,
        common_plan_operation_kind::record_observation, common_plan_operation_kind::set_next_action,
        common_plan_operation_kind::request_replan, common_plan_operation_kind::complete_plan,
        common_plan_operation_kind::fail_plan,
    };
    for (const auto candidate : values) if (value == common_plan_operation_kind_name(candidate)) return candidate;
    return common_plan_operation_kind::add_step;
}

json serialize_plan(const common_plan_state & plan) {
    json steps = json::array();
    for (const auto & step : plan.steps) {
        json value = {
            {"id", step.id}, {"title", step.title}, {"objective", step.objective},
            {"intended_contribution", step.intended_contribution},
            {"mode", step_mode_name(common_plan_step_effective_mode(step))},
            {"status", step_status_name(step.status)}, {"depends_on", step.depends_on},
            {"blocked_by", step.blocked_by}, {"required_evidence", step.required_evidence},
            {"source_memory_ids", step.source_memory_ids}, {"optional", step.optional},
            {"generated_from_memory", step.generated_from_memory},
            {"created_at", step.created_at}, {"updated_at", step.updated_at}
        };
        if (step.selected_tool) value["selected_tool"] = *step.selected_tool;
        if (step.result_summary) value["result_summary"] = *step.result_summary;
        if (step.semantic_alias) value["semantic_alias"] = *step.semantic_alias;
        if (step.tool_call) value["tool_call"] = {{"name", step.tool_call->name}, {"arguments_json", step.tool_call->arguments_json}};
        steps.push_back(std::move(value));
    }
    json constraints = json::array();
    for (const auto & value : plan.constraints) constraints.push_back({{"id", value.id}, {"description", value.description}, {"hard", value.hard}});
    json assumptions = json::array();
    for (const auto & value : plan.assumptions) assumptions.push_back({{"id", value.id}, {"statement", value.statement}, {"confidence", value.confidence}, {"valid", value.valid}, {"evidence_ids", value.evidence_ids}});
    json observations = json::array();
    for (const auto & value : plan.observations) observations.push_back({{"id", value.id}, {"source", value.source}, {"summary", value.summary}, {"confidence", value.confidence}, {"evidence_ids", value.evidence_ids}, {"created_at", value.created_at}});
    return {{"id", plan.id}, {"namespace_id", plan.namespace_id}, {"session_id", plan.session_id}, {"project_id", plan.project_id}, {"turn_id", plan.turn_id}, {"source_revision", plan.source_revision}, {"kind", kind_name(plan.kind)}, {"derived_from_plan_id", plan.derived_from_plan_id}, {"scope", scope_name(plan.scope)}, {"status", status_name(plan.status)}, {"purpose", plan.purpose}, {"goal", plan.goal}, {"success_criteria", plan.success_criteria}, {"required_capabilities", plan.required_capabilities}, {"steps", steps}, {"constraints", constraints}, {"assumptions", assumptions}, {"observations", observations}, {"active_step_id", plan.active_step_id}, {"next_action", plan.next_action}, {"version", plan.version}, {"created_at", plan.created_at}, {"updated_at", plan.updated_at}};
}

bool deserialize_plan(const std::string & text, common_plan_state & plan) {
    const auto value = json::parse(text, nullptr, false);
    if (!value.is_object()) return false;
    plan = {};
    plan.id = value.value("id", std::string{});
    plan.namespace_id = value.value("namespace_id", std::string("local"));
    plan.session_id = value.value("session_id", std::string{});
    plan.project_id = value.value("project_id", std::string{});
    plan.turn_id = value.value("turn_id", std::string{});
    plan.source_revision = value.value("source_revision", std::string{});
    plan.kind = parse_kind(value.value("kind", std::string("task")));
    plan.scope = parse_scope(value.value("scope", std::string("turn")));
    plan.status = parse_status(value.value("status", std::string("proposed")));
    plan.purpose = value.value("purpose", std::string{});
    plan.goal = value.value("goal", std::string{});
    plan.success_criteria = value.value("success_criteria", std::string{});
    plan.required_capabilities = value.value("required_capabilities", std::vector<std::string>{});
    plan.version = value.value("version", uint64_t(0));
    plan.created_at = value.value("created_at", int64_t(0));
    plan.updated_at = value.value("updated_at", int64_t(0));
    if (value.contains("derived_from_plan_id") && value["derived_from_plan_id"].is_string()) plan.derived_from_plan_id = value["derived_from_plan_id"].get<std::string>();
    if (value.contains("active_step_id") && value["active_step_id"].is_string()) plan.active_step_id = value["active_step_id"].get<std::string>();
    if (value.contains("next_action") && value["next_action"].is_string()) plan.next_action = value["next_action"].get<std::string>();
    for (const auto & item : value.value("steps", json::array())) {
        common_plan_step step;
        step.id = item.value("id", std::string{}); step.title = item.value("title", std::string{}); step.objective = item.value("objective", std::string{}); step.intended_contribution = item.value("intended_contribution", std::string{});
        step.mode = parse_step_mode(item.value("mode", std::string("final_response"))); step.status = parse_step_status(item.value("status", std::string("pending"))); step.depends_on = item.value("depends_on", std::vector<std::string>{}); step.blocked_by = item.value("blocked_by", std::vector<std::string>{}); step.required_evidence = item.value("required_evidence", std::vector<std::string>{}); step.source_memory_ids = item.value("source_memory_ids", std::vector<std::string>{}); step.optional = item.value("optional", false); step.generated_from_memory = item.value("generated_from_memory", false); step.created_at = item.value("created_at", int64_t(0)); step.updated_at = item.value("updated_at", int64_t(0));
        if (item.contains("selected_tool") && item["selected_tool"].is_string()) step.selected_tool = item["selected_tool"].get<std::string>();
        if (item.contains("result_summary") && item["result_summary"].is_string()) step.result_summary = item["result_summary"].get<std::string>();
        if (item.contains("semantic_alias") && item["semantic_alias"].is_string()) step.semantic_alias = item["semantic_alias"].get<std::string>();
        if (item.contains("tool_call") && item["tool_call"].is_object()) step.tool_call = common_plan_tool_call{item["tool_call"].value("name", std::string{}), item["tool_call"].value("arguments_json", std::string("{}"))};
        plan.steps.push_back(std::move(step));
    }
    for (const auto & item : value.value("constraints", json::array())) plan.constraints.push_back({item.value("id", std::string{}), item.value("description", std::string{}), item.value("hard", true)});
    for (const auto & item : value.value("assumptions", json::array())) { common_plan_assumption assumption; assumption.id = item.value("id", std::string{}); assumption.statement = item.value("statement", std::string{}); assumption.confidence = item.value("confidence", 0.5f); assumption.valid = item.value("valid", true); assumption.evidence_ids = item.value("evidence_ids", std::vector<std::string>{}); plan.assumptions.push_back(std::move(assumption)); }
    for (const auto & item : value.value("observations", json::array())) { common_plan_observation observation; observation.id = item.value("id", std::string{}); observation.source = item.value("source", std::string{}); observation.summary = item.value("summary", std::string{}); observation.confidence = item.value("confidence", 0.5f); observation.evidence_ids = item.value("evidence_ids", std::vector<std::string>{}); observation.created_at = item.value("created_at", int64_t(0)); plan.observations.push_back(std::move(observation)); }
    return !plan.id.empty();
}

json serialize_event(const common_plan_event & event, const common_plan_operation & operation) {
    return {{"sequence", event.sequence}, {"prior_version", event.prior_version}, {"new_version", event.new_version}, {"accepted", event.accepted}, {"reason_summary", event.reason_summary}, {"created_at", event.created_at}, {"operation", common_plan_operation_kind_name(operation.kind)}, {"plan_id", operation.plan_id}, {"expected_version", operation.expected_version}, {"step_id", operation.step_id}, {"target_id", operation.target_id}, {"value", operation.value}, {"evidence_ids", operation.evidence_ids}};
}

}

common_plan_sqlite_store::~common_plan_sqlite_store() { close(); }

bool common_plan_sqlite_store::open(const std::string & path, std::string & error) {
    close();
    if (!database_.open(path, error) || !database_.execute("PRAGMA journal_mode = WAL;", error) || !ensure_schema(error) || !load_cache(error)) { close(); return false; }
    return true;
}

void common_plan_sqlite_store::close() { cache_.close(); database_.close(); }

bool common_plan_sqlite_store::ensure_schema(std::string & error) {
    return database_.execute("CREATE TABLE IF NOT EXISTS agent_plan (id TEXT PRIMARY KEY, state_json TEXT NOT NULL); CREATE TABLE IF NOT EXISTS agent_plan_event (plan_id TEXT NOT NULL, sequence INTEGER NOT NULL, event_json TEXT NOT NULL, PRIMARY KEY(plan_id, sequence));", error);
}

bool common_plan_sqlite_store::load_cache(std::string & error) {
    if (!cache_.open("", error)) return false;
    common_sqlite_statement statement;
    if (!database_.prepare("SELECT state_json FROM agent_plan ORDER BY id;", statement, error)) return false;
    bool row = false;
    while (statement.step(row, error) && row) { common_plan_state plan; if (!deserialize_plan(reinterpret_cast<const char *>(statement.column_text(0)), plan) || !cache_.create(plan, error)) return false; }
    if (!error.empty()) return false;

    std::map<std::string, std::vector<common_plan_event>> histories;
    if (!database_.prepare("SELECT plan_id,event_json FROM agent_plan_event ORDER BY plan_id,sequence;", statement, error)) return false;
    while (statement.step(row, error) && row) {
        const std::string plan_id = reinterpret_cast<const char *>(statement.column_text(0));
        const auto value = json::parse(reinterpret_cast<const char *>(statement.column_text(1)), nullptr, false);
        if (!value.is_object()) { error = "sqlite plan event contains invalid JSON"; return false; }
        common_plan_event event;
        event.sequence = value.value("sequence", uint64_t(0));
        event.prior_version = value.value("prior_version", uint64_t(0));
        event.new_version = value.value("new_version", uint64_t(0));
        event.accepted = value.value("accepted", false);
        event.reason_summary = value.value("reason_summary", std::string{});
        event.created_at = value.value("created_at", int64_t(0));
        event.operation.plan_id = value.value("plan_id", plan_id);
        event.operation.kind = parse_operation_kind(value.value("operation", std::string{}));
        event.operation.expected_version = value.value("expected_version", uint64_t(0));
        if (value.contains("step_id") && value["step_id"].is_string()) event.operation.step_id = value["step_id"].get<std::string>();
        if (value.contains("target_id") && value["target_id"].is_string()) event.operation.target_id = value["target_id"].get<std::string>();
        if (value.contains("value") && value["value"].is_string()) event.operation.value = value["value"].get<std::string>();
        event.operation.reason_summary = event.reason_summary;
        event.operation.evidence_ids = value.value("evidence_ids", std::vector<std::string>{});
        histories[plan_id].push_back(std::move(event));
    }
    if (!error.empty()) return false;
    for (auto & history : histories) if (!cache_.restore_history(history.first, std::move(history.second), error)) return false;
    return error.empty();
}

bool common_plan_sqlite_store::persist_plan(const common_plan_state & plan, std::string & error) {
    common_sqlite_statement statement;
    if (!database_.prepare("INSERT INTO agent_plan(id,state_json) VALUES(?,?) ON CONFLICT(id) DO UPDATE SET state_json=excluded.state_json;", statement, error) || !statement.bind_text(1, plan.id, error) || !statement.bind_text(2, serialize_plan(plan).dump(), error)) return false;
    bool row = false;
    return statement.step(row, error);
}

bool common_plan_sqlite_store::persist_event(const std::string & plan_id, const common_plan_event & event, const common_plan_operation & operation, std::string & error) {
    common_sqlite_statement statement;
    if (!database_.prepare("INSERT OR REPLACE INTO agent_plan_event(plan_id,sequence,event_json) VALUES(?,?,?);", statement, error) || !statement.bind_text(1, plan_id, error) || !statement.bind_int64(2, static_cast<int64_t>(event.sequence), error) || !statement.bind_text(3, serialize_event(event, operation).dump(), error)) return false;
    bool row = false;
    return statement.step(row, error);
}

bool common_plan_sqlite_store::create(const common_plan_state & plan, std::string & error) {
    if (!database_.is_open()) { error = "plan store is not open"; return false; }
    common_sqlite_transaction transaction(database_);
    if (!transaction.begin(error) || !persist_plan(plan, error) || !transaction.commit(error)) return false;
    return cache_.create(plan, error);
}

std::optional<common_plan_state> common_plan_sqlite_store::get(const std::string & plan_id, std::string & error) { return cache_.get(plan_id, error); }
std::vector<common_plan_state> common_plan_sqlite_store::list(std::string & error) { return cache_.list(error); }
std::vector<common_plan_event> common_plan_sqlite_store::history(const std::string & plan_id, std::string & error) { return cache_.history(plan_id, error); }

bool common_plan_sqlite_store::apply(const common_plan_operation & operation, common_plan_state & updated_plan, std::string & error) {
    if (!database_.is_open()) { error = "plan store is not open"; return false; }
    if (!cache_.apply(operation, updated_plan, error)) return false;
    const auto events = cache_.history(operation.plan_id, error);
    if (events.empty()) return false;
    common_sqlite_transaction transaction(database_);
    if (!transaction.begin(error) || !persist_plan(updated_plan, error) || !persist_event(operation.plan_id, events.back(), operation, error) || !transaction.commit(error)) return false;
    return true;
}

bool common_plan_sqlite_store::erase(const std::string & plan_id, std::string & error) {
    if (!database_.is_open()) { error = "plan store is not open"; return false; }
    common_sqlite_transaction transaction(database_);
    common_sqlite_statement statement;
    if (!transaction.begin(error) || !database_.prepare("DELETE FROM agent_plan_event WHERE plan_id = ?;", statement, error) || !statement.bind_text(1, plan_id, error)) return false;
    bool row = false;
    if (!statement.step(row, error) || !database_.prepare("DELETE FROM agent_plan WHERE id = ?;", statement, error) || !statement.bind_text(1, plan_id, error) || !statement.step(row, error) || !transaction.commit(error)) return false;
    return cache_.erase(plan_id, error);
}
