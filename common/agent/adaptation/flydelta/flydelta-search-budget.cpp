#include "agent/adaptation/flydelta/flydelta-search-budget.h"
#include "hash/hash.h"

#include <map>
#include <nlohmann/json.hpp>

namespace {
using json = nlohmann::ordered_json;
std::string digest(const std::string & text) { return hash_sha256_hex(text.data(), text.size()); }

common_learning_lifecycle_record event(const common_flydelta_experiment_job & job,
        const std::string & id, const json & payload, bool completed) {
    common_learning_lifecycle_record record;
    record.event_id = id;
    record.subject_id = job.id;
    record.idempotency_key = id;
    record.kind = common_learning_lifecycle_kind::flydelta_experiment;
    record.status = completed ? common_learning_lifecycle_status::succeeded : common_learning_lifecycle_status::running;
    record.source_id = "flydelta-search-execution";
    record.namespace_id = job.seed.scope.namespace_id;
    record.project_id = job.seed.scope.project_id;
    record.session_id = job.seed.scope.session_id;
    record.created_at = job.id;
    record.payload_json = payload.dump();
    record.content_hash = "sha256:" + digest(record.payload_json);
    return record;
}

struct ledger {
    std::set<std::string> logical;
    std::map<std::string, json> starts;
    std::map<std::string, json> completions;
};

bool read(const common_learning_lifecycle_store & store, const common_flydelta_experiment_job & job,
        size_t base, size_t limit, ledger & state, common_flydelta_search_consumption & consumption,
        std::string & error) {
    consumption = {};
    if (limit == 0 || limit > 65536 || base > limit) { error = "FlyDelta logical budget is invalid"; return false; }
    std::vector<common_learning_lifecycle_record> records;
    uint64_t watermark = 0;
    if (!store.list_scope_with_watermark(job.seed.scope.namespace_id, job.seed.scope.project_id,
            job.seed.scope.session_id, records, watermark, error)) return false;
    try {
        for (const auto & record : records) {
            if (record.source_id != "flydelta-search-execution" || record.subject_id != job.id) continue;
            if (record.content_hash != "sha256:" + digest(record.payload_json)) {
                error = "FlyDelta execution ledger content hash differs"; return false;
            }
            const auto value = json::parse(record.payload_json);
            if (value.at("schema_version") != 1) { error = "FlyDelta execution ledger version is invalid"; return false; }
            if (value.at("kind") == "flydelta_search_execution_start") {
                if (value.at("base_used") != base || value.at("limit") != limit || value.at("job") != job.id ||
                        value.at("job_fingerprint") != digest(common_flydelta_experiment_job_to_json(job))) {
                    error = "FlyDelta execution reservation budget differs"; return false;
                }
                const auto ids = value.at("logical_ids").get<std::vector<std::string>>();
                if (ids.empty() || ids.size() > 128) { error = "FlyDelta execution row count is invalid"; return false; }
                state.starts.emplace(record.event_id, value);
                for (const auto & id : ids) {
                    if (id.empty() || id.size() > 512) { error = "FlyDelta logical arm identity is invalid"; return false; }
                    state.logical.insert(id);
                }
                consumption.physical_rows_started += ids.size();
            } else if (value.at("kind") == "flydelta_search_execution_finish") {
                state.completions.emplace(value.at("attempt_ref").get<std::string>(), value);
            } else { error = "FlyDelta execution ledger record kind is invalid"; return false; }
        }
        for (const auto & pair : state.completions) {
            const auto start = state.starts.find(pair.first);
            if (start == state.starts.end()) { error = "FlyDelta execution completion has no reservation"; return false; }
            const auto rows = pair.second.at("completed_rows").get<size_t>();
            if (rows > start->second.at("logical_ids").size()) { error = "FlyDelta execution completion exceeds its rows"; return false; }
            consumption.physical_rows_completed += rows;
        }
        if (state.logical.size() > limit - base) { error = "FlyDelta logical execution budget exceeded"; return false; }
        consumption.logical_used = base + state.logical.size();
        consumption.physical_rows_unresolved = consumption.physical_rows_started - consumption.physical_rows_completed;
        return true;
    } catch (const std::exception & exception) {
        error = std::string("FlyDelta execution ledger is malformed: ") + exception.what(); return false;
    }
}
}

bool common_flydelta_load_search_consumption(const common_learning_lifecycle_store & store,
        const common_flydelta_experiment_job & job, size_t base_used, size_t limit,
        common_flydelta_search_consumption & consumption, std::string & error) {
    ledger state;
    return read(store, job, base_used, limit, state, consumption, error);
}

bool common_flydelta_begin_search_execution(common_learning_lifecycle_store & store,
        const common_flydelta_experiment_job & job, size_t base_used, size_t limit,
        const std::vector<std::string> & logical_ids, const std::string & execution_group,
        size_t max_attempts, std::string & attempt_ref,
        common_flydelta_search_consumption & consumption, std::string & error) {
    attempt_ref.clear();
    ledger state;
    if (!read(store, job, base_used, limit, state, consumption, error)) return false;
    if (max_attempts == 0 || max_attempts > 16 || execution_group.empty() || execution_group.size() > 512 ||
            logical_ids.empty() || logical_ids.size() > 128) {
        error = "FlyDelta execution reservation parameters are invalid"; return false;
    }
    const std::set<std::string> unique(logical_ids.begin(), logical_ids.end());
    if (unique.size() != logical_ids.size()) { error = "FlyDelta execution contains duplicate logical rows"; return false; }
    auto merged = state.logical;
    for (const auto & id : logical_ids) {
        if (id.empty() || id.size() > 512) { error = "FlyDelta logical row ID is invalid"; return false; }
        merged.insert(id);
    }
    if (merged.size() > limit - base_used) { error = "FlyDelta logical search budget exhausted"; return false; }
    size_t attempts = 0;
    for (const auto & pair : state.starts) {
        if (pair.second.at("execution_group") == execution_group) {
            if (pair.second.at("logical_ids") != json(logical_ids) || pair.second.at("max_attempts") != max_attempts) {
                error = "FlyDelta execution group identity changed"; return false;
            }
            ++attempts;
        }
    }
    if (attempts >= max_attempts) { error = "FlyDelta physical execution retry limit exhausted"; return false; }
    if (consumption.physical_rows_started > limit * max_attempts ||
            logical_ids.size() > limit * max_attempts - consumption.physical_rows_started) {
        error = "FlyDelta physical search consumption limit exhausted"; return false;
    }
    attempt_ref = "flydelta://execution/" + digest(job.id + "\n" + execution_group) + "/" + std::to_string(attempts + 1);
    const json payload = {{"kind", "flydelta_search_execution_start"}, {"schema_version", 1},
        {"job", job.id}, {"execution_group", execution_group}, {"logical_ids", logical_ids},
        {"job_fingerprint", digest(common_flydelta_experiment_job_to_json(job))},
        {"base_used", base_used}, {"limit", limit}, {"attempt", attempts + 1}, {"max_attempts", max_attempts}};
    if (!store.append(event(job, attempt_ref, payload, false), error)) return false;
    consumption.logical_used = base_used + merged.size();
    consumption.physical_rows_started += logical_ids.size();
    consumption.physical_rows_unresolved += logical_ids.size();
    return true;
}

bool common_flydelta_finish_search_execution(common_learning_lifecycle_store & store,
        const common_flydelta_experiment_job & job, const std::string & attempt_ref,
        size_t completed_rows, bool succeeded, std::string & error) {
    if (attempt_ref.empty() || attempt_ref.size() > 400 || completed_rows > 128) {
        error = "FlyDelta physical completion is invalid"; return false;
    }
    std::vector<common_learning_lifecycle_record> records;
    uint64_t watermark = 0;
    if (!store.list_scope_with_watermark(job.seed.scope.namespace_id, job.seed.scope.project_id,
            job.seed.scope.session_id, records, watermark, error)) return false;
    bool found = false;
    for (const auto & record : records) {
        if (record.event_id != attempt_ref || record.subject_id != job.id ||
                record.source_id != "flydelta-search-execution") continue;
        const auto value = json::parse(record.payload_json, nullptr, false);
        if (!value.is_object() || value.value("kind", "") != "flydelta_search_execution_start" ||
                !value.contains("logical_ids") || !value.at("logical_ids").is_array() ||
                completed_rows > value.at("logical_ids").size() ||
                record.content_hash != "sha256:" + digest(record.payload_json)) {
            error = "FlyDelta completion reservation is invalid"; return false;
        }
        found = true;
        break;
    }
    if (!found) { error = "FlyDelta completion has no authorized start reservation"; return false; }
    return store.append(event(job, attempt_ref + ":finish", {{"kind", "flydelta_search_execution_finish"},
        {"schema_version", 1}, {"attempt_ref", attempt_ref}, {"completed_rows", completed_rows},
        {"succeeded", succeeded}, {"token_consumption_known", false}}, true), error);
}
