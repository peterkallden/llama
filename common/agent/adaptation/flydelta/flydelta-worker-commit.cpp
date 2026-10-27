#include "agent/adaptation/flydelta/flydelta-worker-commit.h"
#include "hash/hash.h"

#include <nlohmann/json.hpp>

namespace {
using json = nlohmann::ordered_json;
std::string digest(const std::string & text) { return hash_sha256_hex(text.data(), text.size()); }

common_learning_lifecycle_record make_record(const common_flydelta_experiment_job & job,
        const json & payload, const std::string & suffix) {
    common_learning_lifecycle_record record;
    record.event_id = "flydelta://worker-commit/" + digest(job.id) + suffix;
    record.subject_id = job.id;
    record.kind = common_learning_lifecycle_kind::flydelta_result;
    record.status = suffix.empty() ? common_learning_lifecycle_status::running : common_learning_lifecycle_status::succeeded;
    record.idempotency_key = record.event_id;
    record.source_id = "flydelta-worker-commit";
    record.namespace_id = job.seed.scope.namespace_id;
    record.project_id = job.seed.scope.project_id;
    record.session_id = job.seed.scope.session_id;
    record.created_at = job.id;
    record.payload_json = payload.dump();
    record.content_hash = "sha256:" + digest(record.payload_json);
    return record;
}
}

bool common_flydelta_append_worker_commit_intent(common_learning_lifecycle_store & store,
        const common_flydelta_experiment_job & job,
        const common_flydelta_experiment_worker_report & report, std::string & error,
        const common_flydelta_experiment_job * scheduling_parent) {
    if (report.state != common_flydelta_experiment_queue_state::succeeded || report.job_id != job.id ||
            !common_flydelta_experiment_job_validate(job, 128, error)) {
        if (error.empty()) error = "FlyDelta commit intent requires a validated successful result";
        return false;
    }
    json payload = {{"kind", "flydelta_worker_commit"}, {"schema_version", 1},
        {"job", json::parse(common_flydelta_experiment_job_to_json(job))},
        {"scheduling_parent", json::parse(common_flydelta_experiment_job_to_json(scheduling_parent ? *scheduling_parent : job))},
        {"safe_summary", report.safe_summary}, {"trace_json", report.trace_json},
        {"has_evaluation_report", report.has_evaluation_report},
        {"counterfactual_reports", json::array()}, {"evaluation_fixture_results", json::array()},
        {"has_next_action", report.has_next_action},
        {"next_action", common_flydelta_next_action_name(report.next_action)},
        {"bootstrap_state_ref", report.bootstrap_zoom_state_ref}, {"search_state_ref", report.search_state_ref},
        {"representation_state_ref", report.representation_augmentation_state_ref},
        {"graft_direction_ref", report.graft_direction_ref}, {"graft_direction_refs", report.graft_direction_refs}};
    for (const auto & item : report.counterfactual_reports)
        payload["counterfactual_reports"].push_back(json::parse(common_flydelta_counterfactual_report_to_json(item)));
    for (const auto & item : report.evaluation_fixture_results)
        payload["evaluation_fixture_results"].push_back(json::parse(common_flydelta_evaluation_fixture_result_to_json(item)));
    if (report.has_evaluation_report)
        payload["evaluation_report"] = json::parse(common_flydelta_evaluation_report_to_json(report.evaluation_report));
    return store.append(make_record(job, payload, ""), error);
}

bool common_flydelta_resolve_worker_commit_parent(const common_learning_lifecycle_store & store,
        const common_flydelta_experiment_job & job,
        common_flydelta_experiment_job & parent, std::string & error) {
    const auto records = store.list(error);
    if (!error.empty()) return false;
    const auto identity = "flydelta://worker-commit/" + digest(job.id);
    for (const auto & record : records) {
        if (record.event_id != identity) continue;
        const auto payload = json::parse(record.payload_json, nullptr, false);
        if (!payload.is_object() || record.content_hash != "sha256:" + digest(record.payload_json) ||
                record.subject_id != job.id || record.namespace_id != job.seed.scope.namespace_id ||
                record.project_id != job.seed.scope.project_id || record.session_id != job.seed.scope.session_id ||
                !payload.contains("job") || payload.at("job").dump() != common_flydelta_experiment_job_to_json(job) ||
                !payload.contains("scheduling_parent") || !common_flydelta_experiment_job_from_json(
                    payload.at("scheduling_parent").dump(), parent, error)) {
            if (error.empty()) error = "FlyDelta worker commit parent is invalid";
            return false;
        }
        auto original = job;
        original.teaching_material_group_ref = parent.teaching_material_group_ref;
        if (common_flydelta_experiment_job_to_json(original) != common_flydelta_experiment_job_to_json(parent)) {
            error = "FlyDelta scheduling parent may only add its authorized teaching group"; return false;
        }
        return true;
    }
    error = "FlyDelta scheduling parent has no durable commit intent";
    return false;
}

bool common_flydelta_recover_worker_commits(common_learning_lifecycle_store & store,
        const std::filesystem::path & queue_root,
        const common_flydelta_experiment_queue_limits & limits,
        const common_flydelta_worker_commit_callback & persist,
        const common_flydelta_worker_commit_callback & schedule, std::string & error) {
    const auto owner = common_flydelta_acquire_queue_owner(queue_root, error);
    if (!owner) return false;
    const auto records = store.list(error);
    if (!error.empty()) return false;
    for (const auto & record : records) {
        if (record.source_id != "flydelta-worker-commit" ||
                record.status != common_learning_lifecycle_status::running) continue;
        bool acknowledged = false;
        if (!store.contains_idempotency(record.event_id + ":ack", acknowledged, error)) return false;
        if (acknowledged) {
            bool valid_ack = false;
            for (const auto & ack : records) {
                if (ack.idempotency_key != record.event_id + ":ack") continue;
                const auto payload = json::parse(ack.payload_json, nullptr, false);
                valid_ack = ack.source_id == record.source_id && ack.subject_id == record.subject_id &&
                    ack.status == common_learning_lifecycle_status::succeeded &&
                    ack.namespace_id == record.namespace_id && ack.project_id == record.project_id &&
                    ack.session_id == record.session_id && ack.content_hash == "sha256:" + digest(ack.payload_json) &&
                    payload.is_object() && payload.contains("intent_hash") && payload.at("intent_hash") == record.content_hash;
                break;
            }
            if (!valid_ack) { error = "FlyDelta worker commit acknowledgement is inconsistent"; return false; }
            continue;
        }
        try {
            const auto payload = json::parse(record.payload_json);
            common_flydelta_experiment_job job;
            if (record.content_hash != "sha256:" + digest(record.payload_json) ||
                    payload.at("kind") != "flydelta_worker_commit" || payload.at("schema_version") != 1 ||
                    !common_flydelta_experiment_job_from_json(payload.at("job").dump(), job, error)) {
                if (error.empty()) error = "FlyDelta worker commit content is invalid";
                return false;
            }
            if (record.subject_id != job.id || record.event_id != make_record(job, payload, "").event_id ||
                    record.namespace_id != job.seed.scope.namespace_id || record.project_id != job.seed.scope.project_id ||
                    record.session_id != job.seed.scope.session_id) {
                error = "FlyDelta worker commit identity or scope differs"; return false;
            }
            common_flydelta_experiment_worker_report report;
            report.job_id = job.id;
            report.completed_job = job;
            report.state = common_flydelta_experiment_queue_state::succeeded;
            report.safe_summary = payload.at("safe_summary").get<std::string>();
            report.trace_json = payload.at("trace_json").get<std::string>();
            report.has_evaluation_report = payload.at("has_evaluation_report").get<bool>();
            if (report.has_evaluation_report && !common_flydelta_evaluation_report_from_json(
                    payload.at("evaluation_report").dump(), report.evaluation_report, error)) return false;
            for (const auto & item : payload.at("counterfactual_reports")) {
                common_flydelta_counterfactual_report value;
                if (!common_flydelta_counterfactual_report_from_json(item.dump(), value, error)) return false;
                report.counterfactual_reports.push_back(std::move(value));
            }
            for (const auto & item : payload.at("evaluation_fixture_results")) {
                common_flydelta_evaluation_fixture_result value;
                if (!common_flydelta_evaluation_fixture_result_from_json(item.dump(), value, error)) return false;
                report.evaluation_fixture_results.push_back(std::move(value));
            }
            report.has_next_action = payload.at("has_next_action").get<bool>();
            bool valid_action = false;
            for (int action = 0; action <= static_cast<int>(common_flydelta_next_action::allow_tfo_lite); ++action) {
                const auto candidate = static_cast<common_flydelta_next_action>(action);
                if (payload.at("next_action") == common_flydelta_next_action_name(candidate)) {
                    report.next_action = candidate; valid_action = true; break;
                }
            }
            if (!valid_action) { error = "FlyDelta commit action is invalid"; return false; }
            report.bootstrap_zoom_state_ref = payload.at("bootstrap_state_ref").get<std::string>();
            report.search_state_ref = payload.at("search_state_ref").get<std::string>();
            report.representation_augmentation_state_ref = payload.at("representation_state_ref").get<std::string>();
            report.graft_direction_ref = payload.at("graft_direction_ref").get<std::string>();
            report.graft_direction_refs = payload.at("graft_direction_refs").get<std::vector<std::string>>();
            if (!persist || !persist(job, report, error)) return false;
            if (!common_flydelta_experiment_queue_acknowledge_recovered_result(
                    queue_root, job, report.safe_summary, limits, error)) return false;
            if (report.has_next_action && report.next_action != common_flydelta_next_action::stop &&
                    report.next_action != common_flydelta_next_action::retain &&
                    (!schedule || !schedule(job, report, error))) return false;
            if (!store.append(make_record(job, {{"kind", "flydelta_worker_commit_ack"},
                    {"intent_hash", record.content_hash}}, ":ack"), error)) return false;
        } catch (const std::exception & exception) {
            error = std::string("FlyDelta worker commit recovery failed: ") + exception.what();
            return false;
        }
    }
    return true;
}
