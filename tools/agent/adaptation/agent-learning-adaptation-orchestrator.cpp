#include "agent-learning-adaptation-orchestrator.h"

#include "hash/hash.h"

#include <chrono>
#include <fstream>
#include <nlohmann/json.hpp>
#include <sstream>

using json = nlohmann::ordered_json;

namespace {

std::string now_iso8601() {
    const auto now = std::chrono::system_clock::now();
    const auto value = std::chrono::system_clock::to_time_t(now);
    char buffer[32]{};
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&value));
    return buffer;
}

bool append_lifecycle(
        common_learning_lifecycle_store & store,
        const std::string & subject_id,
        const common_learning_lifecycle_kind kind,
        const common_learning_lifecycle_status status,
        const std::string & payload_json,
        const std::string & source_id,
        const size_t max_payload_bytes,
        size_t & count,
        std::string & error) {
    const auto suffix = std::string(common_learning_lifecycle_kind_name(kind)) + ":" +
        common_learning_lifecycle_status_name(status);
    common_learning_lifecycle_record record;
    record.event_id = "learning://lifecycle/" +
        hash_sha256_hex((subject_id + ":" + suffix).data(), subject_id.size() + 1 + suffix.size()).substr(0, 24);
    record.subject_id = subject_id;
    record.kind = kind;
    record.status = status;
    record.idempotency_key = record.event_id;
    record.source_id = source_id;
    record.content_hash = "sha256:" + hash_sha256_hex(payload_json.data(), payload_json.size());
    record.created_at = now_iso8601();
    record.payload_json = payload_json;
    if (!common_learning_lifecycle_validate(record, max_payload_bytes, error)) return false;
    if (!store.append(record, error)) return false;
    ++count;
    return true;
}

bool read_text(const std::filesystem::path & path, std::string & value, std::string & error) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        error = "could not read adaptation worker result: " + path.string();
        return false;
    }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    if (!input.good() && !input.eof()) {
        error = "could not read adaptation worker result: " + path.string();
        return false;
    }
    value = buffer.str();
    return true;
}

std::string queue_key(const std::string & job_id) {
    return "job-" + hash_sha256_hex(job_id.data(), job_id.size()).substr(0, 32);
}

std::string candidate_payload(const common_training_candidate & candidate) {
    return json{
        {"id", candidate.id},
        {"status", common_training_candidate_status_name(candidate.status)},
        {"transaction_ids", candidate.transaction_ids},
        {"hypothesis", candidate.hypothesis},
        {"approved_prompt", candidate.approved_prompt},
        {"approved_target", candidate.approved_target},
        {"learning_domain", candidate.learning_domain},
        {"tool_family", candidate.tool_family},
        {"provider_kind", candidate.provider_kind},
    }.dump();
}

std::string corpus_payload(const common_learning_corpus_revision & corpus) {
    return json{
        {"id", corpus.id},
        {"bundle_hash", corpus.bundle_hash},
        {"candidate_ids", corpus.candidate_ids},
        {"replay_candidate_ids", corpus.replay_candidate_ids},
        {"view", {
            {"learning_domain", corpus.view.learning_domain},
            {"tool_family", corpus.view.tool_family},
            {"provider_kinds", std::vector<std::string>(corpus.view.provider_kinds.begin(), corpus.view.provider_kinds.end())},
        }},
    }.dump();
}

} // namespace

bool agent_learning_run_adaptation(
        const common_training_candidate & candidate,
        const agent_learning_adaptation_orchestrator_config & config,
        agent_learning_adaptation_orchestrator_result & result,
        std::string & error) {
    error.clear();
    result = {};
    if (config.queue_root.empty()) {
        error = "adaptation orchestrator requires a queue root";
        return false;
    }
    common_training_candidate_policy candidate_policy;
    if (!common_training_candidate_qualifies(candidate, candidate_policy, error)) return false;

    const auto transactions = config.transaction_store.list(error);
    if (!error.empty()) return false;
    for (const auto & transaction_id : candidate.transaction_ids) {
        bool found = false;
        for (const auto & transaction : transactions) {
            if (transaction.id == transaction_id) { found = true; break; }
        }
        if (!found) {
            error = "training candidate references a transaction not present in the ledger: " + transaction_id;
            return false;
        }
    }

    if (!append_lifecycle(config.lifecycle_store, candidate.id,
            common_learning_lifecycle_kind::candidate,
            common_learning_lifecycle_status::approved,
            candidate_payload(candidate), candidate.id,
            config.max_lifecycle_payload_bytes, result.lifecycle_records, error)) return false;

    common_learning_corpus_policy corpus_policy;
    corpus_policy.max_candidates = 1;
    corpus_policy.max_bytes = 64 * 1024;
    corpus_policy.redaction_policy_id = candidate.redaction_policy_id;
    if (!common_learning_build_corpus({candidate}, corpus_policy, result.corpus, error)) return false;
    if (!append_lifecycle(config.lifecycle_store, result.corpus.id,
            common_learning_lifecycle_kind::corpus_revision,
            common_learning_lifecycle_status::succeeded,
            corpus_payload(result.corpus), candidate.id,
            config.max_lifecycle_payload_bytes, result.lifecycle_records, error)) return false;

    if (!common_learning_make_training_job(result.corpus,
            "train:" + config.serving_model_fingerprint,
            config.code_revision, config.trainer_version, result.job, error)) return false;
    std::string job_json;
    if (!common_learning_training_job_to_json(result.job, job_json, error)) return false;
    if (!append_lifecycle(config.lifecycle_store, result.job.id,
            common_learning_lifecycle_kind::training_job,
            common_learning_lifecycle_status::queued,
            job_json, candidate.id, config.max_lifecycle_payload_bytes,
            result.lifecycle_records, error)) return false;
    if (!agent_learning_enqueue_job(config.queue_root, result.job, result.corpus, error)) return false;
    if (!append_lifecycle(config.lifecycle_store, result.job.id,
            common_learning_lifecycle_kind::training_job,
            common_learning_lifecycle_status::running,
            job_json, candidate.id, config.max_lifecycle_payload_bytes,
            result.lifecycle_records, error)) return false;

    if (!agent_learning_worker_run_once(config.queue_root, config.worker_limits,
            result.worker_report, error)) {
        const auto worker_error = error;
        error.clear();
        std::string lifecycle_error;
        append_lifecycle(config.lifecycle_store, result.job.id,
            common_learning_lifecycle_kind::training_job,
            common_learning_lifecycle_status::failed,
            job_json, candidate.id, config.max_lifecycle_payload_bytes,
            result.lifecycle_records, lifecycle_error);
        error = worker_error;
        if (!lifecycle_error.empty()) error += "; lifecycle failure: " + lifecycle_error;
        return false;
    }
    if (result.worker_report.state != agent_learning_worker_job_state::succeeded) {
        error = "adaptation worker did not succeed: " + result.worker_report.safe_summary;
        std::string lifecycle_error;
        append_lifecycle(config.lifecycle_store, result.job.id,
            common_learning_lifecycle_kind::training_job,
            result.worker_report.state == agent_learning_worker_job_state::cancelled
                ? common_learning_lifecycle_status::cancelled
                : common_learning_lifecycle_status::failed,
            job_json, candidate.id, config.max_lifecycle_payload_bytes,
            result.lifecycle_records, lifecycle_error);
        if (!lifecycle_error.empty()) error += "; lifecycle failure: " + lifecycle_error;
        return false;
    }
    const auto succeeded_directory = config.queue_root / "succeeded" / queue_key(result.job.id);
    std::string result_json;
    if (!read_text(succeeded_directory / "result.json", result_json, error) ||
            !common_learning_training_result_from_json(result_json, result.training_result, error) ||
            !common_learning_validate_training_result(result.job, result.training_result, error)) return false;
    if (!append_lifecycle(config.lifecycle_store, result.job.id,
            common_learning_lifecycle_kind::training_job,
            common_learning_lifecycle_status::succeeded,
            job_json, candidate.id, config.max_lifecycle_payload_bytes,
            result.lifecycle_records, error)) return false;
    if (!append_lifecycle(config.lifecycle_store, result.training_result.adapter_id,
            common_learning_lifecycle_kind::training_result,
            common_learning_lifecycle_status::succeeded,
            result_json, result.job.id, config.max_lifecycle_payload_bytes,
            result.lifecycle_records, error)) return false;

    if (!config.evaluator) {
        error = "adaptation orchestrator requires a host evaluator";
        return false;
    }
    if (!config.evaluator(result.job, result.training_result, result.evaluation, error) ||
            !common_learning_validate_evaluation_report(result.evaluation, error)) return false;
    if (!common_learning_make_evaluated_adapter_manifest(
            result.job, result.training_result, result.evaluation,
            config.base_architecture, config.serving_model_fingerprint,
            config.tokenizer_fingerprint, config.chat_template_fingerprint,
            result.manifest, error)) return false;

    if (!common_learning_verify_adapter_artifact(
            result.manifest, succeeded_directory / "artifacts",
            config.max_artifact_bytes, result.verified_artifact_path, error)) return false;
    common_learning_adapter_registry registry;
    if (!registry.admit(result.manifest, error) ||
            !registry.stage_canary(result.manifest.id, result.evaluation, error) ||
            !append_lifecycle(config.lifecycle_store, result.manifest.id,
                common_learning_lifecycle_kind::adapter,
                common_learning_lifecycle_status::canary,
                json{{"id", result.manifest.id}, {"evaluation_revision", result.evaluation.revision_id}}.dump(),
                result.training_result.adapter_id, config.max_lifecycle_payload_bytes,
                result.lifecycle_records, error) ||
            !registry.activate(result.manifest.id, error)) return false;
    if (!append_lifecycle(config.lifecycle_store, result.manifest.id,
            common_learning_lifecycle_kind::adapter,
            common_learning_lifecycle_status::active,
            json{{"id", result.manifest.id}, {"evaluation_revision", result.evaluation.revision_id}}.dump(),
            result.training_result.adapter_id, config.max_lifecycle_payload_bytes,
            result.lifecycle_records, error)) return false;

    result.profile.id = "profile:adaptation-smoke";
    result.profile.base_model_id = "qwen-small";
    result.profile.base_model_fingerprint = config.serving_model_fingerprint;
    result.profile.tokenizer_fingerprint = config.tokenizer_fingerprint;
    result.profile.chat_template_fingerprint = config.chat_template_fingerprint;
    result.profile.context_size_tokens = 4096;
    result.profile.adapters.push_back({result.manifest.id, 1.0});
    std::vector<common_learning_adapter_manifest> overlays;
    if (!registry.resolve_active_overlays(result.profile, overlays, error) || overlays.size() != 1) {
        if (error.empty()) error = "active adaptation overlay could not be resolved";
        return false;
    }
    return true;
}
