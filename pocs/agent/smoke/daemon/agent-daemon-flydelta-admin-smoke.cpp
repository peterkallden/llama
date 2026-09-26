#include "tools/agent/daemon/agent-daemon-adapter.h"
#include "tools/agent/daemon/agent-daemon-dispatcher.h"

#include "agent/adaptation/flydelta/flydelta-candidate-lifecycle.h"
#include "agent/adaptation/flydelta/flydelta-evaluator.h"
#include "agent/adaptation/flydelta/flydelta-model-adapter.h"
#include "agent/adaptation/flydelta/flydelta-queue.h"
#include "agent/adaptation/flydelta/flydelta-sideband-review-store.h"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unistd.h>

using json = nlohmann::ordered_json;

namespace {

constexpr const char * kCandidateId = "flydelta://sideband/admin-smoke-v1";
constexpr const char * kArtifactPath = "artifact://sideband/admin-smoke-v1";
constexpr const char * kSuiteRef = "suite://admin-smoke-v1";
constexpr const char * kEvaluationRevision = "evaluation:admin-smoke-v1";
constexpr const char * kVerifierRevision = "verifier:admin-smoke-v1";
constexpr const char * kModelProfileFingerprint = "sha256:model-profile-admin-smoke";
constexpr const char * kTokenizerFingerprint = "sha256:tokenizer-admin-smoke";
constexpr const char * kTemplateFingerprint = "sha256:template-admin-smoke";
constexpr const char * kExecutionContextFingerprint = "sha256:execution-admin-smoke";

struct smoke_trace_state {
    std::mutex mutex;
    std::string worker_trace_json;
};

common_flydelta_sideband_manifest make_manifest() {
    common_flydelta_sideband_manifest manifest;
    manifest.id = kCandidateId;
    manifest.status = common_flydelta_sideband_status::experimental;
    manifest.artifact_path = kArtifactPath;
    manifest.artifact_hash = "sha256:artifact-admin-smoke";
    manifest.compatibility.base_model_fingerprint = kModelProfileFingerprint;
    manifest.compatibility.tokenizer_fingerprint = kTokenizerFingerprint;
    manifest.compatibility.template_fingerprint = kTemplateFingerprint;
    manifest.compatibility.architecture = "model-free-test-host";
    manifest.compatibility.inference_layout_revision = "layout:admin-smoke-v1";
    manifest.model_n_embd = 4;
    manifest.model_n_layers = 3;
    manifest.il_start = 1;
    manifest.il_end = 2;
    return manifest;
}

common_flydelta_counterfactual_report make_counterfactual(
        const common_flydelta_experiment_job & job, size_t index) {
    common_flydelta_counterfactual_report report;
    report.experiment_id = job.id + ":fixture:" + std::to_string(index);
    report.fixture_id = "fixture:admin-smoke:" + std::to_string(index);
    report.candidate_id = job.evaluation_candidate_id;
    report.baseline_profile_id = "profile:baseline";
    report.candidate_profile_id = "profile:candidate";
    report.baseline.executed = true;
    report.baseline.verifier_known = true;
    report.baseline.passed = false;
    report.baseline.quality = 0.25f;
    report.baseline.overlay_applied = false;
    report.baseline.evidence_ref = report.fixture_id + ":baseline";
    report.candidate.executed = true;
    report.candidate.verifier_known = true;
    report.candidate.passed = true;
    report.candidate.quality = 0.75f;
    report.candidate.overlay_applied = true;
    report.candidate.intervention_count = 1;
    report.candidate.evidence_ref = report.fixture_id + ":candidate";
    report.outcome = common_flydelta_counterfactual_outcome::helped;
    report.quality_delta = 0.5f;
    return report;
}

bool run_model_free_evaluation(
        const common_flydelta_experiment_job & job,
        common_flydelta_evaluation_report & report,
        std::vector<common_flydelta_evaluation_fixture_result> & fixtures,
        std::string & error) {
    report = {};
    report.revision_id = job.evaluation_revision;
    report.candidate_id = job.evaluation_candidate_id;
    report.baseline_profile_id = "profile:baseline";
    report.candidate_profile_id = "profile:candidate";
    report.test_suite_revision = job.evaluation_suite_ref;
    report.intended_behavior_passed = true;
    report.retention_passed = true;
    report.agent_regression_passed = true;
    report.evaluated_turns = 8;
    report.baseline_successes = 0;
    report.candidate_successes = 8;
    report.candidate_interventions = 8;
    report.false_interventions = 0;
    report.status = "passed";

    fixtures.clear();
    const common_flydelta_evaluation_suite_kind suite_kinds[] = {
        common_flydelta_evaluation_suite_kind::intended,
        common_flydelta_evaluation_suite_kind::holdout,
        common_flydelta_evaluation_suite_kind::retention,
        common_flydelta_evaluation_suite_kind::agent_regression,
    };
    for (size_t index = 0; index < 8; ++index) {
        auto counterfactual = make_counterfactual(job, index);
        common_flydelta_evaluation_fixture_result fixture;
        fixture.candidate_id = job.evaluation_candidate_id;
        fixture.suite_kind = suite_kinds[index % 4];
        fixture.fixture_ref = counterfactual.fixture_id;
        fixture.verifier_revision = job.evaluation_revision;
        fixture.baseline_known = true;
        fixture.baseline_passed = false;
        fixture.candidate_known = true;
        fixture.candidate_passed = true;
        fixture.passed = true;
        fixture.report_ref = counterfactual.experiment_id;
        fixture.counterfactual = std::move(counterfactual);
        if (!common_flydelta_evaluation_fixture_result_validate(fixture, error)) return false;
        fixtures.push_back(std::move(fixture));
    }
    return common_flydelta_evaluation_report_validate(report, error);
}

bool persist_worker_report(
        const std::shared_ptr<common_learning_lifecycle_store> & lifecycle,
        const std::shared_ptr<smoke_trace_state> & trace_state,
        const common_flydelta_experiment_job & job,
        const common_flydelta_experiment_worker_report & report,
        std::string & error) {
    error.clear();
    {
        std::lock_guard<std::mutex> lock(trace_state->mutex);
        trace_state->worker_trace_json = report.trace_json;
    }
    if (!lifecycle || !report.has_evaluation_report) return true;

    common_flydelta_lifecycle_event_context context;
    context.event_id = job.id + ":evaluation";
    context.idempotency_key = context.event_id;
    context.source_id = "model-free-flydelta-admin-smoke";
    context.scope = job.seed.scope;
    context.content_hash = report.evaluation_report.revision_id + ":" +
        report.evaluation_report.candidate_id;
    context.created_at = job.id;
    if (!common_flydelta_append_evaluation_lifecycle(
            *lifecycle, context, report.evaluation_report,
            report.evaluation_fixture_results, error)) return false;

    common_flydelta_promotion_policy policy;
    common_flydelta_promotion_summary summary;
    const std::string summary_id = job.id + ":promotion-summary";
    if (!common_flydelta_promotion_summary_from_reports(
            summary_id, report.counterfactual_reports, policy, summary, error)) return false;
    common_flydelta_lifecycle_event_context summary_context;
    summary_context.event_id = summary_id;
    summary_context.idempotency_key = summary_id;
    summary_context.source_id = "model-free-flydelta-admin-smoke";
    summary_context.scope = job.seed.scope;
    summary_context.content_hash = summary.candidate_id + ":" + summary.id;
    summary_context.created_at = job.id;
    if (!common_flydelta_append_promotion_summary_lifecycle(
            *lifecycle, summary_context, summary, error)) return false;

    for (const auto & counterfactual : report.counterfactual_reports) {
        common_flydelta_lifecycle_event_context result_context;
        result_context.event_id = job.id + ":counterfactual:" +
            counterfactual.experiment_id + ":" + counterfactual.fixture_id;
        result_context.idempotency_key = result_context.event_id;
        result_context.source_id = "model-free-flydelta-admin-smoke";
        result_context.scope = job.seed.scope;
        result_context.content_hash = counterfactual.experiment_id;
        result_context.created_at = job.id;
        if (!common_flydelta_append_counterfactual_lifecycle(
                *lifecycle, result_context, counterfactual, error)) return false;
    }
    return true;
}

bool wait_for_evaluation(
        const common_learning_lifecycle_store & lifecycle,
        std::string & error) {
    for (size_t attempt = 0; attempt < 500; ++attempt) {
        common_flydelta_evaluation_report report;
        if (common_flydelta_load_evaluation_report(
                lifecycle, kCandidateId, report, nullptr, error)) return true;
        error.clear();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    error = "timed out waiting for model-free FlyDelta evaluation persistence";
    return false;
}

int run_smoke() {
    std::string error;
    auto lifecycle = std::make_shared<common_learning_in_memory_lifecycle_store>();
    auto review_store = std::make_shared<common_flydelta_sideband_review_store>(*lifecycle);
    auto registry = std::make_shared<common_flydelta_sideband_registry>();

    auto manifest = make_manifest();
    if (!review_store->apply_and_append(
            *registry,
            common_flydelta_sideband_review{
                1, "review:admin-smoke:admit", "smoke", "", "", "", "", "admit", // reason/ref fields
                common_flydelta_review_source::operator_action,
                common_flydelta_review_action::admit_experimental, manifest},
            true, error)) {
        std::fprintf(stderr, "failed to seed FlyDelta registry: %s\n", error.c_str());
        return 1;
    }
    manifest.status = common_flydelta_sideband_status::candidate;
    common_flydelta_sideband_review promote_review;
    promote_review.event_id = "review:admin-smoke:candidate";
    promote_review.actor_id = "smoke";
    promote_review.evaluation_revision = kEvaluationRevision;
    promote_review.action = common_flydelta_review_action::promote_to_candidate;
    promote_review.manifest = manifest;
    if (!review_store->apply_and_append(*registry, promote_review, true, error)) {
        std::fprintf(stderr, "failed to seed FlyDelta candidate: %s\n", error.c_str());
        return 1;
    }

    const auto queue_root = std::filesystem::path("/tmp") /
        ("llama-agent-flydelta-admin-smoke-" + std::to_string(static_cast<long long>(getpid())));
    std::filesystem::remove_all(queue_root);
    auto trace_state = std::make_shared<smoke_trace_state>();

    common_flydelta_evaluator_callbacks callbacks;
    callbacks.run_evaluation = run_model_free_evaluation;
    common_flydelta_model_capabilities capabilities;
    capabilities.host_verification = true;
    auto adapter = common_flydelta_model_adapter_from_evaluator(
        common_flydelta_evaluator_config{}, callbacks, capabilities, error);
    if (!adapter) {
        std::fprintf(stderr, "failed to create model-free FlyDelta adapter: %s\n", error.c_str());
        std::filesystem::remove_all(queue_root);
        return 1;
    }

    common_agent_daemon_runtime runtime;
    daemon_options options;
    options.default_mode = "chat";
    options.model_profile = "model-free-smoke";
    runtime.config_store = std::make_shared<common_agent_daemon_config_store>(
        std::make_shared<const daemon_options>(options));
    runtime.flydelta_review_lifecycle_store = lifecycle;
    runtime.flydelta_sideband_review_store = review_store;
    runtime.flydelta_sideband_registry = registry;
    runtime.flydelta_job_enqueue = [queue_root](
            const common_flydelta_experiment_job & job, std::string & enqueue_error) {
        return common_flydelta_experiment_queue_enqueue(
            queue_root, job, common_flydelta_experiment_queue_limits{}, enqueue_error);
    };

    auto worker_trace = trace_state;
    common_agent_daemon_flydelta_worker_config flydelta_config;
    flydelta_config.enabled = true;
    flydelta_config.worker_count = 1;
    flydelta_config.queue_root = queue_root;
    flydelta_config.model_adapter = adapter;
    flydelta_config.poll_interval = std::chrono::milliseconds(5);
    flydelta_config.persist_completed_report = [lifecycle, worker_trace](
            const common_flydelta_experiment_job & job,
            const common_flydelta_experiment_worker_report & report,
            std::string & persist_error) {
        return persist_worker_report(lifecycle, worker_trace, job, report, persist_error);
    };

    int result_code = 0;
    {
        common_agent_daemon_dispatcher dispatcher(
            std::move(runtime), 8, 2, std::move(flydelta_config));

        const auto execute_admin = [&](const json & request,
                const char * expected_event,
                common_agent_daemon_command_result & result) {
            common_agent_daemon_command command;
            std::string command_error;
            if (!parse_agent_daemon_command(
                    request, options, common_agent_runtime_host_mode::chat,
                    command, command_error)) {
                error = command_error;
                return false;
            }
            if (!dispatcher.execute(command, result, command_error)) {
                error = command_error.empty() ? result.error : command_error;
                return false;
            }
            if (result.event != expected_event) {
                error = "unexpected FlyDelta admin event: " + result.event;
                return false;
            }
            std::printf("flydelta_admin_trace=%s\n", json{
                {"category", "model-free-functional-smoke"},
                {"operation", request.value("command", "")},
                {"event", result.event},
                {"target_request_id", result.target_request_id},
                {"payload", result.payload_json.empty() ? json::object() : json::parse(result.payload_json)},
            }.dump().c_str());
            return true;
        };

        common_agent_daemon_command_result result;
        if (!execute_admin(json{
                {"request_id", "admin-evaluate"},
                {"command", "flydelta.evaluate_candidate"},
                {"candidate_id", kCandidateId},
                {"candidate_manifest_ref", kArtifactPath},
                {"suite_ref", kSuiteRef},
                {"evaluation_revision", kEvaluationRevision},
                {"verifier_revision", kVerifierRevision},
                {"model_profile_fingerprint", kModelProfileFingerprint},
                {"tokenizer_fingerprint", kTokenizerFingerprint},
                {"template_fingerprint", kTemplateFingerprint},
                {"execution_context_fingerprint", kExecutionContextFingerprint},
            }, "flydelta.evaluation.queued", result) ||
                !wait_for_evaluation(*lifecycle, error)) {
            std::fprintf(stderr, "model-free evaluation smoke failed: %s\n", error.c_str());
            result_code = 1;
        } else if (!execute_admin(json{
                {"request_id", "admin-get-evaluation"},
                {"command", "flydelta.get_evaluation"},
                {"candidate_id", kCandidateId},
            }, "flydelta.evaluation.loaded", result) ||
                !execute_admin(json{
                    {"request_id", "admin-get-summary"},
                    {"command", "flydelta.get_promotion_summary"},
                    {"candidate_id", kCandidateId},
                }, "flydelta.promotion_summary.loaded", result) ||
                !execute_admin(json{
                    {"request_id", "admin-review"},
                    {"command", "flydelta.review_candidate"},
                    {"candidate_id", kCandidateId},
                    {"decision", "approve_canary"},
                    {"reason", "model-free admin smoke approval"},
                    {"actor_id", "smoke-operator"},
                }, "flydelta.review.recorded", result) ||
                !execute_admin(json{
                    {"request_id", "admin-stage"},
                    {"command", "flydelta.stage_canary"},
                    {"candidate_id", kCandidateId},
                    {"explicit_host_approval", true},
                    {"reason", "model-free admin smoke canary"},
                    {"actor_id", "smoke-operator"},
                }, "flydelta.canary.staged", result)) {
            std::fprintf(stderr, "model-free admin smoke failed: %s\n", error.c_str());
            result_code = 1;
        }
    }

    if (result_code == 0) {
        const auto current = registry->list().find(kCandidateId);
        std::string replay_error;
        common_flydelta_sideband_registry replayed;
        if (current == registry->list().end() ||
                current->second.status != common_flydelta_sideband_status::canary ||
                !review_store->replay(replayed, replay_error) ||
                replayed.list().at(kCandidateId).status != common_flydelta_sideband_status::canary) {
            std::fprintf(stderr, "durable canary replay failed: %s\n", replay_error.c_str());
            result_code = 1;
        } else {
            std::string trace_json;
            {
                std::lock_guard<std::mutex> lock(trace_state->mutex);
                trace_json = trace_state->worker_trace_json;
            }
            const auto trace = json::parse(trace_json);
            if (trace.value("kind", "") != "flydelta_trace" ||
                    trace.value("job_id", "").empty() ||
                    trace.value("phase", "") != "evaluation" ||
                    trace.value("model_evaluations", 0U) == 0U) {
                std::fprintf(stderr, "worker trace is incomplete: %s\n", trace_json.c_str());
                result_code = 1;
            } else {
                std::printf("flydelta_worker_trace=%s\n", trace_json.c_str());
                std::printf("flydelta_admin_smoke=passed category=model-free-functional\n");
            }
        }
    }
    std::filesystem::remove_all(queue_root);
    return result_code;
}

} // namespace

int main() {
    return run_smoke();
}
