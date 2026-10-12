#include "agent/adaptation/flydelta/flydelta-queue.h"

#include <filesystem>

#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (false)

static common_flydelta_experiment_job make_job() {
    common_flydelta_experiment_job job;
    job.id = "flydelta://job/queue-1";
    job.kind = common_flydelta_experiment_job_kind::basis;
    job.seed.id = "evidence://repair/queue-1/flydelta";
    job.seed.behavior_key = "tool_use/diagnostics/missing-argument";
    job.seed.source = common_adaptation_evidence_source::tool_repair;
    job.seed.scope.namespace_id = "local";
    job.seed.scope.project_id = "project";
    job.seed.scope.session_id = "session";
    job.seed.task_fingerprint = "sha256:task";
    job.seed.model_profile_fingerprint = "sha256:model";
    job.seed.tokenizer_fingerprint = "sha256:tokenizer";
    job.seed.template_fingerprint = "sha256:template";
    job.seed.execution_context_fingerprint = "sha256:execution-context";
    job.seed.baseline_ref = "execution:failed";
    job.seed.candidate_ref = "execution:repaired";
    job.seed.verifier_ref = "verifier:v1";
    job.seed.evidence_ref = "evidence://repair/queue-1";
    job.seed.transaction_ids = {"learning://failed", "learning://repaired"};
    job.behavior_delta_ids = {"flydelta://delta/queue-1"};
    job.code_revision = "test:v1";
    return job;
}

int main() {
    namespace fs = std::filesystem;
    const auto root = fs::temp_directory_path() / "llama-agent-flydelta-queue-test";
    std::error_code ignored;
    fs::remove_all(root, ignored);
    std::string error;
    const auto limits = common_flydelta_experiment_queue_limits{};
    const auto job = make_job();

    CHECK(common_flydelta_experiment_queue_enqueue(root, job, limits, error));
    CHECK(fs::directory_iterator(root / "pending") != fs::directory_iterator{});
    CHECK(!common_flydelta_experiment_queue_enqueue(root, job, limits, error));

    common_flydelta_claimed_experiment_job claimed;
    CHECK(common_flydelta_experiment_queue_claim_next(root, limits, claimed, error));
    CHECK(claimed.job.id == job.id);
    CHECK(claimed.job.seed.scope.session_id == job.seed.scope.session_id);
    CHECK(fs::exists(root / "running" / claimed.queue_key / "job.json"));
    CHECK(common_flydelta_experiment_queue_complete(root, claimed,
        common_flydelta_experiment_queue_state::succeeded, "basis candidate ready", limits, error));
    CHECK(fs::exists(root / "succeeded" / claimed.queue_key / "state.json"));

    common_flydelta_claimed_experiment_job empty;
    CHECK(common_flydelta_experiment_queue_claim_next(root, limits, empty, error));
    CHECK(empty.queue_key.empty());

    auto small_limits = limits;
    small_limits.max_job_bytes = 8;
    auto oversized = make_job();
    oversized.id = "flydelta://job/queue-oversized";
    CHECK(!common_flydelta_experiment_queue_enqueue(root, oversized, small_limits, error));

    fs::remove_all(root, ignored);
    return 0;
}
