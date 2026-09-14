#include "agent/adaptation/flydelta/flydelta-collection.h"

#include <filesystem>

#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (false)

static common_adaptation_evidence evidence() {
    common_adaptation_evidence value;
    value.id = "evidence://repair/collector-1";
    value.source = common_adaptation_evidence_source::tool_repair;
    value.scope.namespace_id = "local";
    value.scope.project_id = "project";
    value.scope.session_id = "session";
    value.scope.turn_id = "turn";
    value.task_fingerprint = "sha256:task";
    value.baseline_ref = "execution:failed";
    value.candidate_ref = "execution:repaired";
    value.verifier_ref = "verifier:v1";
    value.transaction_ids = {"learning://failed", "learning://repaired"};
    value.cause = common_learning_cause::model_behavior;
    value.host_verified = true;
    return value;
}

static common_flydelta_experiment_collection_request request() {
    common_flydelta_experiment_collection_request value;
    value.enabled = true;
    value.kind = common_flydelta_experiment_job_kind::counterfactual;
    value.evidence = evidence();
    value.behavior_key = "tool_use/diagnostics/missing-argument";
    value.model_profile_fingerprint = "sha256:model";
    value.tokenizer_fingerprint = "sha256:tokenizer";
    value.template_fingerprint = "sha256:template";
    value.execution_context_fingerprint = "sha256:execution-context";
    value.capture_manifest_ids = {"flydelta://capture/manifest-1"};
    value.alpha_search.candidates = {0.05f, 0.1f};
    value.alpha_search.max_candidates = 2;
    value.code_revision = "collector-test:v1";
    return value;
}

int main() {
    std::string error;
    const auto root = std::filesystem::temp_directory_path() /
        "llama-agent-flydelta-collection-test";
    std::error_code ignored;
    std::filesystem::remove_all(root, ignored);

    auto value = request();
    common_flydelta_experiment_collection_result result;
    CHECK(common_flydelta_collect_experiment_job(root, {}, value, result, error));
    CHECK(result == common_flydelta_experiment_collection_result::enqueued);
    CHECK(std::filesystem::exists(root / "pending"));

    CHECK(common_flydelta_collect_experiment_job(root, {}, value, result, error));
    CHECK(result == common_flydelta_experiment_collection_result::already_present);

    common_flydelta_claimed_experiment_job claimed;
    CHECK(common_flydelta_experiment_queue_claim_next(root, {}, claimed, error));
    CHECK(claimed.job.seed.scope.project_id == "project");
    CHECK(claimed.job.seed.scope.turn_id == "turn");
    CHECK(claimed.job.kind == common_flydelta_experiment_job_kind::counterfactual);
    CHECK(claimed.job.capture_manifest_ids.size() == 1);

    value.enabled = false;
    CHECK(common_flydelta_collect_experiment_job(root, {}, value, result, error));
    CHECK(result == common_flydelta_experiment_collection_result::disabled);

    value.enabled = true;
    value.evidence.host_verified = false;
    CHECK(!common_flydelta_collect_experiment_job(root, {}, value, result, error));

    value = request();
    value.repair_delta_ids = {"flydelta://delta/invalid-for-counterfactual"};
    CHECK(!common_flydelta_collect_experiment_job(root, {}, value, result, error));

    std::filesystem::remove_all(root, ignored);
    return 0;
}
