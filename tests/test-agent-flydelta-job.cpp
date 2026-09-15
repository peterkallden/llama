#include "agent/adaptation/flydelta/flydelta-job.h"

#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (false)

static common_flydelta_experiment_seed seed() {
    common_flydelta_experiment_seed value;
    value.id = "evidence://repair/1/flydelta";
    value.behavior_key = "tool_use/diagnostics/missing-argument";
    value.source = common_adaptation_evidence_source::tool_repair;
    value.scope.namespace_id = "local";
    value.scope.project_id = "project";
    value.scope.session_id = "session";
    value.task_fingerprint = "sha256:task";
    value.model_profile_fingerprint = "sha256:model";
    value.tokenizer_fingerprint = "sha256:tokenizer";
    value.template_fingerprint = "sha256:template";
    value.execution_context_fingerprint = "sha256:execution-context";
    value.baseline_ref = "execution:failed";
    value.candidate_ref = "execution:repaired";
    value.verifier_ref = "verifier:v1";
    value.evidence_ref = "evidence://repair/1";
    value.transaction_ids = {"learning://failed", "learning://repaired"};
    return value;
}

int main() {
    std::string error;
    auto job = common_flydelta_experiment_job{};
    job.id = "flydelta://job/basis-1";
    job.kind = common_flydelta_experiment_job_kind::basis;
    job.seed = seed();
    job.behavior_delta_ids = {"flydelta://delta/1"};
    job.code_revision = "test:v1";
    CHECK(common_flydelta_experiment_job_validate(job, 8, error));
    const auto text = common_flydelta_experiment_job_to_json(job);
    common_flydelta_experiment_job parsed;
    CHECK(common_flydelta_experiment_job_from_json(text, parsed, error));
    CHECK(parsed.kind == common_flydelta_experiment_job_kind::basis);
    CHECK(parsed.seed.behavior_key == job.seed.behavior_key);
    CHECK(parsed.seed.scope.namespace_id == job.seed.scope.namespace_id);
    CHECK(parsed.seed.scope.project_id == job.seed.scope.project_id);
    CHECK(parsed.seed.scope.session_id == job.seed.scope.session_id);
    CHECK(parsed.seed.scope.turn_id == job.seed.scope.turn_id);

    for (const auto source : {
            common_adaptation_evidence_source::planning_revision,
            common_adaptation_evidence_source::research_alternative,
            common_adaptation_evidence_source::dataset_resource,
            common_adaptation_evidence_source::workflow_code,
            common_adaptation_evidence_source::procedure_blueprint,
            common_adaptation_evidence_source::user_correction}) {
        job.seed.source = source;
        const auto source_text = common_flydelta_experiment_job_to_json(job);
        common_flydelta_experiment_job source_parsed;
        CHECK(common_flydelta_experiment_job_from_json(source_text, source_parsed, error));
        CHECK(source_parsed.seed.source == source);
    }

    job.id = "flydelta://job/direction-1";
    job.kind = common_flydelta_experiment_job_kind::direction;
    CHECK(common_flydelta_experiment_job_validate(job, 8, error));
    const auto direction_text = common_flydelta_experiment_job_to_json(job);
    common_flydelta_experiment_job direction_parsed;
    CHECK(common_flydelta_experiment_job_from_json(direction_text, direction_parsed, error));
    CHECK(direction_parsed.kind == common_flydelta_experiment_job_kind::direction);

    job.kind = common_flydelta_experiment_job_kind::counterfactual;
    job.capture_manifest_ids = {"flydelta://capture/1"};
    job.alpha_search.candidates = {0.05f, 0.1f};
    job.alpha_search.max_candidates = 2;
    CHECK(common_flydelta_experiment_job_validate(job, 8, error));

    job.id = "flydelta://job/search-pipeline-1";
    job.kind = common_flydelta_experiment_job_kind::search_pipeline;
    job.behavior_delta_ids = {"flydelta://delta/1"};
    CHECK(common_flydelta_experiment_job_validate(job, 8, error));
    const auto pipeline_text = common_flydelta_experiment_job_to_json(job);
    common_flydelta_experiment_job pipeline_parsed;
    CHECK(common_flydelta_experiment_job_from_json(pipeline_text, pipeline_parsed, error));
    CHECK(pipeline_parsed.kind == common_flydelta_experiment_job_kind::search_pipeline);
    CHECK(pipeline_parsed.capture_manifest_ids == job.capture_manifest_ids);
    CHECK(pipeline_parsed.behavior_delta_ids == job.behavior_delta_ids);

    job.kind = common_flydelta_experiment_job_kind::delta_memory;
    job.seed.split = common_flydelta_training_split::validation;
    job.training_example_ids = {"flydelta://training/1"};
    CHECK(!common_flydelta_experiment_job_validate(job, 8, error));
    return 0;
}
