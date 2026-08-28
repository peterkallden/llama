#include "agent/adaptation/trainer-protocol.h"

#include <cassert>

int main() {
    common_learning_training_job job;
    job.id = "learning://job/1";
    job.corpus_revision_id = "learning://corpus/1";
    job.base_training_fingerprint = "base:qwen-small:training-v1";
    job.code_revision = "trainer-revision-1";
    std::string error;
    assert(common_learning_validate_training_job(job, error));

    common_learning_training_result result;
    result.job_id = job.id;
    result.adapter_id = "agent-adaptation-v1";
    result.artifact_path = "adapters/agent-adaptation-v1.gguf";
    result.artifact_sha256 = "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    result.base_training_fingerprint = job.base_training_fingerprint;
    result.trainer_kind = job.trainer_kind;
    result.evaluation_revision = "adapter-eval-1";
    result.evaluation_status = "passed";
    assert(common_learning_validate_training_result(job, result, error));

    result.base_training_fingerprint = "base:other";
    assert(!common_learning_validate_training_result(job, result, error));
    result.base_training_fingerprint = job.base_training_fingerprint;
    result.evaluation_status = "failed";
    assert(!common_learning_validate_training_result(job, result, error));
    return 0;
}
