#include "agent/adaptation/trainer-protocol.h"

#include <cassert>

int main() {
    common_learning_corpus_revision corpus;
    corpus.id = "learning://corpus/1";
    corpus.bundle_hash = "identity:fnv1a64:0123456789abcdef";
    corpus.seed = 99;
    common_learning_training_job job;
    std::string error;
    assert(common_learning_make_training_job(corpus, "base:qwen-small:training-v1",
        "trainer-revision-1", "fake-trainer-1", job, error));
    assert(job.seed == 99);
    common_learning_training_job other_job;
    assert(common_learning_make_training_job(corpus, "base:qwen-large:training-v1",
        "trainer-revision-1", "fake-trainer-1", other_job, error));
    assert(other_job.id != job.id);

    common_learning_training_result result;
    result.job_id = job.id;
    result.adapter_id = "agent-adaptation-v1";
    result.artifact_path = "adapters/agent-adaptation-v1.gguf";
    result.artifact_sha256 = "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    result.corpus_bundle_hash = job.corpus_bundle_hash;
    result.base_training_fingerprint = job.base_training_fingerprint;
    result.trainer_kind = job.trainer_kind;
    result.trainer_version = job.trainer_version;
    result.evaluation_revision = "adapter-eval-1";
    result.evaluation_status = "passed";
    assert(common_learning_validate_training_result(job, result, error));

    result.base_training_fingerprint = "base:other";
    assert(!common_learning_validate_training_result(job, result, error));
    result.base_training_fingerprint = job.base_training_fingerprint;
    result.corpus_bundle_hash = job.corpus_bundle_hash;
    result.evaluation_status = "failed";
    assert(!common_learning_validate_training_result(job, result, error));
    return 0;
}
