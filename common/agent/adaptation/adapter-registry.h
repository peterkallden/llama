#pragma once

#include "agent/adaptation/evaluation-contract.h"
#include "agent/adaptation/trainer-protocol.h"
#include "agent/runtime/model-profile.h"

#include <string>
#include <vector>

enum class common_learning_adapter_status {
    candidate,
    canary,
    active,
    retired,
    rejected,
};

const char * common_learning_adapter_status_name(common_learning_adapter_status status);

struct common_learning_adapter_manifest {
    int schema_version = 1;
    std::string id;
    common_learning_adapter_status status = common_learning_adapter_status::candidate;
    std::string base_architecture;
    std::string serving_model_fingerprint;
    std::string training_model_fingerprint;
    std::string tokenizer_fingerprint;
    std::string chat_template_fingerprint;
    std::string corpus_revision_id;
    std::string corpus_bundle_hash;
    std::string trainer_version;
    std::string artifact_path;
    std::string artifact_sha256;
    std::string evaluation_revision;
    std::string evaluation_status;
};

bool common_learning_validate_adapter_manifest(
        const common_learning_adapter_manifest & manifest,
        std::string & error);

bool common_learning_make_adapter_manifest(
        const common_learning_training_job & job,
        const common_learning_training_result & result,
        const std::string & base_architecture,
        const std::string & serving_model_fingerprint,
        const std::string & tokenizer_fingerprint,
        const std::string & chat_template_fingerprint,
        common_learning_adapter_manifest & manifest,
        std::string & error);

// A trainer result may legitimately be `not_run`: training and evaluation
// are separate lifecycle stages.  Use this constructor after a host-owned
// evaluation report has passed all gates; it binds the report to the
// otherwise unevaluated result without rewriting the trainer's record.
bool common_learning_make_evaluated_adapter_manifest(
        const common_learning_training_job & job,
        const common_learning_training_result & result,
        const common_learning_evaluation_report & evaluation,
        const std::string & base_architecture,
        const std::string & serving_model_fingerprint,
        const std::string & tokenizer_fingerprint,
        const std::string & chat_template_fingerprint,
        common_learning_adapter_manifest & manifest,
        std::string & error);

class common_learning_adapter_registry {
public:
    bool admit(const common_learning_adapter_manifest & manifest, std::string & error);
    bool stage_canary(
            const std::string & id,
            const common_learning_evaluation_report & report,
            std::string & error);
    bool resolve_active_overlays(
            const common_agent_model_profile & profile,
            std::vector<common_learning_adapter_manifest> & overlays,
            std::string & error) const;
    bool activate(const std::string & id, std::string & error);
    bool retire(const std::string & id, std::string & error);
    bool reject(const std::string & id, std::string & error);
    std::vector<common_learning_adapter_manifest> list() const;
    std::string active_id() const;

private:
    std::vector<common_learning_adapter_manifest> manifests;
};
