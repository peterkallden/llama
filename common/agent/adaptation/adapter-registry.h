#pragma once

#include <string>
#include <vector>

enum class common_learning_adapter_status {
    candidate,
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
    std::string artifact_path;
    std::string artifact_sha256;
    std::string evaluation_revision;
    std::string evaluation_status;
};

bool common_learning_validate_adapter_manifest(
        const common_learning_adapter_manifest & manifest,
        std::string & error);

class common_learning_adapter_registry {
public:
    bool admit(const common_learning_adapter_manifest & manifest, std::string & error);
    bool activate(const std::string & id, std::string & error);
    bool retire(const std::string & id, std::string & error);
    bool reject(const std::string & id, std::string & error);
    std::vector<common_learning_adapter_manifest> list() const;
    std::string active_id() const;

private:
    std::vector<common_learning_adapter_manifest> manifests;
};
