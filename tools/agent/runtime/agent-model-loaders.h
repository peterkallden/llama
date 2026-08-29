#pragma once

#include "agent-model-residency.h"

#include "chat.h"
#include "llama.h"

#include <memory>

class common_agent_server_context_host;

// Backend-specific resources are hidden behind the residency contract, but
// remain inspectable by the runtime session adapter that eventually creates a
// per-session inference context.
struct common_agent_runtime_loaded_model final
    : common_agent_runtime_resident_model {
    ~common_agent_runtime_loaded_model() override;

    common_agent_model_selection selection;
    llama_model * model = nullptr;
    common_chat_templates_ptr chat_templates;
    std::shared_ptr<common_agent_server_context_host> server_context_host;
};

struct common_agent_runtime_model_loader_config {
    int n_gpu_layers = 0;
    int n_threads = 2;
    bool fit_params = true;
};

class common_agent_runtime_cli_model_loader final
    : public common_agent_runtime_model_loader {
public:
    explicit common_agent_runtime_cli_model_loader(
            common_agent_runtime_model_loader_config config = {})
        : config_(config) {}

    bool load(
            const common_agent_model_selection & selection,
            std::shared_ptr<common_agent_runtime_resident_model> & model,
            std::string & error) override;

private:
    common_agent_runtime_model_loader_config config_;
};

class common_agent_runtime_server_context_model_loader final
    : public common_agent_runtime_model_loader {
public:
    explicit common_agent_runtime_server_context_model_loader(
            common_agent_runtime_model_loader_config config = {})
        : config_(config) {}

    bool load(
            const common_agent_model_selection & selection,
            std::shared_ptr<common_agent_runtime_resident_model> & model,
            std::string & error) override;

private:
    common_agent_runtime_model_loader_config config_;
};

std::shared_ptr<common_agent_runtime_loaded_model>
common_agent_runtime_loaded_model_cast(
        const std::shared_ptr<common_agent_runtime_resident_model> & model);
