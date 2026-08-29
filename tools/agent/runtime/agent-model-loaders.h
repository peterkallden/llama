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

class common_agent_runtime_cli_model_loader final
    : public common_agent_runtime_model_loader {
public:
    bool load(
            const common_agent_model_selection & selection,
            std::shared_ptr<common_agent_runtime_resident_model> & model,
            std::string & error) override;
};

class common_agent_runtime_server_context_model_loader final
    : public common_agent_runtime_model_loader {
public:
    bool load(
            const common_agent_model_selection & selection,
            std::shared_ptr<common_agent_runtime_resident_model> & model,
            std::string & error) override;
};

std::shared_ptr<common_agent_runtime_loaded_model>
common_agent_runtime_loaded_model_cast(
        const std::shared_ptr<common_agent_runtime_resident_model> & model);
