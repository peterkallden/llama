#include "agent-model-loaders.h"

#include "agent-server-context-host.h"

#include "common.h"

#include <filesystem>
#include <mutex>

namespace {

bool regular_file(const std::string & path, const char * label, std::string & error) {
    std::error_code ec;
    const std::filesystem::path file(path);
    if (!std::filesystem::exists(file, ec) ||
            !std::filesystem::is_regular_file(file, ec)) {
        error = std::string("resident ") + label + " is not a regular file: " + path;
        return false;
    }
    return true;
}

void initialize_llama_once() {
    static std::once_flag once;
    std::call_once(once, []() {
        common_init();
        llama_backend_init();
        llama_numa_init(GGML_NUMA_STRATEGY_DISABLED);
    });
}

} // namespace

common_agent_runtime_loaded_model::~common_agent_runtime_loaded_model() {
    server_context_host.reset();
    chat_templates.reset();
    if (model != nullptr) {
        llama_model_free(model);
        model = nullptr;
    }
}

bool common_agent_runtime_cli_model_loader::load(
        const common_agent_model_selection & selection,
        std::shared_ptr<common_agent_runtime_resident_model> & model,
        std::string & error) {
    model.reset();
    if (selection.backend != "cli") {
        error = "CLI model loader received a non-CLI profile";
        return false;
    }
    if (!selection.mmproj.empty()) {
        error = "CLI model loader does not support mmproj yet; use server-context";
        return false;
    }
    if (!regular_file(selection.path, "model", error)) return false;

    initialize_llama_once();
    auto loaded = std::make_shared<common_agent_runtime_loaded_model>();
    loaded->selection = selection;
    llama_model_params params = llama_model_default_params();
    loaded->model = llama_model_load_from_file(selection.path.c_str(), params);
    if (loaded->model == nullptr) {
        error = "failed to load CLI model: " + selection.path;
        return false;
    }
    loaded->chat_templates = common_chat_templates_init(loaded->model, "");
    if (!loaded->chat_templates) {
        error = "failed to initialize CLI chat templates: " + selection.path;
        return false;
    }
    model = std::move(loaded);
    error.clear();
    return true;
}

bool common_agent_runtime_server_context_model_loader::load(
        const common_agent_model_selection & selection,
        std::shared_ptr<common_agent_runtime_resident_model> & model,
        std::string & error) {
    model.reset();
    if (selection.backend != "server-context") {
        error = "server-context model loader received a non-server-context profile";
        return false;
    }
    if (!regular_file(selection.path, "model", error)) return false;
    if (!selection.mmproj.empty() && !regular_file(selection.mmproj, "mmproj", error)) {
        return false;
    }

    initialize_llama_once();
    common_agent_inference_options options;
    options.model = selection.path;
    options.mmproj = selection.mmproj;
    options.context_size_tokens = selection.context_size_tokens;
    auto host = std::make_shared<common_agent_server_context_host>();
    if (!host->start(make_agent_server_context_host_config(options), error)) {
        return false;
    }

    auto loaded = std::make_shared<common_agent_runtime_loaded_model>();
    loaded->selection = selection;
    loaded->server_context_host = std::move(host);
    model = std::move(loaded);
    error.clear();
    return true;
}

std::shared_ptr<common_agent_runtime_loaded_model>
common_agent_runtime_loaded_model_cast(
        const std::shared_ptr<common_agent_runtime_resident_model> & model) {
    return std::dynamic_pointer_cast<common_agent_runtime_loaded_model>(model);
}
