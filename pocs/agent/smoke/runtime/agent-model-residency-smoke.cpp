#include "tools/agent/runtime/agent-model-residency.h"
#include "tools/agent/runtime/agent-model-loaders.h"

#include <cassert>
#include <memory>
#include <unordered_map>

namespace {

struct fake_model final : common_agent_runtime_resident_model {
    explicit fake_model(std::string value) : id(std::move(value)) {}
    std::string id;
};

class fake_loader final : public common_agent_runtime_model_loader {
public:
    bool load(const common_agent_model_selection & selection,
            std::shared_ptr<common_agent_runtime_resident_model> & model,
            std::string & error) override {
        ++loads;
        model = std::make_shared<fake_model>(selection.profile_id);
        error.clear();
        return true;
    }

    size_t loads = 0;
};

common_agent_model_catalog make_catalog() {
    common_agent_model_catalog catalog;
    catalog.directory = "/models";
    catalog.max_loaded_generation_models = 1;
    catalog.bases = {
        {"small", {"generation", "cli", "small.gguf", "", "lazy"}},
        {"research", {"generation", "server-context", "research.gguf", "", "lazy"}},
    };
    catalog.profiles = {
        {"small", {"small", {}, 4096, "lazy"}},
        {"research", {"research", {}, 4096, "lazy"}},
    };
    catalog.default_profile = "small";
    return catalog;
}

} // namespace

int main() {
    auto cli = std::make_shared<fake_loader>();
    auto server = std::make_shared<fake_loader>();
    std::unordered_map<std::string,
        std::shared_ptr<common_agent_runtime_model_loader>> loaders{
            {"cli", cli}, {"server-context", server}};
    common_agent_runtime_model_residency residency(make_catalog(), std::move(loaders));

    std::string error;
    common_agent_runtime_model_resident_handle first;
    assert(residency.acquire("small", first, error));
    assert(first.valid() && cli->loads == 1 && server->loads == 0);

    common_agent_runtime_model_resident_handle reused;
    assert(residency.acquire("small", reused, error));
    assert(reused.model == first.model && cli->loads == 1);
    assert(!residency.acquire("research", first, error));
    assert(error.find("pinned") != std::string::npos);

    assert(residency.release(reused, error));
    assert(residency.release(first, error));

    common_agent_runtime_model_resident_handle second;
    assert(residency.acquire("research", second, error));
    assert(second.valid() && server->loads == 1);
    assert(residency.list().size() == 1);
    assert(residency.release(second, error));

    // Concrete loaders reject an incompatible backend and fail closed before
    // touching llama.cpp when the selected model path is unavailable.
    common_agent_model_selection missing_cli;
    missing_cli.profile_id = "missing-cli";
    missing_cli.backend = "cli";
    missing_cli.path = "/definitely/missing/model.gguf";
    common_agent_runtime_cli_model_loader cli_loader;
    std::shared_ptr<common_agent_runtime_resident_model> loaded;
    assert(!cli_loader.load(missing_cli, loaded, error));
    assert(error.find("regular file") != std::string::npos);

    common_agent_model_selection wrong_backend = missing_cli;
    wrong_backend.backend = "server-context";
    assert(!cli_loader.load(wrong_backend, loaded, error));
    assert(error.find("non-CLI") != std::string::npos);

    common_agent_runtime_server_context_model_loader server_loader;
    assert(!server_loader.load(wrong_backend, loaded, error));
    assert(error.find("regular file") != std::string::npos);
    error.clear();
    return error.empty() ? 0 : 1;
}
