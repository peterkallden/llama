#include "tools/agent/runtime/agent-model-loaders.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>

namespace {

struct options {
    std::string model_1;
    std::string model_2;
    std::string backend = "server-context";
    int n_gpu_layers = 0;
    int n_threads = 3;
};

bool parse_args(int argc, char ** argv, options & value) {
    if (const char * model = std::getenv("LLAMA_AGENT_MODEL_1")) value.model_1 = model;
    if (const char * model = std::getenv("LLAMA_AGENT_MODEL_2")) value.model_2 = model;
    for (int i = 1; i < argc; ++i) {
        auto next = [&](const char * name) -> const char * {
            if (i + 1 >= argc) {
                std::cerr << "missing value for " << name << '\n';
                return nullptr;
            }
            return argv[++i];
        };
        const std::string argument = argv[i];
        if (argument == "--model-1") {
            const char * model = next(argv[i]); if (!model) return false; value.model_1 = model;
        } else if (argument == "--model-2") {
            const char * model = next(argv[i]); if (!model) return false; value.model_2 = model;
        } else if (argument == "--backend") {
            const char * backend = next(argv[i]); if (!backend) return false; value.backend = backend;
        } else if (argument == "--n-gpu-layers") {
            const char * layers = next(argv[i]); if (!layers) return false; value.n_gpu_layers = std::stoi(layers);
        } else if (argument == "--threads") {
            const char * threads = next(argv[i]); if (!threads) return false; value.n_threads = std::stoi(threads);
        } else if (argument == "--help" || argument == "-h") {
            return false;
        } else {
            std::cerr << "unknown argument: " << argument << '\n';
            return false;
        }
    }
    return true;
}

bool regular_model(const std::string & path) {
    std::error_code error;
    return !path.empty() && std::filesystem::is_regular_file(path, error);
}

} // namespace

int main(int argc, char ** argv) {
    options value;
    if (!parse_args(argc, argv, value)) {
        std::cerr << "usage: " << argv[0]
                  << " --model-1 PATH --model-2 PATH"
                  << " [--backend cli|server-context] [--threads N] [--n-gpu-layers N]\n";
        return 2;
    }
    if (!regular_model(value.model_1) || !regular_model(value.model_2)) {
        std::cerr << "two-model smoke skipped: provide two regular GGUF files\n";
        return 77;
    }
    if (value.model_1 == value.model_2) {
        std::cerr << "two-model smoke requires two distinct model paths\n";
        return 2;
    }
    if (value.backend != "cli" && value.backend != "server-context") {
        std::cerr << "--backend must be cli or server-context\n";
        return 2;
    }

    const common_agent_runtime_model_loader_config loader_config{
        value.n_gpu_layers,
        value.n_threads,
        true,
    };
    std::shared_ptr<common_agent_runtime_model_loader> loader;
    if (value.backend == "cli") {
        loader = std::make_shared<common_agent_runtime_cli_model_loader>(loader_config);
    } else {
#ifdef LLAMA_AGENT_ANDROID_CLI_ONLY
        std::cerr << "server-context backend is unavailable in this host\n";
        return 77;
#else
        loader = std::make_shared<common_agent_runtime_server_context_model_loader>(loader_config);
#endif
    }

    common_agent_model_selection first_selection;
    first_selection.profile_id = "smoke-model-1";
    first_selection.base_model_id = "smoke-base-1";
    first_selection.backend = value.backend;
    first_selection.path = value.model_1;
    first_selection.context_size_tokens = 4096;
    first_selection.load_policy = "resident";

    common_agent_model_selection second_selection = first_selection;
    second_selection.profile_id = "smoke-model-2";
    second_selection.base_model_id = "smoke-base-2";
    second_selection.path = value.model_2;

    std::shared_ptr<common_agent_runtime_resident_model> first;
    std::shared_ptr<common_agent_runtime_resident_model> second;
    std::string error;
    if (!loader->load(first_selection, first, error)) {
        std::cerr << "model 1 load failed: " << error << '\n';
        return 1;
    }
    if (!loader->load(second_selection, second, error)) {
        std::cerr << "model 2 load failed: " << error << '\n';
        return 1;
    }
    if (!first || !second || first == second) {
        std::cerr << "two-model smoke did not produce two distinct resident resources\n";
        return 1;
    }

    std::cout << "two_model_load=passed\n"
              << "backend=" << value.backend << '\n'
              << "model_1=" << value.model_1 << '\n'
              << "model_2=" << value.model_2 << '\n';
    return 0;
}
