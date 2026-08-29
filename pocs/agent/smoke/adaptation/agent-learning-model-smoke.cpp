#include "tools/agent/adaptation/agent-learning-transaction-store.h"
#include "tools/agent/runtime/agent-runtime-session-host.h"

#include "memory/memory-in-memory.h"
#include "plan/plan-in-memory.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

namespace {

struct options {
    std::string model;
    std::string prompt = "Hi there. Reply briefly.";
    int n_predict = 32;
    int n_threads = 3;
    int n_gpu_layers = 0;
};

bool parse_args(int argc, char ** argv, options & value) {
    if (const char * environment_model = std::getenv("LLAMA_AGENT_MODEL")) {
        value.model = environment_model;
    }
    for (int i = 1; i < argc; ++i) {
        auto next = [&](const char * name) -> const char * {
            if (i + 1 >= argc) {
                std::cerr << "missing value for " << name << '\n';
                return nullptr;
            }
            return argv[++i];
        };
        if (std::string(argv[i]) == "--model") {
            const char * next_value = next(argv[i]); if (!next_value) return false; value.model = next_value;
        } else if (std::string(argv[i]) == "--prompt") {
            const char * next_value = next(argv[i]); if (!next_value) return false; value.prompt = next_value;
        } else if (std::string(argv[i]) == "--n-predict") {
            const char * next_value = next(argv[i]); if (!next_value) return false; value.n_predict = std::stoi(next_value);
        } else if (std::string(argv[i]) == "--threads") {
            const char * next_value = next(argv[i]); if (!next_value) return false; value.n_threads = std::stoi(next_value);
        } else if (std::string(argv[i]) == "--n-gpu-layers") {
            const char * next_value = next(argv[i]); if (!next_value) return false; value.n_gpu_layers = std::stoi(next_value);
        } else if (std::string(argv[i]) == "--help" || std::string(argv[i]) == "-h") {
            return false;
        } else {
            std::cerr << "unknown argument: " << argv[i] << '\n';
            return false;
        }
    }
    return true;
}

} // namespace

int main(int argc, char ** argv) {
    options value;
    if (!parse_args(argc, argv, value)) {
        std::cerr << "usage: " << argv[0]
                  << " --model MODEL [--prompt TEXT] [--n-predict N] [--threads N] [--n-gpu-layers N]\n";
        return 2;
    }
    if (value.model.empty() || !std::filesystem::is_regular_file(value.model)) {
        std::cerr << "model smoke skipped: provide --model or LLAMA_AGENT_MODEL\n";
        return 77;
    }

    common_memory_in_memory_store memory_store;
    common_plan_in_memory_store plan_store;
    std::string error;
    if (!memory_store.open("", error) || !plan_store.open("", error)) {
        std::cerr << "model smoke could not open host stores: " << error << '\n';
        return 1;
    }
    const auto root = std::filesystem::temp_directory_path() / "llama-agent-model-adaptation-smoke";
    std::error_code ec;
    std::filesystem::remove_all(root, ec);
    const auto transaction_path = root / "transactions.jsonl";

    common_agent_runtime_policy policy;
    policy.agent_inference_backend = "server-context";
    common_agent_runtime_config runtime_config;
    runtime_config.generation_config.n_predict = value.n_predict;
    runtime_config.generation_config.n_threads = value.n_threads;
    runtime_config.enable_adaptation_capture = true;
    runtime_config.adaptation_transaction_backend = "jsonl";
    runtime_config.adaptation_transaction_path = transaction_path.string();
    runtime_config.adaptation_config.collection_allowed = true;

    common_agent_runtime_session_host runtime(
        make_agent_runtime_session_host_config({
            memory_store,
            plan_store,
            {
                value.prompt,
                "model-adaptation-smoke-session",
                "model-adaptation-smoke",
                {},
                std::nullopt,
                value.model,
                value.n_predict,
                value.n_gpu_layers,
                true,
                "server-context",
                common_memory_scope::session,
                common_plan_scope::turn,
                value.n_threads,
            },
            std::move(policy),
            std::move(runtime_config),
            {},
            common_memory_scope::session,
            false,
            {},
            {},
        }));

    common_agent_runtime_session_host_turn_result turn;
    if (!runtime.run_turn({
            common_agent_runtime_host_mode::chat,
            value.prompt,
            "model-adaptation-smoke-session",
            "model-adaptation-smoke",
            {},
            "model-adaptation-smoke-turn",
            common_memory_scope::session,
            common_plan_scope::turn,
            value.n_predict,
        }, turn, error)) {
        std::cerr << "model smoke turn failed: " << error << '\n';
        std::filesystem::remove_all(root, ec);
        return 1;
    }

    auto ledger = make_agent_learning_transaction_store(
        "jsonl", transaction_path.string(), error);
    if (!ledger) {
        std::cerr << "model smoke could not reopen adaptation ledger: " << error << '\n';
        std::filesystem::remove_all(root, ec);
        return 1;
    }
    const auto observations = ledger->list(error);
    if (!error.empty()) {
        std::cerr << "model smoke could not read adaptation ledger: " << error << '\n';
        std::filesystem::remove_all(root, ec);
        return 1;
    }

    std::cout << "model_turn=passed\n"
              << "response_generation_status=" << static_cast<int>(turn.response_generation_status) << '\n'
              << "decoded_tokens=" << turn.total_decoded_tokens << '\n'
              << "adaptation_ledger_readable=yes\n"
              << "learning_observations=" << observations.size() << '\n';
    std::filesystem::remove_all(root, ec);
    return 0;
}
