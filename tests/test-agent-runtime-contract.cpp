#include "agent/runtime/agent-inference-contracts.h"
#include "tools/agent/runtime/agent-runtime-control.h"

#include <cassert>

int main() {
    const common_agent_inference_options defaults;
    assert(defaults.model.empty());
    assert(defaults.n_predict == -1);
    assert(defaults.n_gpu_layers == 0);
    assert(defaults.fit_params);
    assert(defaults.n_threads == 2);
    assert(defaults.context_size_tokens == 0);
    assert(defaults.mmproj.empty());

    common_agent_inference_options android_request;
    android_request.model = "models/qwen.gguf";
    android_request.n_threads = 4;
    android_request.context_size_tokens = 3072;
    android_request.n_gpu_layers = 99;
    assert(android_request.model == "models/qwen.gguf");
    assert(android_request.n_threads == 4);
    assert(android_request.context_size_tokens == 3072);
    assert(android_request.n_gpu_layers == 99);

    auto cancellation = std::make_shared<common_agent_runtime_cancellation_state>();
    assert(cancellation->request_cancel("android turn cancelled"));
    assert(cancellation->is_cancelled());
    cancellation->reset();
    assert(!cancellation->is_cancelled());
    assert(cancellation->reason().empty());

    assert(static_cast<int>(agent_inference_backend::cli) !=
        static_cast<int>(agent_inference_backend::server_context));
    return 0;
}
