#pragma once

#include "astc-vulkan-llama-provider.h"

#include <memory>
#include <string>

struct llama_context;

// Owns the already-prepared ASTC provider for one llama context and registers
// every supported runtime seam: legacy FFN-down, named matrix tensors, and
// token embeddings.  Keeping this in the ASTC PoC prevents CLI, server,
// perplexity, and replay tools from drifting into different D1/D2/E1 paths.
class astc_vulkan_runtime_attachment {
public:
    bool prepare_and_attach(llama_context * context,
                            const astc_vulkan_llama_provider::options & options,
                            std::string & error);
    void reset();

    bool ready() const { return provider_ != nullptr && provider_->ready(); }
    astc_vulkan_llama_provider * provider() { return provider_.get(); }

private:
    std::unique_ptr<astc_vulkan_llama_provider> provider_;
};
