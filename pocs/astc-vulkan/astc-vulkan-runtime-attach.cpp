#include "astc-vulkan-runtime-attach.h"

#include "llama-ext.h"

bool astc_vulkan_runtime_attachment::prepare_and_attach(
        llama_context * context,
        const astc_vulkan_llama_provider::options & options,
        std::string & error) {
    reset();
    if (context == nullptr) {
        error = "ASTC runtime attachment requires a llama context";
        return false;
    }

    auto provider = std::make_unique<astc_vulkan_llama_provider>();
    if (!provider->prepare(options, error)) return false;

    const bool attached =
        llama_set_ffn_down_runtime_provider(
            context, astc_vulkan_llama_provider::is_ready_callback,
            astc_vulkan_llama_provider::run_callback, provider.get()) &&
        llama_set_ffn_down_runtime_native_binding(
            context, astc_vulkan_llama_provider::native_bind_callback) &&
        llama_set_ffn_down_runtime_native_generation_begin(
            context, astc_vulkan_llama_provider::native_generation_begin_callback) &&
        llama_set_tensor_runtime_provider(
            context, astc_vulkan_llama_provider::tensor_is_ready_callback,
            astc_vulkan_llama_provider::tensor_run_callback, provider.get()) &&
        llama_set_tensor_runtime_native_binding(
            context, astc_vulkan_llama_provider::tensor_native_bind_callback) &&
        llama_set_tensor_runtime_native_generation_begin(
            context, astc_vulkan_llama_provider::tensor_generation_begin_callback) &&
        llama_set_embedding_runtime_provider(
            context, astc_vulkan_llama_provider::embedding_is_ready_callback,
            astc_vulkan_llama_provider::embedding_run_callback, provider.get()) &&
        llama_set_embedding_runtime_native_binding(
            context, astc_vulkan_llama_provider::embedding_native_bind_callback) &&
        llama_set_embedding_runtime_native_generation_begin(
            context, astc_vulkan_llama_provider::embedding_generation_begin_callback);
    if (!attached) {
        error = "llama context rejected one or more ASTC runtime callbacks";
        return false;
    }

    provider_ = std::move(provider);
    return true;
}

void astc_vulkan_runtime_attachment::reset() {
    provider_.reset();
}
