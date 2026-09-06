#pragma once

#include "ggml.h"

#include <cstdint>

// Narrow seam between the generic ggml-vulkan scheduler and an optional
// native ASTC node. This header deliberately exposes no Vulkan or llama types.
namespace ggml_vk_astc_binding {

// Vulkan handles are represented as integers so the binding remains usable by
// providers that include either Vulkan-Hpp or the C Vulkan headers. Handles
// and the command buffer are valid only during dispatch_fn.
struct buffer_view {
    uint64_t native_buffer = 0;
    uint64_t native_device = 0;
    uint64_t offset = 0;
    uint64_t size = 0;
};

using get_buffer_fn = bool (*)(void * backend_context,
                               const ggml_tensor * tensor,
                               buffer_view * result);

struct dispatch_context {
    void * backend_context = nullptr;
    ggml_tensor * node = nullptr;
    uint32_t tensor_index = 0;
    uint64_t native_device = 0;
    uint64_t native_command_buffer = 0;
    get_buffer_fn get_buffer = nullptr;
};

using dispatch_fn = bool (*)(const dispatch_context & context, void * user_data);

// Bind one graph node to a native dispatcher. The owner must unbind the node
// before its ggml_tensor lifetime ends. Binding is intentionally node-scoped:
// it prevents an experimental ASTC provider from claiming unrelated Vulkan
// operations.
void bind_node(ggml_tensor * node, dispatch_fn fn, void * user_data);
void unbind_node(ggml_tensor * node);
bool can_dispatch(const ggml_tensor * node);

// Returns true when a registered callback consumed the node. For a node bound
// through bind_node(), false is a provider failure and must not fall back to a
// different Vulkan implementation because the node has no ordinary op body.
bool try_dispatch(const dispatch_context & context);

} // namespace ggml_vk_astc_binding
