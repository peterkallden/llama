#pragma once

#include "ggml-vulkan-external-op.h"

#include <string>

namespace ggml_vk_astc_external_op {

// Installs the ASTC-owned dispatcher into the optional generic Vulkan hook.
// This is intentionally called by the opt-in provider only; normal ggml and
// Vulkan runs never install it.
bool install(std::string & error);
void uninstall();
bool get_default_device(ggml_vk_external_op_device_context & result, std::string & error);

void bind_node(ggml_tensor * node,
               ggml_vk_external_op_dispatch_fn fn,
               void * user_data);
void unbind_node(ggml_tensor * node);

} // namespace ggml_vk_astc_external_op
