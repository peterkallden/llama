#pragma once

#include "ggml.h"

#include <cstdint>

// Narrow seam between the generic ggml-vulkan scheduler and the ASTC runtime.
// The binding owns no Vulkan objects and does not depend on llama or the ASTC
// cache format. It is intentionally an opt-in hook: when no native dispatcher
// is installed, ggml-vulkan continues with its ordinary operation path.
namespace ggml_vk_astc_binding {

using dispatch_fn = bool (*)(void * backend_context, ggml_tensor * node,
                             uint32_t tensor_index, void * user_data);

// Installs the process-local native dispatcher. The caller owns user_data and
// must clear the dispatcher before destroying it. Replacing an existing
// dispatcher is allowed so reloads remain transactional.
void set_dispatcher(dispatch_fn fn, void * user_data);

void clear_dispatcher(dispatch_fn fn, void * user_data);

// Returns true only when the installed dispatcher consumed the node. A false
// result is the normal path and leaves the existing ggml-vulkan execution
// unchanged.
bool try_dispatch(void * backend_context, ggml_tensor * node,
                  uint32_t tensor_index);

} // namespace ggml_vk_astc_binding

