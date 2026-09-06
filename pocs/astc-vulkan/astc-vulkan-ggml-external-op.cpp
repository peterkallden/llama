#include "astc-vulkan-ggml-external-op.h"

#include "ggml-backend.h"

#include <mutex>
#include <unordered_map>

namespace ggml_vk_astc_external_op {

namespace {

struct node_binding {
    ggml_vk_external_op_dispatch_fn fn = nullptr;
    void * user_data = nullptr;
};

std::mutex g_mutex;
std::unordered_map<const ggml_tensor *, node_binding> g_bindings;
bool g_installed = false;
uint32_t g_install_refs = 0;

bool can_dispatch(const ggml_tensor * node, void *) {
    if (node == nullptr) {
        return false;
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_bindings.find(node) != g_bindings.end();
}

bool dispatch(const ggml_vk_external_op_dispatch_context * context, void *) {
    if (context == nullptr || context->node == nullptr) {
        return false;
    }
    node_binding binding;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        const auto it = g_bindings.find(context->node);
        if (it == g_bindings.end()) {
            return false;
        }
        binding = it->second;
    }
    return binding.fn != nullptr && binding.fn(context, binding.user_data);
}

} // namespace

bool install(std::string & error) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_installed) {
        ++g_install_refs;
        return true;
    }

    ggml_backend_reg_t reg = ggml_backend_reg_by_name("Vulkan");
    if (reg == nullptr) {
        error = "Vulkan backend registry is unavailable";
        return false;
    }
    void * proc = ggml_backend_reg_get_proc_address(reg, GGML_VULKAN_EXTERNAL_OP_API_NAME);
    if (proc == nullptr) {
        error = "Vulkan backend does not expose the generic external-op interface";
        return false;
    }
    const auto get_api = reinterpret_cast<ggml_vk_external_op_get_api_fn>(proc);
    const ggml_vk_external_op_api * api = get_api != nullptr ? get_api() : nullptr;
    if (api == nullptr || api->api_version != GGML_VULKAN_EXTERNAL_OP_API_VERSION ||
        api->set_dispatcher == nullptr) {
        error = "unsupported Vulkan external-op interface version";
        return false;
    }

    const ggml_vk_external_op_dispatcher dispatcher = {
        can_dispatch,
        dispatch,
        nullptr,
    };
    api->set_dispatcher(&dispatcher);
    g_installed = true;
    g_install_refs = 1;
    return true;
}

void uninstall() {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_installed || g_install_refs == 0) {
        return;
    }
    if (--g_install_refs != 0) {
        return;
    }
    ggml_backend_reg_t reg = ggml_backend_reg_by_name("Vulkan");
    if (reg != nullptr) {
        void * proc = ggml_backend_reg_get_proc_address(reg, GGML_VULKAN_EXTERNAL_OP_API_NAME);
        if (proc != nullptr) {
            const auto get_api = reinterpret_cast<ggml_vk_external_op_get_api_fn>(proc);
            const ggml_vk_external_op_api * api = get_api != nullptr ? get_api() : nullptr;
            if (api != nullptr && api->set_dispatcher != nullptr) {
                api->set_dispatcher(nullptr);
            }
        }
    }
    g_bindings.clear();
    g_installed = false;
}

bool get_default_device(ggml_vk_external_op_device_context & result, std::string & error) {
    ggml_backend_reg_t reg = ggml_backend_reg_by_name("Vulkan");
    if (reg == nullptr) {
        error = "Vulkan backend registry is unavailable";
        return false;
    }
    void * proc = ggml_backend_reg_get_proc_address(reg, GGML_VULKAN_EXTERNAL_DEVICE_API_NAME);
    if (proc == nullptr) {
        error = "Vulkan backend does not expose the generic external-device interface";
        return false;
    }
    const auto get_device = reinterpret_cast<ggml_vk_external_op_get_device_fn>(proc);
    if (get_device == nullptr || !get_device(&result)) {
        error = "Vulkan backend could not provide a default device context";
        return false;
    }
    error.clear();
    return true;
}

void bind_node(ggml_tensor * node,
               ggml_vk_external_op_dispatch_fn fn,
               void * user_data) {
    if (node == nullptr || fn == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    g_bindings[node] = { fn, user_data };
}

void unbind_node(ggml_tensor * node) {
    if (node == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    g_bindings.erase(node);
}

} // namespace ggml_vk_astc_external_op
