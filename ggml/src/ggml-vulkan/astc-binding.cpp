#include "astc-binding.h"

#include <mutex>
#include <unordered_map>

namespace ggml_vk_astc_binding {

namespace {

struct node_binding {
    dispatch_fn fn = nullptr;
    void * user_data = nullptr;
};

std::mutex g_nodes_mutex;
std::unordered_map<const ggml_tensor *, node_binding> g_nodes;

} // namespace

void bind_node(ggml_tensor * node, dispatch_fn fn, void * user_data) {
    if (node == nullptr || fn == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_nodes_mutex);
    g_nodes[node] = { fn, user_data };
}

void unbind_node(ggml_tensor * node) {
    if (node == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_nodes_mutex);
    g_nodes.erase(node);
}

bool can_dispatch(const ggml_tensor * node) {
    if (node == nullptr) {
        return false;
    }
    std::lock_guard<std::mutex> lock(g_nodes_mutex);
    return g_nodes.find(node) != g_nodes.end();
}

bool try_dispatch(const dispatch_context & context) {
    node_binding binding;
    {
        std::lock_guard<std::mutex> lock(g_nodes_mutex);
        const auto it = g_nodes.find(context.node);
        if (it == g_nodes.end()) {
            return false;
        }
        binding = it->second;
    }
    return binding.fn(context, binding.user_data);
}

} // namespace ggml_vk_astc_binding
