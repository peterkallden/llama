#include "astc-binding.h"

#include <atomic>

namespace ggml_vk_astc_binding {

namespace {

std::atomic<dispatch_fn> g_dispatcher{nullptr};
std::atomic<void *> g_user_data{nullptr};

} // namespace

void set_dispatcher(dispatch_fn fn, void * user_data) {
    // Publish the user data before the function pointer so a reader that sees
    // a dispatcher always sees its matching context.
    g_user_data.store(user_data, std::memory_order_release);
    g_dispatcher.store(fn, std::memory_order_release);
}

void clear_dispatcher(dispatch_fn fn, void * user_data) {
    dispatch_fn expected = fn;
    if (g_dispatcher.compare_exchange_strong(expected, nullptr,
                                              std::memory_order_acq_rel,
                                              std::memory_order_acquire)) {
        void * expected_user_data = user_data;
        g_user_data.compare_exchange_strong(expected_user_data, nullptr,
                                             std::memory_order_acq_rel,
                                             std::memory_order_acquire);
    }
}

bool try_dispatch(void * backend_context, ggml_tensor * node,
                  uint32_t tensor_index) {
    const dispatch_fn fn = g_dispatcher.load(std::memory_order_acquire);
    if (fn == nullptr) return false;
    void * const user_data = g_user_data.load(std::memory_order_acquire);
    return fn(backend_context, node, tensor_index, user_data);
}

} // namespace ggml_vk_astc_binding

