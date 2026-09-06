#pragma once

#include "ggml-backend.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GGML_VULKAN_EXTERNAL_OP_API_NAME "ggml.vulkan.external_op.v1"
#define GGML_VULKAN_EXTERNAL_OP_API_VERSION 1u
#define GGML_VULKAN_EXTERNAL_DEVICE_API_NAME "ggml.vulkan.external_device.v1"

// Generic Vulkan interop contract. The external owner keeps ownership of its
// private resources; these views are borrowed only during dispatch.
struct ggml_vk_external_op_buffer_view {
    uint64_t native_buffer;
    uint64_t native_device;
    uint64_t offset;
    uint64_t size;
};

typedef bool (*ggml_vk_external_op_get_buffer_fn)(
        void * backend_context,
        const struct ggml_tensor * tensor,
        struct ggml_vk_external_op_buffer_view * result);

struct ggml_vk_external_op_dispatch_context {
    void * backend_context;
    struct ggml_tensor * node;
    uint32_t tensor_index;
    uint64_t native_physical_device;
    uint64_t native_device;
    uint64_t native_queue;
    uint32_t native_queue_family;
    uint64_t native_command_buffer;
    ggml_vk_external_op_get_buffer_fn get_buffer;
};

// Generic device handles for an opt-in owner that must create resources on
// the same Vulkan device as ggml. The caller borrows these handles and never
// destroys them through this interface.
struct ggml_vk_external_op_device_context {
    uint64_t native_physical_device;
    uint64_t native_device;
    uint64_t native_queue;
    uint32_t native_queue_family;
};

typedef bool (*ggml_vk_external_op_get_device_fn)(
        struct ggml_vk_external_op_device_context * result);

typedef bool (*ggml_vk_external_op_dispatch_fn)(
        const struct ggml_vk_external_op_dispatch_context * context,
        void * user_data);

typedef bool (*ggml_vk_external_op_can_dispatch_fn)(
        const struct ggml_tensor * node,
        void * user_data);

// Optional dispatch-time preflight. Unlike can_dispatch(), this callback sees
// the actual backend device and command buffer, so an external owner can
// decline a node without turning a device mismatch into a hard failure.
typedef bool (*ggml_vk_external_op_can_dispatch_context_fn)(
        const struct ggml_vk_external_op_dispatch_context * context,
        void * user_data);

struct ggml_vk_external_op_dispatcher {
    ggml_vk_external_op_can_dispatch_fn can_dispatch;
    ggml_vk_external_op_can_dispatch_context_fn can_dispatch_context;
    ggml_vk_external_op_dispatch_fn dispatch;
    void * user_data;
};

struct ggml_vk_external_op_api {
    uint32_t api_version;
    void (*set_dispatcher)(const struct ggml_vk_external_op_dispatcher * dispatcher);
};

typedef const struct ggml_vk_external_op_api * (*ggml_vk_external_op_get_api_fn)(void);

GGML_API const struct ggml_vk_external_op_api * ggml_vk_external_op_get_api(void);

GGML_API void ggml_vk_external_op_set_dispatcher(
        const struct ggml_vk_external_op_dispatcher * dispatcher);

GGML_API bool ggml_vk_external_op_get_default_device(
        struct ggml_vk_external_op_device_context * result);

#ifdef __cplusplus
}
#endif
