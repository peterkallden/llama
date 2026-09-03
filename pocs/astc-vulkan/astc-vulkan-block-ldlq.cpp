#include "astc-vulkan-block-ldlq.h"

const char * astc_vulkan_ldlq_order_name(astc_vulkan_ldlq_order order) {
    switch (order) {
        case astc_vulkan_ldlq_order::forward: return "forward";
        case astc_vulkan_ldlq_order::reverse: return "reverse";
        case astc_vulkan_ldlq_order::pivot:   return "pivot";
    }
    return "unknown";
}
