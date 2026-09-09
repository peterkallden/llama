#pragma once

#include <string>

// Capability is deliberately separate from the semantic label.  A role can
// be known to the catalog and prepared for native binding before its source
// builder/cache-generation path exists.
struct astc_vulkan_tensor_role_capability {
    std::string semantic_role;
    bool matrix_candidate = false;
    bool native_binding_ready = false;
    bool source_builder_ready = false;
};

// Source-name-derived labels used by cache inspection, discovery and future
// compiled-container catalogs. The exact GGUF tensor name remains the only
// authoritative runtime lookup key.
std::string astc_vulkan_tensor_semantic_role(const std::string & name);
std::string astc_vulkan_tensor_canonical_path(const std::string & name);

// Central role registry for cache planning/runtime preparation.  The current
// implementation has a source builder for FFN-down only; the other matrix
// roles are intentionally marked planned-only until their offline source
// construction and replay gates are implemented.
astc_vulkan_tensor_role_capability astc_vulkan_tensor_role_capability_for_name(
        const std::string & name);
