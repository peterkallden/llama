#pragma once

#include "astc-vulkan-cache.h"
#include "astc-vulkan-compiled-catalog.h"

#include <string>

// Metadata-only bootstrap binding. It owns no payload memory and creates no
// Vulkan resources; those remain owned by the existing cache/page/runtime
// components.
struct astc_vulkan_catalog_binding {
    astc_vulkan_cache_validation cache;
    astc_vulkan_compiled_catalog catalog;
};

// Load and verify a compiled catalog for a validated sidecar cache. An empty
// or "auto" catalog path resolves to <cache-root>/catalog.astcc.
bool astc_vulkan_load_catalog_for_cache(
    const std::string & model_path,
    const std::string & requested_cache_path,
    const std::string & requested_catalog_path,
    astc_vulkan_catalog_binding & result,
    std::string & error);
