#pragma once

#include <string>

// Source-name-derived labels used by cache inspection, discovery and future
// compiled-container catalogs. The exact GGUF tensor name remains the only
// authoritative runtime lookup key.
std::string astc_vulkan_tensor_semantic_role(const std::string & name);
std::string astc_vulkan_tensor_canonical_path(const std::string & name);
