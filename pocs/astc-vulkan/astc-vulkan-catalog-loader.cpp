#include "astc-vulkan-catalog-loader.h"

#include <filesystem>

bool astc_vulkan_load_catalog_for_cache(
    const std::string & model_path,
    const std::string & requested_cache_path,
    const std::string & requested_catalog_path,
    astc_vulkan_catalog_binding & result,
    std::string & error) {
    result = {};
    if (!astc_vulkan_cache_validate(model_path, requested_cache_path, result.cache, error)) {
        return false;
    }
    const std::filesystem::path catalog_path =
        requested_catalog_path.empty() || requested_catalog_path == "auto"
            ? std::filesystem::path(result.cache.paths.root) / "catalog.astcc"
            : std::filesystem::path(requested_catalog_path);
    if (!astc_vulkan_read_compiled_catalog(catalog_path.string(), result.catalog, error)) {
        error = "compiled ASTC catalog unavailable: " + catalog_path.string() +
            " (" + error + ")";
        return false;
    }
    if (!astc_vulkan_compiled_catalog_matches_manifest(
            result.catalog, result.cache.manifest, error)) {
        return false;
    }
    error.clear();
    return true;
}
