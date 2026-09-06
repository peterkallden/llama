#include "astc-vulkan-runtime-overlay.h"

#include <algorithm>
#include <limits>

void astc_vulkan_runtime_overlay::reset() {
    ready_ = false;
    catalog_ = {};
    page_owner_.reset();
    summary_ = {};
}

bool astc_vulkan_runtime_overlay::prepare(
        const astc_vulkan_runtime_overlay_options & options, std::string & error) {
    reset();
    if (options.source_model_path.empty() || options.runtime_model_path.empty()) {
        error = "ASTC runtime overlay requires source and runtime model paths";
        return false;
    }
    if (options.atlas_slots_x == 0) {
        error = "ASTC runtime overlay requires at least one atlas slot per row";
        return false;
    }
    if (options.memory_budget.effective_device_limit_bytes == 0) {
        error = "ASTC runtime overlay requires a detected device memory budget";
        return false;
    }

    if (!astc_vulkan_model_cache_load_catalog_for_runtime(
            options.source_model_path, options.runtime_model_path, options.cache_path,
            options.policy, catalog_, error)) {
        return false;
    }

    astc_vulkan_model_cache_plan resident_plan;
    if (!astc_vulkan_model_cache_plan_residency(
            catalog_.plan, options.memory_budget, resident_plan, error)) {
        reset();
        return false;
    }
    catalog_.plan.residency = resident_plan.residency;

    std::vector<astc_vulkan_model_cache_storage_page> pages;
    if (!astc_vulkan_model_cache_make_storage_pages(
            catalog_.plan, options.max_page_payload_bytes, pages, error) ||
        !page_owner_.prepare(catalog_.plan, pages, options.atlas_slots_x, error)) {
        reset();
        return false;
    }

    // A page is resident only when every artifact in it belongs to the
    // selected residency prefix. This keeps a page's physical lifetime
    // deterministic and avoids silently loading artifacts beyond budget.
    std::vector<size_t> resident_item_to_entry;
    resident_item_to_entry.reserve(catalog_.plan.entries.size());
    for (size_t entry_index = 0; entry_index < catalog_.plan.entries.size(); ++entry_index) {
        if (!catalog_.plan.entries[entry_index].use_native_fallback) {
            resident_item_to_entry.push_back(entry_index);
        }
    }
    std::vector<bool> entry_resident(catalog_.plan.entries.size(), false);
    for (const size_t item_index : catalog_.plan.residency.resident_items) {
        if (item_index < resident_item_to_entry.size()) {
            entry_resident[resident_item_to_entry[item_index]] = true;
        }
    }
    std::vector<size_t> resident_pages;
    for (const auto & page : page_owner_.pages()) {
        const bool all_resident = !page.entry_indices.empty() &&
            std::all_of(page.entry_indices.begin(), page.entry_indices.end(),
                        [&entry_resident](size_t index) {
                            return index < entry_resident.size() && entry_resident[index];
                        });
        if (all_resident) resident_pages.push_back(page.page_index);
    }
    if (!page_owner_.set_resident_pages(resident_pages, error)) {
        reset();
        return false;
    }

    summary_.planned_artifacts = catalog_.plan.entries.size();
    summary_.pages = page_owner_.pages().size();
    summary_.resident_pages = resident_pages.size();
    summary_.resident_device_bytes = catalog_.plan.residency.device_bytes;
    summary_.resident_host_bytes = catalog_.plan.residency.host_bytes;
    summary_.preload_all = catalog_.plan.residency.preload_all;
    summary_.requires_streaming = catalog_.plan.residency.requires_streaming;
    for (size_t index = 0; index < catalog_.plan.entries.size(); ++index) {
        astc_vulkan_page_resolve resolve;
        std::string resolve_error;
        if (!page_owner_.resolve_entry(index, resolve, resolve_error)) {
            error = resolve_error;
            reset();
            return false;
        }
        if (resolve.resident) ++summary_.resident_artifacts;
        else ++summary_.native_fallbacks;
    }
    ready_ = true;
    error.clear();
    return true;
}

bool astc_vulkan_runtime_overlay::resolve_tensor(
        const std::string & tensor_name, astc_vulkan_page_material & material,
        std::string & error) const {
    material = {};
    if (!ready_) {
        error = "ASTC runtime overlay is not prepared";
        return false;
    }
    return page_owner_.load_tensor_material(catalog_, tensor_name, material, error);
}
