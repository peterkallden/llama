#include "astc-vulkan-page-owner.h"

#include "astc-vulkan-stream-loader.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <unordered_map>

namespace {

bool same_key(const astc_vulkan_model_cache_storage_key & lhs,
              const astc_vulkan_model_cache_storage_key & rhs) {
    return lhs.footprint == rhs.footprint &&
           lhs.representation == rhs.representation &&
           lhs.paired_semantic == rhs.paired_semantic &&
           lhs.normalization == rhs.normalization &&
           lhs.has_row_scales == rhs.has_row_scales &&
           lhs.has_pair_map == rhs.has_pair_map;
}

bool entry_key(const astc_vulkan_model_cache_entry & entry,
               astc_vulkan_model_cache_storage_key & key) {
    if (entry.use_native_fallback) return false;
    key.footprint = entry.storage.footprint;
    key.representation = entry.storage.representation;
    key.paired_semantic = entry.paired_semantic;
    key.normalization = entry.normalization;
    key.has_row_scales = entry.has_row_scales;
    key.has_pair_map = entry.has_pair_map;
    return true;
}

} // namespace

void astc_vulkan_page_owner::reset() {
    prepared_ = false;
    atlas_slots_x_ = 0;
    plan_ = {};
    pages_.clear();
    entry_to_page_.clear();
}

bool astc_vulkan_page_owner::prepare(
        const astc_vulkan_model_cache_plan & plan,
        const std::vector<astc_vulkan_model_cache_storage_page> & pages,
        uint32_t atlas_slots_x, std::string & error) {
    reset();
    if (atlas_slots_x == 0) {
        error = "ASTC page owner requires at least one atlas slot per row";
        return false;
    }
    entry_to_page_.assign(plan.entries.size(), static_cast<size_t>(-1));
    std::unordered_map<std::string, uint32_t> next_slot_by_class;
    pages_.reserve(pages.size());
    for (size_t page_index = 0; page_index < pages.size(); ++page_index) {
        const auto & source = pages[page_index];
        astc_vulkan_page_slot page;
        page.page_index = page_index;
        page.key = source.key;
        page.entry_indices = source.entry_indices;
        page.payload_bytes = source.payload_bytes;
        page.host_bytes = source.host_bytes;
        std::string class_key = std::to_string(static_cast<unsigned>(source.key.footprint)) + ":" +
            std::to_string(static_cast<unsigned>(source.key.representation)) + ":" +
            std::to_string(static_cast<unsigned>(source.key.paired_semantic)) + ":" +
            std::to_string(static_cast<unsigned>(source.key.normalization)) + ":" +
            (source.key.has_row_scales ? "1" : "0");
        const uint32_t class_slot = next_slot_by_class[class_key]++;
        page.slot_index = class_slot;
        page.slot_x = class_slot % atlas_slots_x;
        page.slot_y = class_slot / atlas_slots_x;
        for (const size_t entry_index : source.entry_indices) {
            if (entry_index >= plan.entries.size()) {
                error = "ASTC page references an out-of-range model entry";
                reset();
                return false;
            }
            const auto & entry = plan.entries[entry_index];
            astc_vulkan_model_cache_storage_key entry_storage_key;
            if (!entry_key(entry, entry_storage_key) ||
                !same_key(entry_storage_key, source.key)) {
                error = "ASTC page mixes incompatible storage contracts";
                reset();
                return false;
            }
            if (entry_to_page_[entry_index] != static_cast<size_t>(-1)) {
                error = "ASTC model entry occurs in multiple pages";
                reset();
                return false;
            }
            entry_to_page_[entry_index] = page_index;
        }
        pages_.push_back(std::move(page));
    }
    plan_ = plan;
    atlas_slots_x_ = atlas_slots_x;
    prepared_ = true;
    error.clear();
    return true;
}

bool astc_vulkan_page_owner::set_state(
        size_t page_index, astc_vulkan_page_state expected,
        astc_vulkan_page_state next, std::string & error) {
    if (!prepared_ || page_index >= pages_.size()) {
        error = "ASTC page owner page index is out of range";
        return false;
    }
    astc_vulkan_page_slot & page = pages_[page_index];
    if (page.state != expected) {
        error = "invalid ASTC page lifecycle transition";
        return false;
    }
    page.state = next;
    error.clear();
    return true;
}

bool astc_vulkan_page_owner::begin_upload(size_t page_index, std::string & error) {
    return set_state(page_index, astc_vulkan_page_state::unloaded,
                     astc_vulkan_page_state::uploading, error);
}

bool astc_vulkan_page_owner::finish_upload(size_t page_index, std::string & error) {
    return set_state(page_index, astc_vulkan_page_state::uploading,
                     astc_vulkan_page_state::resident, error);
}

bool astc_vulkan_page_owner::begin_evict(size_t page_index, std::string & error) {
    return set_state(page_index, astc_vulkan_page_state::resident,
                     astc_vulkan_page_state::evicting, error);
}

bool astc_vulkan_page_owner::finish_evict(size_t page_index, std::string & error) {
    return set_state(page_index, astc_vulkan_page_state::evicting,
                     astc_vulkan_page_state::unloaded, error);
}

bool astc_vulkan_page_owner::set_resident_pages(
        const std::vector<size_t> & page_indices, std::string & error) {
    if (!prepared_) {
        error = "ASTC page owner is not prepared";
        return false;
    }
    for (const size_t page_index : page_indices) {
        if (!begin_upload(page_index, error) || !finish_upload(page_index, error)) {
            reset();
            return false;
        }
    }
    error.clear();
    return true;
}

bool astc_vulkan_page_owner::set_resident_prefix(size_t page_count, std::string & error) {
    if (!prepared_ || page_count > pages_.size()) {
        error = "ASTC page owner residency prefix is out of range";
        return false;
    }
    std::vector<size_t> indices;
    indices.reserve(page_count);
    for (size_t index = 0; index < page_count; ++index) indices.push_back(index);
    return set_resident_pages(indices, error);
}

bool astc_vulkan_page_owner::resolve_entry(
        size_t entry_index, astc_vulkan_page_resolve & result, std::string & error) const {
    result = {};
    result.entry_index = entry_index;
    if (!prepared_ || entry_index >= plan_.entries.size()) {
        error = "ASTC page owner entry index is out of range";
        return false;
    }
    const auto & entry = plan_.entries[entry_index];
    if (entry.use_native_fallback) {
        error.clear();
        return true;
    }
    const size_t page_index = entry_to_page_[entry_index];
    if (page_index == static_cast<size_t>(-1) || page_index >= pages_.size()) {
        error = "ASTC model entry has no page assignment";
        return false;
    }
    const auto & page = pages_[page_index];
    result.page_index = page_index;
    result.slot_x = page.slot_x;
    result.slot_y = page.slot_y;
    result.slot_index = page.slot_index;
    result.resident = page.state == astc_vulkan_page_state::resident;
    result.use_native_fallback = !result.resident;
    error.clear();
    return true;
}

bool astc_vulkan_page_owner::resolve_tensor(
        const std::string & tensor_name, astc_vulkan_page_resolve & result,
        std::string & error) const {
    for (size_t index = 0; index < plan_.entries.size(); ++index) {
        if (plan_.entries[index].tensor_name == tensor_name) {
            return resolve_entry(index, result, error);
        }
    }
    result = {};
    error = "ASTC tensor is not present in the page plan: " + tensor_name;
    return false;
}

bool astc_vulkan_page_owner::load_entry_material(
        const astc_vulkan_model_cache_catalog & catalog, size_t entry_index,
        astc_vulkan_page_material & result, std::string & error) const {
    result = {};
    if (!resolve_entry(entry_index, result.resolve, error)) return false;
    if (result.resolve.use_native_fallback) {
        error.clear();
        return true;
    }
    if (entry_index >= catalog.plan.entries.size() ||
        catalog.plan.entries[entry_index].tensor_name != plan_.entries[entry_index].tensor_name ||
        catalog.plan.entries[entry_index].artifact_id != plan_.entries[entry_index].artifact_id) {
        error = "ASTC page material catalog does not match page plan";
        return false;
    }
    const auto & entry = plan_.entries[entry_index];
    if (catalog.validation.source) {
        if (!catalog.validation.source->read_range(
                astc_vulkan_cache_blob_kind::payload,
                entry.storage.byte_offset, entry.storage.byte_size,
                result.payload, error)) return false;
    } else if (!astc_vulkan_read_file_range(catalog.validation.paths.payload,
                                            entry.storage.byte_offset, entry.storage.byte_size,
                                            result.payload, error)) return false;
    if (entry.storage.layout_byte_size != 0) {
        if (catalog.validation.source) {
            if (!catalog.validation.source->read_range(
                    astc_vulkan_cache_blob_kind::layout,
                    entry.storage.layout_byte_offset, entry.storage.layout_byte_size,
                    result.paired_layout, error)) return false;
        } else if (!astc_vulkan_read_file_range(catalog.validation.paths.layout,
                                                entry.storage.layout_byte_offset,
                                                entry.storage.layout_byte_size,
                                                result.paired_layout, error)) return false;
    }
    if (entry.row_scale_byte_size != 0) {
        if (entry.row_scale_byte_size % sizeof(float) != 0 ||
            entry.row_scale_byte_size / sizeof(float) != entry.storage.height) {
            error = "ASTC page row-scale range does not match tensor height";
            return false;
        }
        std::vector<uint8_t> bytes;
        if (catalog.validation.source) {
            if (!catalog.validation.source->read_range(
                    astc_vulkan_cache_blob_kind::row_scales,
                    entry.row_scale_byte_offset, entry.row_scale_byte_size,
                    bytes, error)) return false;
        } else if (!astc_vulkan_read_file_range(catalog.validation.paths.row_scales,
                                                entry.row_scale_byte_offset,
                                                entry.row_scale_byte_size, bytes, error)) return false;
        result.row_scales.resize(entry.storage.height);
        std::memcpy(result.row_scales.data(), bytes.data(), bytes.size());
    }
    if (entry.pair_map_byte_size != 0) {
        if (entry.pair_map_byte_size !=
                static_cast<uint64_t>((entry.storage.height + 9u) / 10u) * 10u) {
            error = "ASTC page pair-map range does not match tensor height";
            return false;
        }
        if (catalog.validation.source) {
            if (!catalog.validation.source->read_range(
                    astc_vulkan_cache_blob_kind::pair_map,
                    entry.pair_map_byte_offset, entry.pair_map_byte_size,
                    result.pair_map, error)) return false;
        } else if (!astc_vulkan_read_file_range(catalog.validation.paths.pair_map,
                                                entry.pair_map_byte_offset,
                                                entry.pair_map_byte_size,
                                                result.pair_map, error)) return false;
    }
    error.clear();
    return true;
}

bool astc_vulkan_page_owner::load_tensor_material(
        const astc_vulkan_model_cache_catalog & catalog,
        const std::string & tensor_name, astc_vulkan_page_material & result,
        std::string & error) const {
    for (size_t index = 0; index < plan_.entries.size(); ++index) {
        if (plan_.entries[index].tensor_name == tensor_name) {
            return load_entry_material(catalog, index, result, error);
        }
    }
    result = {};
    error = "ASTC tensor is not present in the page plan: " + tensor_name;
    return false;
}
