#pragma once

#include "astc-vulkan-model-cache.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// Runtime-local page lifecycle. The first owner implementation keeps a
// static residency set; uploading/eviction states are explicit so a later
// asynchronous Vulkan owner can reuse the same contract without changing the
// scheduler or manifest.
enum class astc_vulkan_page_state : uint8_t {
    unloaded,
    uploading,
    resident,
    evicting,
};

struct astc_vulkan_page_slot {
    size_t page_index = 0;
    astc_vulkan_model_cache_storage_key key{};
    std::vector<size_t> entry_indices;
    uint64_t payload_bytes = 0;
    uint64_t host_bytes = 0;
    // Virtual block-slot coordinates. A future VkImage owner translates
    // these to texel coordinates after choosing the actual atlas extent.
    uint32_t slot_x = 0;
    uint32_t slot_y = 0;
    uint32_t slot_index = 0;
    astc_vulkan_page_state state = astc_vulkan_page_state::unloaded;
};

struct astc_vulkan_page_resolve {
    size_t page_index = static_cast<size_t>(-1);
    size_t entry_index = static_cast<size_t>(-1);
    uint32_t slot_x = 0;
    uint32_t slot_y = 0;
    uint32_t slot_index = 0;
    bool resident = false;
    bool use_native_fallback = true;
};

struct astc_vulkan_page_material {
    astc_vulkan_page_resolve resolve;
    std::vector<uint8_t> payload;
    std::vector<uint8_t> paired_layout;
    std::vector<float> row_scales;
};

// A thin runtime owner above the offline page planner. It validates the
// planner output, assigns deterministic per-storage-class slots and answers
// tensor/entry residency queries. It deliberately does not allocate Vulkan
// images or rewrite payloads; those remain in resource/tensor_session code.
class astc_vulkan_page_owner {
public:
    bool prepare(const astc_vulkan_model_cache_plan & plan,
                 const std::vector<astc_vulkan_model_cache_storage_page> & pages,
                 uint32_t atlas_slots_x, std::string & error);

    bool set_resident_pages(const std::vector<size_t> & page_indices,
                            std::string & error);
    bool set_resident_prefix(size_t page_count, std::string & error);

    bool begin_upload(size_t page_index, std::string & error);
    bool finish_upload(size_t page_index, std::string & error);
    bool begin_evict(size_t page_index, std::string & error);
    bool finish_evict(size_t page_index, std::string & error);

    bool resolve_entry(size_t entry_index, astc_vulkan_page_resolve & result,
                       std::string & error) const;
    bool resolve_tensor(const std::string & tensor_name,
                        astc_vulkan_page_resolve & result,
                        std::string & error) const;
    // Reads exactly one resident entry from the validated catalog. The
    // caller can pass the returned material directly to tensor_session or
    // sidecar; non-resident/native entries return a successful fallback
    // result with empty buffers.
    bool load_entry_material(const astc_vulkan_model_cache_catalog & catalog,
                             size_t entry_index,
                             astc_vulkan_page_material & result,
                             std::string & error) const;
    bool load_tensor_material(const astc_vulkan_model_cache_catalog & catalog,
                              const std::string & tensor_name,
                              astc_vulkan_page_material & result,
                              std::string & error) const;

    void reset();
    bool ready() const { return prepared_; }
    const std::vector<astc_vulkan_page_slot> & pages() const { return pages_; }
    const astc_vulkan_model_cache_plan & plan() const { return plan_; }

private:
    bool set_state(size_t page_index, astc_vulkan_page_state expected,
                   astc_vulkan_page_state next, std::string & error);

    bool prepared_ = false;
    uint32_t atlas_slots_x_ = 0;
    astc_vulkan_model_cache_plan plan_{};
    std::vector<astc_vulkan_page_slot> pages_;
    std::vector<size_t> entry_to_page_;
};
