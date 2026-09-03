#include "astc-vulkan-ffn-adapter.h"

bool astc_vulkan_ffn_adapter::prepare(
        const astc_vulkan_manifest & manifest, const std::string & tensor_name,
        uint32_t expected_columns, bool sampled_astc_supported, uint32_t expected_rows,
        astc_vulkan_ffn_binding & binding, std::string & error) const {
    if (!astc_vulkan_validate_manifest(manifest, error)) return false;
    binding = {};
    const astc_vulkan_tensor_record * record = astc_vulkan_find_tensor(manifest, tensor_name);
    if (record == nullptr) {
        binding.fallback_reason = "FFN tensor is absent from ASTC manifest";
        error.clear();
        return true;
    }
    if (manifest.version == 1) {
        binding.record = *record;
        binding.fallback_reason = "manifest v1 lacks runtime reconstruction metadata";
        error.clear();
        return true;
    }
    if (record->representation != astc_vulkan_representation::kScalar &&
        record->representation != astc_vulkan_representation::kGaugeLumaAlpha) {
        binding.record = *record;
        binding.fallback_reason = "FFN tensor representation requires the normal llama fallback";
        error.clear();
        return true;
    }
    if (expected_columns == 0 || record->width != expected_columns ||
        (expected_rows != 0 && record->height != expected_rows)) {
        binding.record = *record;
        binding.fallback_reason = "FFN tensor shape does not match model shape";
        error.clear();
        return true;
    }
    if (!sampled_astc_supported) {
        binding.record = *record;
        binding.fallback_reason = "device does not support sampled ASTC";
        error.clear();
        return true;
    }
    binding.status = astc_vulkan_binding_status::kReady;
    binding.record = *record;
    binding.reconstruction = {record->scale_l, record->scale_a, record->offset, 0.0f};
    error.clear();
    return true;
}

bool astc_vulkan_ffn_adapter::upload(
        VkPhysicalDevice physical_device, VkDevice device, VkQueue queue,
        uint32_t queue_family, const astc_vulkan_ffn_binding & binding,
        const std::vector<uint8_t> & payload, std::string & error) {
    if (binding.status != astc_vulkan_binding_status::kReady) {
        error = "ASTC FFN binding requires normal llama fallback";
        return false;
    }
    return session_.upload(physical_device, device, queue, queue_family,
                           binding.record, binding.reconstruction, payload, error);
}
