#include "astc-vulkan-sidecar.h"

#include "astc-vulkan-resource.h"
#include "astc-vulkan-paired-layout.h"

#include <algorithm>
#include <limits>
#include <vector>

astc_vulkan_sidecar::~astc_vulkan_sidecar() {
    reset();
}

void astc_vulkan_sidecar::reset() {
    paired_dispatch_.reset();
    paired_tensor_.reset();
    dispatch_.reset();
    stream_tensor_.reset();
    adapter_.reset();
    shared_device_.reset();
    physical_device_ = VK_NULL_HANDLE;
    device_ = VK_NULL_HANDLE;
    queue_ = VK_NULL_HANDLE;
    queue_family_ = UINT32_MAX;
    binding_ = {};
    dispatch_spirv_.clear();
    dispatch_samples_ = 0;
    native_mode_ = false;
    paired_layout_.clear();
    paired_row_scales_.clear();
    paired_pair_map_.clear();
    paired_semantic_ = astc_vulkan_paired_semantic::direct_rgb;
    memory_budget_ = {};
}

bool astc_vulkan_sidecar::init(astc_vulkan_footprint footprint, std::string & error,
                               bool allow_experimental) {
    auto shared_device = std::make_shared<astc_vulkan_shared_device>();
    if (!shared_device->init(error)) return false;
    return init(std::move(shared_device), footprint, error, allow_experimental);
}

bool astc_vulkan_sidecar::init(std::shared_ptr<astc_vulkan_shared_device> shared_device,
                               astc_vulkan_footprint footprint, std::string & error,
                               bool allow_experimental) {
    reset();
    if (!shared_device || !shared_device->ready()) {
        error = "ASTC Vulkan sidecar requires a ready shared Vulkan device";
        return false;
    }
    if (!astc_vulkan_footprint_is_valid(footprint)) {
        error = "invalid ASTC Vulkan footprint";
        return false;
    }
    if (astc_vulkan_footprint_is_experimental(footprint) && !allow_experimental) {
        error = "experimental ASTC Vulkan footprint requires explicit opt-in";
        return false;
    }
    if (!shared_device->supports(footprint)) {
        error = "shared Vulkan device does not support requested ASTC sampled format";
        return false;
    }
    shared_device_ = std::move(shared_device);
    physical_device_ = shared_device_->physical_device();
    device_ = shared_device_->device();
    queue_ = shared_device_->queue();
    queue_family_ = shared_device_->queue_family();
    memory_budget_ = shared_device_->memory_budget();
    footprint_ = footprint;
    error.clear();
    return true;
}

bool astc_vulkan_sidecar::init_borrowed(
        VkPhysicalDevice physical_device, VkDevice device, VkQueue queue,
        uint32_t queue_family, astc_vulkan_footprint footprint,
        std::string & error, bool allow_experimental) {
    auto shared_device = std::make_shared<astc_vulkan_shared_device>();
    if (!shared_device->init_borrowed(physical_device, device, queue, queue_family, error)) {
        return false;
    }
    return init(std::move(shared_device), footprint, error, allow_experimental);
}

bool astc_vulkan_sidecar::load_manifest(const std::string & path, std::string & error) {
    astc_vulkan_manifest loaded;
    if (!astc_vulkan_read_manifest(path, loaded, error)) return false;
    return set_manifest(loaded, error);
}

bool astc_vulkan_sidecar::set_manifest(const astc_vulkan_manifest & manifest,
                                       std::string & error) {
    if (!astc_vulkan_validate_manifest(manifest, error)) return false;
    dispatch_.reset();
    paired_dispatch_.reset();
    paired_tensor_.reset();
    stream_tensor_.reset();
    adapter_.reset();
    binding_ = {};
    dispatch_spirv_.clear();
    dispatch_samples_ = 0;
    manifest_ = manifest;
    error.clear();
    return true;
}

bool astc_vulkan_sidecar::bind_tensor(
        const std::string & tensor_name, uint32_t expected_columns, uint32_t expected_rows,
        const std::vector<uint8_t> & payload, astc_vulkan_ffn_binding & binding,
        std::string & error, const std::vector<uint8_t> & paired_layout,
        astc_vulkan_paired_semantic paired_semantic, const std::vector<float> & row_scales,
        const std::vector<uint8_t> & pair_map) {
    if (!ready()) {
        error = "ASTC Vulkan sidecar is not initialized";
        return false;
    }
    dispatch_.reset();
    paired_dispatch_.reset();
    paired_tensor_.reset();
    paired_layout_.clear();
    paired_row_scales_.clear();
    paired_pair_map_.clear();
    paired_semantic_ = astc_vulkan_paired_semantic::direct_rgb;
    adapter_.reset();
    binding_ = {};
    const astc_vulkan_tensor_record * record = astc_vulkan_find_tensor(manifest_, tensor_name);
    if (record != nullptr && record->representation == astc_vulkan_representation::kPairedD2) {
        binding = {};
        binding.record = *record;
        if (expected_columns == 0 || record->width != expected_columns ||
            (expected_rows != 0 && record->height != expected_rows)) {
            binding.fallback_reason = "paired-D2 tensor shape does not match model shape";
            error.clear();
            return true;
        }
        if (!astc_vulkan_validate_payload(*record, payload.data(), payload.size(), error) ||
            !astc_vulkan_validate_layout_map(*record, paired_layout.data(), paired_layout.size(), error)) {
            return false;
        }
        if (!row_scales.empty() && row_scales.size() != record->height) {
            error = "paired-D2 row-scale count does not match tensor height";
            return false;
        }
        if (!pair_map.empty() && pair_map.size() != static_cast<size_t>((record->height + 9u) / 10u) * 10u) {
            error = "paired-D2 pair-map size does not match tensor height";
            return false;
        }
        const uint32_t storage_height = astc_vulkan_paired_storage_height(record->height);
        const VkFormat format = astc_vulkan_vk_format(static_cast<uint8_t>(footprint_));
        uint64_t image_bytes = 0;
        uint64_t staging_bytes = 0;
        uint64_t host_bytes = 0;
        if (record->footprint != footprint_ ||
            !astc_vulkan_sampled_image_memory_requirement(device_, format, record->width, storage_height, image_bytes) ||
            !astc_vulkan_upload_staging_memory_requirement(physical_device_, device_, payload.size(), staging_bytes) ||
            image_bytes > std::numeric_limits<uint64_t>::max() - staging_bytes ||
            payload.size() > std::numeric_limits<uint64_t>::max() - staging_bytes ||
            (host_bytes = static_cast<uint64_t>(payload.size()) + staging_bytes) >
                std::numeric_limits<uint64_t>::max() - paired_layout.size() ||
            paired_layout.size() > std::numeric_limits<uint64_t>::max() -
                static_cast<uint64_t>(row_scales.size()) * sizeof(float) ||
            !astc_vulkan_budget_can_reserve(memory_budget_, 0, image_bytes + staging_bytes,
                                            host_bytes + static_cast<uint64_t>(paired_layout.size()) +
                                                static_cast<uint64_t>(row_scales.size()) * sizeof(float), error)) {
            binding.fallback_reason = error.empty() ? "paired-D2 Vulkan memory budget denied upload" : error;
            error.clear();
            return true;
        }
        const astc_vulkan_reconstruction reconstruction{record->scale_l, 0.0f, record->offset, 0.0f};
        if (!paired_tensor_.upload(physical_device_, device_, queue_, queue_family_, *record,
                                   reconstruction, payload, error, storage_height)) {
            return false;
        }
        binding.status = astc_vulkan_binding_status::kReady;
        binding.reconstruction = reconstruction;
        paired_layout_ = paired_layout;
        paired_row_scales_ = row_scales;
        paired_pair_map_ = pair_map;
        paired_semantic_ = paired_semantic;
        binding_ = binding;
        error.clear();
        return true;
    }
    if (!adapter_.prepare(manifest_, tensor_name, expected_columns, true,
                          expected_rows, binding, error)) return false;
    if (binding.status != astc_vulkan_binding_status::kReady) return true;
    if (binding.record.footprint != footprint_) {
        binding.status = astc_vulkan_binding_status::kFallback;
        binding.fallback_reason = "tensor footprint does not match sidecar format";
        error.clear();
        return true;
    }
    const VkFormat format = astc_vulkan_vk_format(static_cast<uint8_t>(footprint_));
    uint64_t image_bytes = 0;
    uint64_t staging_bytes = 0;
    if (!astc_vulkan_sampled_image_memory_requirement(device_, format,
                                                       binding.record.width, binding.record.height,
                                                       image_bytes) ||
        !astc_vulkan_upload_staging_memory_requirement(physical_device_, device_, payload.size(),
                                                        staging_bytes) ||
        image_bytes > std::numeric_limits<uint64_t>::max() - staging_bytes ||
        payload.size() > std::numeric_limits<uint64_t>::max() - staging_bytes ||
        !astc_vulkan_budget_can_reserve(memory_budget_, 0, image_bytes + staging_bytes,
                                        static_cast<uint64_t>(payload.size()) + staging_bytes, error)) {
        binding.status = astc_vulkan_binding_status::kFallback;
        binding.fallback_reason = error.empty() ? "ASTC Vulkan memory budget denied upload" : error;
        error.clear();
        return true;
    }
    if (!adapter_.upload(physical_device_, device_, queue_, queue_family_,
                         binding, payload, error)) return false;
    binding_ = binding;
    error.clear();
    return true;
}

bool astc_vulkan_sidecar::run(const std::vector<uint32_t> & spirv,
                              const std::vector<float> & activations,
                              std::vector<float> & output, std::string & error) {
    if (!ready() || binding_.status != astc_vulkan_binding_status::kReady) {
        error = "ASTC Vulkan sidecar has no ready tensor";
        return false;
    }
    if (native_mode_) {
        dispatch_.reset();
        paired_dispatch_.reset();
        dispatch_spirv_.clear();
        dispatch_samples_ = 0;
        native_mode_ = false;
    }
    if (activations.empty() || activations.size() % binding_.record.width != 0) {
        error = "ASTC Vulkan sidecar activation count is not divisible by tensor width";
        return false;
    }
    const uint32_t samples = static_cast<uint32_t>(
        activations.size() / binding_.record.width);
    if (binding_.record.representation == astc_vulkan_representation::kPairedD2) {
        if (!paired_dispatch_.ready() || dispatch_samples_ != samples || dispatch_spirv_ != spirv) {
            if (!paired_dispatch_.init(physical_device_, device_, queue_, queue_family_, paired_tensor_,
                                      paired_layout_, spirv, binding_.record.width,
                                      binding_.record.height, samples, error, paired_semantic_,
                                      paired_row_scales_, paired_pair_map_)) return false;
            dispatch_spirv_ = spirv;
            dispatch_samples_ = samples;
        }
        return paired_dispatch_.run(activations, binding_.reconstruction, output, error);
    }
    if (!dispatch_.ready() || dispatch_samples_ != samples || dispatch_spirv_ != spirv) {
        if (!dispatch_.init(physical_device_, device_, queue_, queue_family_,
                            adapter_.session(), spirv, binding_.record.width,
                            binding_.record.height, samples, error)) return false;
        dispatch_spirv_ = spirv;
        dispatch_samples_ = samples;
    }
    return dispatch_.run(activations, binding_.reconstruction, output, error);
}

bool astc_vulkan_sidecar::record_native(
        const std::vector<uint32_t> & spirv, VkDevice native_device,
        VkCommandBuffer command_buffer, VkBuffer activation_buffer,
        VkDeviceSize activation_offset, VkDeviceSize activation_size,
        VkBuffer output_buffer, VkDeviceSize output_offset,
        VkDeviceSize output_size, uint32_t samples,
        const astc_vulkan_reconstruction & reconstruction,
        uint32_t row_base, uint32_t band_height, std::string & error) {
    if (!ready() || binding_.status != astc_vulkan_binding_status::kReady ||
        native_device == VK_NULL_HANDLE || native_device != device_ ||
        command_buffer == VK_NULL_HANDLE || activation_buffer == VK_NULL_HANDLE ||
        output_buffer == VK_NULL_HANDLE || samples == 0 || spirv.empty()) {
        error = "ASTC native dispatch device or resource contract mismatch";
        return false;
    }
    if (binding_.record.representation == astc_vulkan_representation::kPairedD2) {
        if (!paired_dispatch_.ready() || !native_mode_ || dispatch_samples_ != samples ||
            dispatch_spirv_ != spirv) {
            paired_dispatch_.reset();
            if (!paired_dispatch_.init_native(
                    physical_device_, device_, queue_, queue_family_, paired_tensor_,
                    paired_layout_, spirv, binding_.record.width, binding_.record.height,
                    samples, error, paired_semantic_, paired_row_scales_, paired_pair_map_)) {
                return false;
            }
            dispatch_spirv_ = spirv;
            dispatch_samples_ = samples;
        }
        native_mode_ = true;
        return paired_dispatch_.record_external(
            command_buffer, activation_buffer, activation_offset, activation_size,
            output_buffer, output_offset, output_size, reconstruction,
            row_base, band_height, error);
    }
    if (!dispatch_.ready() || !native_mode_ || dispatch_samples_ != samples ||
        dispatch_spirv_ != spirv) {
        dispatch_.reset();
        if (!dispatch_.init_native(
                physical_device_, device_, queue_, queue_family_, adapter_.session(),
                spirv, binding_.record.width, binding_.record.height, samples, error)) {
            return false;
        }
        dispatch_spirv_ = spirv;
        dispatch_samples_ = samples;
    }
    native_mode_ = true;
    return dispatch_.record_external(
        command_buffer, activation_buffer, activation_offset, activation_size,
        output_buffer, output_offset, output_size, reconstruction,
        row_base, band_height, error);
}

bool astc_vulkan_sidecar::run_streamed(
        const std::string & payload_path, uint64_t payload_offset,
        uint64_t max_resident_payload_bytes, const std::vector<uint32_t> & spirv,
        const std::vector<float> & activations, std::vector<float> & output,
        std::string & error) {
    if (!ready() || binding_.status != astc_vulkan_binding_status::kReady ||
        payload_path.empty() || spirv.empty() || activations.empty() ||
        activations.size() % binding_.record.width != 0) {
        error = "ASTC Vulkan sidecar streamed run has invalid inputs";
        return false;
    }
    const bool paired = binding_.record.representation == astc_vulkan_representation::kPairedD2;
    astc_vulkan_stream_geometry geometry;
    if (!astc_vulkan_make_stream_geometry(footprint_, binding_.record.width,
                                          binding_.record.height, paired, geometry, error)) {
        return false;
    }
    std::vector<astc_vulkan_stream_band> bands;
    if (!astc_vulkan_plan_stream(geometry, max_resident_payload_bytes, bands, error)) return false;
    astc_vulkan_stream_payload_reader reader;
    if (!reader.open(payload_path, payload_offset, geometry, error)) return false;

    const uint32_t samples = static_cast<uint32_t>(activations.size() / binding_.record.width);
    output.assign(static_cast<size_t>(samples) * binding_.record.height, 0.0f);
    // A streamed execution cannot retain a descriptor pointing at the
    // resident full-tensor image. Tear down only the image/session objects;
    // the sidecar device and immutable dispatch inputs remain reusable.
    dispatch_.reset();
    paired_dispatch_.reset();
    stream_tensor_.reset();
    paired_tensor_.reset();
    dispatch_spirv_.clear();
    dispatch_samples_ = 0;

    std::vector<uint8_t> band_payload;
    std::vector<float> band_output;
    for (const auto & band : bands) {
        if (!reader.read_band(band, band_payload, error)) return false;
        if (paired) {
            if (!paired_tensor_.upload_band(physical_device_, device_, queue_, queue_family_,
                                            binding_.record, binding_.reconstruction,
                                            geometry, band, band_payload, error)) return false;
            if (!paired_dispatch_.ready()) {
                if (!paired_dispatch_.init(physical_device_, device_, queue_, queue_family_,
                                           paired_tensor_, paired_layout_, spirv,
                                           binding_.record.width, binding_.record.height, samples,
                                           error, paired_semantic_, paired_row_scales_, paired_pair_map_,
                                           band.physical_height, band.physical_y)) return false;
            } else if (!paired_dispatch_.rebind_texture(paired_tensor_, band.physical_height,
                                                        band.physical_y, error)) {
                return false;
            }
            if (!paired_dispatch_.run_band(activations, binding_.reconstruction,
                                           band.logical_row_base, band.logical_row_count,
                                           band_output, error)) return false;
        } else {
            if (!stream_tensor_.upload_band(physical_device_, device_, queue_, queue_family_,
                                            binding_.record, binding_.reconstruction,
                                            geometry, band, band_payload, error)) return false;
            if (!dispatch_.ready()) {
                if (!dispatch_.init(physical_device_, device_, queue_, queue_family_,
                                    stream_tensor_, spirv, binding_.record.width,
                                    binding_.record.height, samples, error)) return false;
            } else if (!dispatch_.rebind_texture(stream_tensor_, error)) {
                return false;
            }
            if (!dispatch_.run_band(activations, binding_.reconstruction,
                                    band.logical_row_base, band.logical_row_count,
                                    band_output, error)) return false;
        }
        if (band_output.size() != static_cast<size_t>(samples) * band.logical_row_count) {
            error = "ASTC Vulkan sidecar streamed dispatch returned an invalid band";
            return false;
        }
        for (uint32_t sample = 0; sample < samples; ++sample) {
            std::copy_n(band_output.begin() + static_cast<size_t>(sample) * band.logical_row_count,
                        band.logical_row_count,
                        output.begin() + static_cast<size_t>(sample) * binding_.record.height +
                            band.logical_row_base);
        }
    }
    error.clear();
    return true;
}
