#pragma once

#include <vulkan/vulkan.h>
#include "astc-vulkan-driver.h"
#include "astc-vulkan-stream-loader.h"

#include <cstdint>
#include <string>
#include <vector>

struct astc_vulkan_image_resources {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkSampler sampler = VK_NULL_HANDLE;
};

uint32_t astc_vulkan_find_memory_type(VkPhysicalDevice physical_device,
                                      uint32_t type_bits,
                                      VkMemoryPropertyFlags properties);
bool astc_vulkan_supports_sampled_transfer(VkPhysicalDevice physical_device,
                                            VkFormat format);
bool astc_vulkan_supports_sampled_transfer_extent(VkPhysicalDevice physical_device,
                                                   VkFormat format,
                                                   uint32_t width, uint32_t height);
bool astc_vulkan_create_sampled_image(VkPhysicalDevice physical_device,
                                       VkDevice device, VkFormat format,
                                       uint32_t width, uint32_t height,
                                       astc_vulkan_image_resources & resources);
// Returns the allocation requirement for the exact sampled image created by
// astc_vulkan_create_sampled_image, without allocating or uploading it.
bool astc_vulkan_sampled_image_memory_requirement(VkDevice device, VkFormat format,
                                                   uint32_t width, uint32_t height,
                                                   uint64_t & bytes);
// Returns the allocation requirement of the host-visible transfer buffer used
// by upload(). The result includes Vulkan alignment, not only payload bytes.
bool astc_vulkan_upload_staging_memory_requirement(VkPhysicalDevice physical_device,
                                                   VkDevice device, uint64_t payload_bytes,
                                                   uint64_t & bytes);
void astc_vulkan_destroy_sampled_image(VkDevice device,
                                       astc_vulkan_image_resources & resources);

VkFormat astc_vulkan_vk_format(uint8_t footprint);

// Owns one sampled ASTC image. The first runtime integration intentionally
// uses one tensor per image; atlas placement can be added without changing the
// upload or descriptor ownership contract.
class astc_vulkan_texture {
public:
    astc_vulkan_texture() = default;
    ~astc_vulkan_texture();
    astc_vulkan_texture(const astc_vulkan_texture &) = delete;
    astc_vulkan_texture & operator=(const astc_vulkan_texture &) = delete;

    bool upload(VkPhysicalDevice physical_device, VkDevice device, VkQueue queue,
                uint32_t queue_family, uint8_t footprint, uint32_t width,
                uint32_t height, const std::vector<uint8_t> & payload,
                std::string & error);
    // Replaces the contents of an existing image with an identically-sized
    // ASTC payload. Image/view/sampler handles remain stable for descriptor
    // reuse in streamed offline batches.
    bool update_payload(VkPhysicalDevice physical_device, VkDevice device, VkQueue queue,
                        uint32_t queue_family, uint8_t footprint,
                        const std::vector<uint8_t> & payload, std::string & error);
    bool upload_band(VkPhysicalDevice physical_device, VkDevice device, VkQueue queue,
                     uint32_t queue_family, const astc_vulkan_stream_geometry & geometry,
                     const astc_vulkan_stream_band & band,
                     const std::vector<uint8_t> & payload, std::string & error);
    void reset();
    VkImageView view() const { return resources_.view; }
    VkSampler sampler() const { return resources_.sampler; }
    uint32_t width() const { return width_; }
    uint32_t height() const { return height_; }

private:
    VkDevice device_ = VK_NULL_HANDLE;
    astc_vulkan_image_resources resources_{};
    uint8_t footprint_ = 0xff;
    uint32_t width_ = 0;
    uint32_t height_ = 0;
};

// Binds one validated manifest record to one uploaded sampled image. The
// shader consumes reconstruction separately, keeping storage and math policy
// independent and allowing L+A or scalar modes without changing the loader.
class astc_vulkan_tensor_session {
public:
    bool upload(VkPhysicalDevice physical_device, VkDevice device, VkQueue queue,
                uint32_t queue_family, const astc_vulkan_tensor_record & record,
                const astc_vulkan_reconstruction & reconstruction,
                const std::vector<uint8_t> & payload, std::string & error,
                uint32_t storage_height = 0);
    bool upload_band(VkPhysicalDevice physical_device, VkDevice device, VkQueue queue,
                     uint32_t queue_family, const astc_vulkan_tensor_record & record,
                     const astc_vulkan_reconstruction & reconstruction,
                     const astc_vulkan_stream_geometry & geometry,
                     const astc_vulkan_stream_band & band,
                     const std::vector<uint8_t> & payload, std::string & error);
    void reset() { texture_.reset(); record_ = {}; reconstruction_ = {}; }
    const astc_vulkan_texture & texture() const { return texture_; }
    const astc_vulkan_tensor_record & record() const { return record_; }
    const astc_vulkan_reconstruction & reconstruction() const { return reconstruction_; }

private:
    astc_vulkan_texture texture_;
    astc_vulkan_tensor_record record_{};
    astc_vulkan_reconstruction reconstruction_{};
};
