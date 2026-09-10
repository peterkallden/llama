#pragma once

#include "astc-vulkan-cache.h"
#include "astc-vulkan-compiled-model.h"

#include <filesystem>
#include <memory>
#include <string>

enum class astc_vulkan_compiled_source_mode {
    memory_file,
    private_file,
};

// Compiled-model source adapter. The normal llama loader still receives a
// path for the embedded GGUF (a Linux memfd when available), while ASTC
// sections are consumed directly from the container-owned memory blobs.
class astc_vulkan_compiled_source {
public:
    astc_vulkan_compiled_source() = default;
    ~astc_vulkan_compiled_source();

    astc_vulkan_compiled_source(const astc_vulkan_compiled_source &) = delete;
    astc_vulkan_compiled_source & operator=(const astc_vulkan_compiled_source &) = delete;

    bool open(const std::string & compiled_model_path, std::string & error);
    void reset();

    const std::string & model_path() const { return model_path_; }
    const std::string & source_fingerprint() const { return source_fingerprint_; }
    const std::shared_ptr<const astc_vulkan_cache_source> & cache_source() const {
        return cache_source_;
    }
    astc_vulkan_compiled_source_mode materialization_mode() const { return mode_; }
    const char * materialization_mode_name() const;
    bool ready() const { return !model_path_.empty() && cache_source_ != nullptr; }

private:
    void close_model_file();

    std::filesystem::path root_;
    std::string model_path_;
    std::string source_fingerprint_;
    std::shared_ptr<astc_vulkan_compiled_model> model_;
    std::shared_ptr<const astc_vulkan_cache_source> cache_source_;
    int model_fd_ = -1;
    astc_vulkan_compiled_source_mode mode_ = astc_vulkan_compiled_source_mode::private_file;
};
