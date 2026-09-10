#pragma once

#include <filesystem>
#include <string>

// Compatibility source for the first compiled-model runtime path. The
// existing llama loader and ASTC overlay are file-oriented, so this adapter
// materializes validated ASTCM001 sections in a private RAII-owned directory.
class astc_vulkan_compiled_source {
public:
    astc_vulkan_compiled_source() = default;
    ~astc_vulkan_compiled_source();

    astc_vulkan_compiled_source(const astc_vulkan_compiled_source &) = delete;
    astc_vulkan_compiled_source & operator=(const astc_vulkan_compiled_source &) = delete;

    bool open(const std::string & compiled_model_path, std::string & error);
    void reset();

    const std::string & model_path() const { return model_path_; }
    const std::string & cache_path() const { return cache_path_; }
    const std::string & source_fingerprint() const { return source_fingerprint_; }
    bool ready() const { return !model_path_.empty() && !cache_path_.empty(); }

private:
    std::filesystem::path root_;
    std::string model_path_;
    std::string cache_path_;
    std::string source_fingerprint_;
};

