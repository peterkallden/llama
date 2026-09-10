#pragma once

#include <filesystem>
#include <string>

enum class astc_vulkan_compiled_source_mode {
    memory_file,
    private_file,
};

// Compatibility source for the first compiled-model runtime path. The
// existing llama loader and ASTC overlay are file-oriented: on Linux the
// embedded GGUF is exposed through a RAII-owned memory file, while ASTC
// sections are materialized in a private RAII-owned directory.
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
    astc_vulkan_compiled_source_mode materialization_mode() const { return mode_; }
    const char * materialization_mode_name() const;
    bool ready() const { return !model_path_.empty() && !cache_path_.empty(); }

private:
    void close_model_file();

    std::filesystem::path root_;
    std::string model_path_;
    std::string cache_path_;
    std::string source_fingerprint_;
    int model_fd_ = -1;
    astc_vulkan_compiled_source_mode mode_ = astc_vulkan_compiled_source_mode::private_file;
};
