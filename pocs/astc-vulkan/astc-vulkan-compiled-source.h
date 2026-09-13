#pragma once

#include "astc-vulkan-cache.h"
#include "astc-vulkan-compiled-model.h"
#include "astc-vulkan-native-tensor-table.h"

#include "gguf.h"

#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>

enum class astc_vulkan_compiled_source_mode {
    memory_file,
    private_file,
};

// Compiled-model source adapter. V1 supplies an embedded full GGUF through a
// memory file. V2 exposes GGUF metadata directly to llama_model_init_from_user
// and supplies only native residual tensors through the callback below.
class astc_vulkan_compiled_source {
public:
    astc_vulkan_compiled_source() = default;
    ~astc_vulkan_compiled_source();

    astc_vulkan_compiled_source(const astc_vulkan_compiled_source &) = delete;
    astc_vulkan_compiled_source & operator=(const astc_vulkan_compiled_source &) = delete;

    bool open(const std::string & compiled_model_path, std::string & error);
    void reset();

    const std::string & model_path() const { return model_path_; }
    gguf_context * metadata() const { return metadata_; }
    bool uses_user_loader() const { return metadata_ != nullptr; }
    static void set_tensor_data_callback(struct ggml_tensor * tensor, void * userdata);
    bool callback_ok() const { return callback_error_.empty(); }
    const std::string & callback_error() const { return callback_error_; }
    const std::string & source_fingerprint() const { return source_fingerprint_; }
    const std::shared_ptr<const astc_vulkan_cache_source> & cache_source() const {
        return cache_source_;
    }
    astc_vulkan_compiled_source_mode materialization_mode() const { return mode_; }
    const char * materialization_mode_name() const;
    bool ready() const { return (uses_user_loader() || !model_path_.empty()) && cache_source_ != nullptr; }

private:
    void close_model_file();
    bool initialize_v2(std::string & error);
    void set_tensor_data(struct ggml_tensor * tensor);

    std::filesystem::path root_;
    std::string model_path_;
    std::string source_fingerprint_;
    std::shared_ptr<astc_vulkan_compiled_model> model_;
    std::shared_ptr<const astc_vulkan_cache_source> cache_source_;
    astc_vulkan_native_tensor_table native_table_;
    std::unordered_map<std::string, size_t> native_index_;
    gguf_context * metadata_ = nullptr;
    std::string callback_error_;
    int model_fd_ = -1;
    astc_vulkan_compiled_source_mode mode_ = astc_vulkan_compiled_source_mode::private_file;
};
