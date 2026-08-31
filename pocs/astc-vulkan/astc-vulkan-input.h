#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct ggml_vk_astc_loaded_matrix {
    uint32_t rows = 0;
    uint32_t columns = 0;
    std::vector<float> values;
};

struct ggml_vk_astc_tensor_info {
    std::string name;
    uint32_t columns = 0;
    uint32_t rows = 0;
    uint32_t type = 0;
    size_t bytes = 0;
    bool rank2 = false;
};

struct ggml_vk_astc_activation_trace {
    uint32_t samples = 0;
    uint32_t columns = 0;
    std::vector<float> values;
};

bool ggml_vk_astc_load_gguf_matrix(const std::string & path,
                                   const std::string & tensor_name,
                                   ggml_vk_astc_loaded_matrix & matrix,
                                   std::string & error);

bool ggml_vk_astc_list_gguf_tensors(const std::string & path,
                                    std::vector<ggml_vk_astc_tensor_info> & tensors,
                                    std::string & error);

// Private PoC trace format: four little-endian uint32 fields (magic, version,
// sample count, column count), followed by sample_count * column_count F32s.
bool ggml_vk_astc_load_activation_trace(const std::string & path,
                                        ggml_vk_astc_activation_trace & trace,
                                        std::string & error);

bool ggml_vk_astc_write_activation_trace(const std::string & path,
                                         const ggml_vk_astc_activation_trace & trace,
                                         std::string & error);
