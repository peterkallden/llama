#include "astc-vulkan-input.h"

#include "ggml.h"
#include "gguf.h"

#include <cstdio>
#include <cstring>
#include <fstream>

namespace {

constexpr uint32_t kTraceMagic = 0x43545341; // "ASTC" in little endian.
constexpr uint32_t kTraceVersion = 1;

bool read_exact(FILE * file, void * data, size_t size) {
    return std::fread(data, 1, size, file) == size;
}

} // namespace

bool ggml_vk_astc_load_gguf_matrix(const std::string & path,
                                   const std::string & tensor_name,
                                   ggml_vk_astc_loaded_matrix & matrix,
                                   std::string & error) {
    gguf_init_params params{};
    params.no_alloc = true;
    gguf_context * context = gguf_init_from_file(path.c_str(), params);
    if (context == nullptr) {
        error = "failed to open GGUF: " + path;
        return false;
    }
    const int64_t tensor_id = gguf_find_tensor(context, tensor_name.c_str());
    if (tensor_id < 0) {
        error = "tensor not found: " + tensor_name;
        gguf_free(context);
        return false;
    }
    const int64_t * dimensions = gguf_get_tensor_ne(context, tensor_id);
    // GGUF exposes a fixed-width dimension array; dimensions at and above the
    // tensor rank are set to one.  Reject rank > 2 without relying on an API
    // that is not part of gguf.h's public reader surface.
    if (dimensions == nullptr || dimensions[0] <= 0 || dimensions[1] <= 0 ||
        dimensions[2] != 1) {
        error = "tensor must be a non-empty rank-2 matrix";
        gguf_free(context);
        return false;
    }
    const ggml_type type = gguf_get_tensor_type(context, tensor_id);
    if (type != GGML_TYPE_F32 && type != GGML_TYPE_F16) {
        error = "only F32 and F16 GGUF tensors are supported by the PoC reader";
        gguf_free(context);
        return false;
    }
    const size_t value_count = static_cast<size_t>(dimensions[0]) * dimensions[1];
    const size_t element_size = type == GGML_TYPE_F32 ? sizeof(float) : sizeof(ggml_fp16_t);
    if (gguf_get_tensor_size(context, tensor_id) != value_count * element_size) {
        error = "tensor byte size does not match its dimensions";
        gguf_free(context);
        return false;
    }

    FILE * file = std::fopen(path.c_str(), "rb");
    if (file == nullptr) {
        error = "failed to reopen GGUF data: " + path;
        gguf_free(context);
        return false;
    }
    const size_t offset = gguf_get_data_offset(context) + gguf_get_tensor_offset(context, tensor_id);
    const bool seek_ok = std::fseek(file, static_cast<long>(offset), SEEK_SET) == 0;
    if (!seek_ok) {
        error = "failed to seek to GGUF tensor data";
        std::fclose(file);
        gguf_free(context);
        return false;
    }

    matrix.rows = static_cast<uint32_t>(dimensions[1]);
    matrix.columns = static_cast<uint32_t>(dimensions[0]);
    matrix.values.resize(value_count);
    bool read_ok = false;
    if (type == GGML_TYPE_F32) {
        read_ok = read_exact(file, matrix.values.data(), value_count * sizeof(float));
    } else {
        std::vector<ggml_fp16_t> values(value_count);
        read_ok = read_exact(file, values.data(), value_count * sizeof(ggml_fp16_t));
        if (read_ok) {
            ggml_fp16_to_fp32_row(values.data(), matrix.values.data(), static_cast<int64_t>(value_count));
        }
    }
    std::fclose(file);
    gguf_free(context);
    if (!read_ok) {
        error = "failed to read GGUF tensor data";
        matrix = {};
        return false;
    }
    return true;
}

bool ggml_vk_astc_list_gguf_tensors(const std::string & path,
                                    std::vector<ggml_vk_astc_tensor_info> & tensors,
                                    std::string & error) {
    gguf_init_params params{};
    params.no_alloc = true;
    gguf_context * context = gguf_init_from_file(path.c_str(), params);
    if (context == nullptr) {
        error = "failed to open GGUF: " + path;
        return false;
    }
    tensors.clear();
    const int64_t count = gguf_get_n_tensors(context);
    tensors.reserve(count > 0 ? static_cast<size_t>(count) : 0);
    for (int64_t tensor_id = 0; tensor_id < count; ++tensor_id) {
        const int64_t * dimensions = gguf_get_tensor_ne(context, tensor_id);
        if (dimensions == nullptr || dimensions[0] <= 0 || dimensions[1] <= 0) {
            error = "GGUF tensor has invalid dimensions";
            gguf_free(context);
            tensors.clear();
            return false;
        }
        ggml_vk_astc_tensor_info info;
        info.name = gguf_get_tensor_name(context, tensor_id);
        info.columns = static_cast<uint32_t>(dimensions[0]);
        info.rows = static_cast<uint32_t>(dimensions[1]);
        info.rank2 = dimensions[2] == 1;
        info.type = static_cast<uint32_t>(gguf_get_tensor_type(context, tensor_id));
        info.bytes = gguf_get_tensor_size(context, tensor_id);
        tensors.push_back(std::move(info));
    }
    gguf_free(context);
    return true;
}

bool ggml_vk_astc_load_activation_trace(const std::string & path,
                                        ggml_vk_astc_activation_trace & trace,
                                        std::string & error) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        error = "failed to open activation trace: " + path;
        return false;
    }
    uint32_t header[4]{};
    file.read(reinterpret_cast<char *>(header), sizeof(header));
    if (!file || header[0] != kTraceMagic || header[1] != kTraceVersion ||
        header[2] == 0 || header[3] == 0) {
        error = "invalid activation trace header";
        return false;
    }
    const size_t value_count = static_cast<size_t>(header[2]) * header[3];
    trace.samples = header[2];
    trace.columns = header[3];
    trace.values.resize(value_count);
    file.read(reinterpret_cast<char *>(trace.values.data()),
              static_cast<std::streamsize>(value_count * sizeof(float)));
    if (!file) {
        error = "activation trace ended before all values were read";
        trace = {};
        return false;
    }
    return true;
}

bool ggml_vk_astc_write_activation_trace(const std::string & path,
                                         const ggml_vk_astc_activation_trace & trace,
                                         std::string & error) {
    const size_t value_count = static_cast<size_t>(trace.samples) * trace.columns;
    if (trace.samples == 0 || trace.columns == 0 || trace.values.size() != value_count) {
        error = "activation trace has invalid dimensions or value count";
        return false;
    }
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file) {
        error = "failed to create activation trace: " + path;
        return false;
    }
    const uint32_t header[4] = { kTraceMagic, kTraceVersion, trace.samples, trace.columns };
    file.write(reinterpret_cast<const char *>(header), sizeof(header));
    file.write(reinterpret_cast<const char *>(trace.values.data()),
               static_cast<std::streamsize>(value_count * sizeof(float)));
    if (!file) {
        error = "failed to write activation trace: " + path;
        return false;
    }
    return true;
}
