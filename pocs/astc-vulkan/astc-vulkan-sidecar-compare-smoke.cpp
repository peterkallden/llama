#include "astc-vulkan-input.h"
#include "astc-vulkan-sidecar.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

namespace {

template<typename T>
std::vector<T> read_binary(const std::string & path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) return {};
    const std::streamsize size = file.tellg();
    if (size <= 0 || size % static_cast<std::streamsize>(sizeof(T)) != 0) return {};
    std::vector<T> result(static_cast<size_t>(size) / sizeof(T));
    file.seekg(0);
    file.read(reinterpret_cast<char *>(result.data()), size);
    return file ? result : std::vector<T>();
}

bool read_metadata(const std::string & path, astc_vulkan_reconstruction & result) {
    std::ifstream file(path);
    if (!file) return false;
    bool have_l = false, have_a = false, have_offset = false;
    std::string line;
    while (std::getline(file, line)) {
        const size_t separator = line.find('=');
        if (separator == std::string::npos) continue;
        const std::string key = line.substr(0, separator);
        const std::string value = line.substr(separator + 1);
        try {
            if (key == "scale_l") { result.scale_l = std::stof(value); have_l = true; }
            else if (key == "scale_a") { result.scale_a = std::stof(value); have_a = true; }
            else if (key == "offset") { result.offset = std::stof(value); have_offset = true; }
        } catch (...) { return false; }
    }
    return have_l && have_a && have_offset;
}

struct case_data {
    std::string name, payload_path, decoded_path, metadata_path;
    std::string manifest_path, payload_blob_path, tensor_name;
    std::vector<uint8_t> payload;
    std::vector<float> decoded;
    astc_vulkan_reconstruction reconstruction;
};

bool run_case(case_data & data, astc_vulkan_footprint footprint,
              const std::vector<uint32_t> & spirv, const ggml_vk_astc_activation_trace & trace,
              const std::vector<float> & weights, uint32_t width, uint32_t height,
              const std::vector<float> * reference_output,
              std::string & error) {
    astc_vulkan_tensor_record manifest_record;
    if (!data.manifest_path.empty() || !data.payload_blob_path.empty()) {
        astc_vulkan_manifest manifest;
        if (data.manifest_path.empty() || data.payload_blob_path.empty() ||
            !astc_vulkan_read_manifest(data.manifest_path, manifest, error)) {
            error = "invalid " + data.name + " ASTC comparison manifest";
            return false;
        }
        const std::string tensor_name = data.tensor_name.empty() ?
            "blk.0.ffn_down.weight" : data.tensor_name;
        const astc_vulkan_tensor_record * record =
            astc_vulkan_find_tensor(manifest, tensor_name);
        const std::vector<uint8_t> blob = read_binary<uint8_t>(data.payload_blob_path);
        if (record == nullptr || !astc_vulkan_validate_payload_blob(
                manifest, blob.size(), error) ||
            record->byte_offset > blob.size() ||
            record->byte_size > blob.size() - record->byte_offset) {
            error = "invalid " + data.name + " ASTC manifest payload range";
            return false;
        }
        manifest_record = *record;
        data.payload.assign(blob.begin() + static_cast<size_t>(record->byte_offset),
                            blob.begin() + static_cast<size_t>(record->byte_offset + record->byte_size));
        data.reconstruction.scale_l = record->scale_l;
        data.reconstruction.scale_a = record->scale_a;
        data.reconstruction.offset = record->offset;
    } else {
        data.payload = read_binary<uint8_t>(data.payload_path);
    }
    data.decoded = read_binary<float>(data.decoded_path);
    if (data.payload.empty() || data.decoded.size() != static_cast<size_t>(width) * height * 4 ||
        (data.manifest_path.empty() && !read_metadata(data.metadata_path, data.reconstruction))) {
        error = "invalid " + data.name + " ASTC comparison inputs";
        return false;
    }
    if (!data.manifest_path.empty() &&
        (manifest_record.width != width || manifest_record.height != height ||
         manifest_record.footprint != footprint)) {
        error = data.name + " ASTC manifest shape/footprint does not match CLI";
        return false;
    }
    astc_vulkan_manifest manifest;
    astc_vulkan_tensor_record record{
        "blk.0.ffn_down.weight", width, height, footprint, 0,
        astc_vulkan_image_bytes(footprint, width, height),
        astc_vulkan_representation::kScalar, data.reconstruction.scale_l,
        data.reconstruction.scale_a, data.reconstruction.offset,
        astc_vulkan_payload_hash64(data.payload.data(), data.payload.size())};
    manifest.tensors.push_back(record);
    astc_vulkan_sidecar sidecar;
    if (!sidecar.set_manifest(manifest, error) || !sidecar.init(footprint, error)) return false;
    astc_vulkan_ffn_binding binding;
    if (!sidecar.bind_tensor(record.name, width, height, data.payload, binding, error) ||
        binding.status != astc_vulkan_binding_status::kReady) {
        if (error.empty()) error = data.name + " ASTC comparison fell back";
        return false;
    }
    std::vector<float> activations(static_cast<size_t>(trace.samples) * width);
    for (uint32_t sample = 0; sample < trace.samples; ++sample) {
        std::copy_n(trace.values.begin() + static_cast<size_t>(sample) * trace.columns,
                    width, activations.begin() + static_cast<size_t>(sample) * width);
    }
    std::vector<float> output;
    if (!sidecar.run(spirv, activations, output, error)) return false;
    double dispatch_error = 0.0, source_error = 0.0, source_energy = 0.0;
    double model_error = 0.0, model_energy = 0.0;
    for (uint32_t sample = 0; sample < trace.samples; ++sample) {
        for (uint32_t row = 0; row < height; ++row) {
            double decoded_dot = 0.0, source_dot = 0.0;
            for (uint32_t column = 0; column < width; ++column) {
                const size_t texel = (static_cast<size_t>(row) * width + column) * 4;
                const float latent = (data.decoded[texel] + data.decoded[texel + 1] +
                                      data.decoded[texel + 2]) / 3.0f;
                const float weight = binding.reconstruction.scale_l * latent +
                                     binding.reconstruction.scale_a * data.decoded[texel + 3] +
                                     binding.reconstruction.offset;
                const float activation = activations[static_cast<size_t>(sample) * width + column];
                decoded_dot += weight * activation;
                source_dot += weights[static_cast<size_t>(row) * width + column] * activation;
            }
            const double delta = output[static_cast<size_t>(sample) * height + row] - decoded_dot;
            dispatch_error += delta * delta;
            const double source_delta = decoded_dot - source_dot;
            source_error += source_delta * source_delta;
            source_energy += source_dot * source_dot;
            if (reference_output != nullptr) {
                const size_t output_index = static_cast<size_t>(sample) * height + row;
                const double model_delta = output[output_index] - (*reference_output)[output_index];
                model_error += model_delta * model_delta;
                model_energy += static_cast<double>((*reference_output)[output_index]) *
                                (*reference_output)[output_index];
            }
        }
    }
    std::printf("sidecar-compare name=%s samples=%u rows=%u columns=%u "
                "gpu-vs-cpu-mse=%.8g astc-vs-source-relative-mse=%.8g",
                data.name.c_str(), trace.samples, height, width,
                dispatch_error / (trace.samples * height),
                source_error / std::max(source_energy, 1e-12));
    if (reference_output != nullptr) {
        std::printf(" model-output-mse=%.8g model-output-relative-mse=%.8g",
                    model_error / (trace.samples * height),
                    model_error / std::max(model_energy, 1e-12));
    }
    std::printf("\n");
    return true;
}

} // namespace

int main(int argc, char ** argv) {
    std::string shader, activation_path, weights_path;
    uint32_t width = 0, height = 0;
    case_data scalar{"scalar"}, gauge{"gauge"};
    std::string reference_output_path;
    for (int index = 1; index + 1 < argc; index += 2) {
        const std::string option = argv[index];
        const std::string value = argv[index + 1];
        if (option == "--shader") shader = value;
        else if (option == "--activations") activation_path = value;
        else if (option == "--weights") weights_path = value;
        else if (option == "--width") width = static_cast<uint32_t>(std::stoul(value));
        else if (option == "--height") height = static_cast<uint32_t>(std::stoul(value));
        else if (option == "--scalar-payload") scalar.payload_path = value;
        else if (option == "--scalar-decoded") scalar.decoded_path = value;
        else if (option == "--scalar-metadata") scalar.metadata_path = value;
        else if (option == "--gauge-payload") gauge.payload_path = value;
        else if (option == "--gauge-decoded") gauge.decoded_path = value;
        else if (option == "--gauge-metadata") gauge.metadata_path = value;
        else if (option == "--reference-output") reference_output_path = value;
        else if (option == "--scalar-manifest") scalar.manifest_path = value;
        else if (option == "--scalar-payload-blob") scalar.payload_blob_path = value;
        else if (option == "--scalar-tensor") scalar.tensor_name = value;
        else if (option == "--gauge-manifest") gauge.manifest_path = value;
        else if (option == "--gauge-payload-blob") gauge.payload_blob_path = value;
        else if (option == "--gauge-tensor") gauge.tensor_name = value;
        else { std::fprintf(stderr, "unknown option: %s\n", option.c_str()); return 2; }
    }
    const bool scalar_manifest = !scalar.manifest_path.empty() || !scalar.payload_blob_path.empty();
    const bool gauge_manifest = !gauge.manifest_path.empty() || !gauge.payload_blob_path.empty();
    if (shader.empty() || activation_path.empty() || weights_path.empty() || width == 0 || height == 0 ||
        scalar.decoded_path.empty() || gauge.decoded_path.empty() ||
        (scalar_manifest ? (scalar.manifest_path.empty() || scalar.payload_blob_path.empty()) :
                           (scalar.payload_path.empty() || scalar.metadata_path.empty())) ||
        (gauge_manifest ? (gauge.manifest_path.empty() || gauge.payload_blob_path.empty()) :
                          (gauge.payload_path.empty() || gauge.metadata_path.empty()))) return 2;
    const std::vector<uint32_t> spirv = read_binary<uint32_t>(shader);
    const std::vector<float> weights = read_binary<float>(weights_path);
    const std::vector<float> reference_output = reference_output_path.empty() ?
        std::vector<float>() : read_binary<float>(reference_output_path);
    ggml_vk_astc_activation_trace trace;
    std::string error;
    if (spirv.empty() || weights.size() != static_cast<size_t>(width) * height ||
        !ggml_vk_astc_load_activation_trace(activation_path, trace, error) || trace.columns < width ||
        trace.samples == 0 ||
        (!reference_output_path.empty() && reference_output.size() !=
            static_cast<size_t>(trace.samples) * height)) {
        std::fprintf(stderr, "invalid ASTC sidecar comparison inputs: %s\n", error.c_str());
        return 2;
    }
    const std::vector<float> * reference = reference_output_path.empty() ? nullptr : &reference_output;
    if (!run_case(scalar, astc_vulkan_footprint::k6x6, spirv, trace, weights, width, height, reference, error) ||
        !run_case(gauge, astc_vulkan_footprint::k6x6, spirv, trace, weights, width, height, reference, error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    return 0;
}
