#include "astc-vulkan-input.h"
#include "astc-vulkan-yaqa.h"

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
    std::vector<T> values(static_cast<size_t>(size) / sizeof(T));
    file.seekg(0);
    file.read(reinterpret_cast<char *>(values.data()), size);
    return file ? values : std::vector<T>();
}

bool read_metadata(const std::string & path, float & scale_l, float & scale_a, float & offset) {
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
            if (key == "scale_l") { scale_l = std::stof(value); have_l = true; }
            else if (key == "scale_a") { scale_a = std::stof(value); have_a = true; }
            else if (key == "offset") { offset = std::stof(value); have_offset = true; }
        } catch (...) { return false; }
    }
    return have_l && have_a && have_offset;
}

} // namespace

int main(int argc, char ** argv) {
    std::string rgba_path, weights_path, metadata_path, input_path, output_path;
    uint32_t rows = 0, columns = 0;
    for (int index = 1; index < argc; ++index) {
        const std::string option = argv[index];
        if (index + 1 >= argc) break;
        const std::string value = argv[++index];
        if (option == "--rgba") rgba_path = value;
        else if (option == "--weights") weights_path = value;
        else if (option == "--metadata") metadata_path = value;
        else if (option == "--input-trace") input_path = value;
        else if (option == "--output-trace") output_path = value;
        else if (option == "--rows") rows = static_cast<uint32_t>(std::stoul(value));
        else if (option == "--columns") columns = static_cast<uint32_t>(std::stoul(value));
        else { std::fprintf(stderr, "unknown option: %s\n", option.c_str()); return 2; }
    }
    if (rgba_path.empty() || weights_path.empty() || metadata_path.empty() || input_path.empty() || output_path.empty() ||
        rows == 0 || columns == 0) {
        std::fprintf(stderr, "usage: %s --rgba decoded.rgba.f32 --weights weights.f32 "
                            "--metadata validation.metadata --input-trace input.trace "
                            "--output-trace output.trace --rows N --columns N\n", argv[0]);
        return 2;
    }
    const std::vector<float> rgba = read_binary<float>(rgba_path);
    const std::vector<float> weights = read_binary<float>(weights_path);
    if (rgba.size() != static_cast<size_t>(rows) * columns * 4 ||
        weights.size() != static_cast<size_t>(rows) * columns) {
        std::fprintf(stderr, "source dimensions do not match rows/columns\n");
        return 2;
    }
    float scale_l = 0.0f, scale_a = 0.0f, offset = 0.0f;
    if (!read_metadata(metadata_path, scale_l, scale_a, offset)) {
        std::fprintf(stderr, "invalid or incomplete artifact metadata: %s\n", metadata_path.c_str());
        return 2;
    }
    std::vector<float> error(weights.size());
    for (size_t index = 0; index < weights.size(); ++index) {
        const float * texel = rgba.data() + index * 4;
        const float decoded = scale_l * (texel[0] + texel[1] + texel[2]) / 3.0f +
                              scale_a * texel[3] + offset;
        error[index] = decoded - weights[index];
    }
    ggml_vk_astc_activation_trace input, output;
    std::string input_error, output_error;
    if (!ggml_vk_astc_load_activation_trace(input_path, input, input_error) ||
        !ggml_vk_astc_load_activation_trace(output_path, output, output_error) ||
        input.samples != output.samples || input.samples == 0 ||
        input.columns != columns || output.columns != rows) {
        std::fprintf(stderr, "input/output traces are incompatible: %s / %s\n",
                     input_error.c_str(), output_error.c_str());
        return 2;
    }
    const double score = astc_vulkan_yaqa_trace_score(error, rows, columns,
                                                       input.values, output.values, input.samples);
    std::printf("yaqa-trace-score rows=%u columns=%u samples=%u score=%.9g\n",
                rows, columns, input.samples, score);
    return std::isfinite(score) ? 0 : 1;
}
