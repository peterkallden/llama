#include "astc-vulkan-input.h"

#include <cstdio>
#include <string>
#include <vector>

namespace {

bool parse_args(int argc, char ** argv, std::string & output,
                std::vector<std::string> & inputs) {
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string option = argv[i];
        const std::string value = argv[i + 1];
        if (option == "--output") output = value;
        else if (option == "--input") inputs.push_back(value);
        else return false;
    }
    return !output.empty() && !inputs.empty();
}

} // namespace

// Concatenates versioned activation traces captured from separate model prompts.
// This is deliberately an offline corpus utility: all inputs must have the same
// activation width, and it preserves sample order for explicit cal/val/holdout
// splits in the selector.
int main(int argc, char ** argv) {
    std::string output;
    std::vector<std::string> inputs;
    if (!parse_args(argc, argv, output, inputs)) {
        std::fprintf(stderr, "usage: %s --output merged.trace --input a.trace [--input b.trace ...]\n", argv[0]);
        return 2;
    }

    ggml_vk_astc_activation_trace merged;
    std::string error;
    for (const auto & path : inputs) {
        ggml_vk_astc_activation_trace trace;
        if (!ggml_vk_astc_load_activation_trace(path, trace, error)) {
            std::fprintf(stderr, "failed to read %s: %s\n", path.c_str(), error.c_str());
            return 1;
        }
        if (merged.columns == 0) merged.columns = trace.columns;
        if (trace.columns != merged.columns ||
            UINT32_MAX - merged.samples < trace.samples) {
            std::fprintf(stderr, "trace dimensions do not form a valid corpus\n");
            return 1;
        }
        merged.samples += trace.samples;
        merged.values.insert(merged.values.end(), trace.values.begin(), trace.values.end());
    }
    if (!ggml_vk_astc_write_activation_trace(output, merged, error)) {
        std::fprintf(stderr, "failed to write %s: %s\n", output.c_str(), error.c_str());
        return 1;
    }
    std::printf("merged %zu traces into %u samples x %u columns at %s\n",
                inputs.size(), merged.samples, merged.columns, output.c_str());
    return 0;
}
