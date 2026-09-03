#include "astc-vulkan-cache.h"

#include <cstdio>
#include <string>

namespace {

void print_paths(const astc_vulkan_cache_paths & paths) {
    std::printf("astc-cache root=%s\n", paths.root.c_str());
    std::printf("astc-cache manifest=%s\n", paths.manifest.c_str());
    std::printf("astc-cache payload=%s\n", paths.payload.c_str());
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
            "usage: %s create --model model.gguf --manifest artifact.manifest --payload payload.bin "
            "[--layout layout.bin --provenance provenance.txt --cache path|auto]\n"
            "       %s inspect --model model.gguf [--cache path|auto]\n", argv[0], argv[0]);
        return 2;
    }
    const std::string command = argv[1];
    std::string model, manifest, payload, layout, provenance, cache = "auto";
    for (int index = 2; index + 1 < argc; index += 2) {
        const std::string option = argv[index];
        const std::string value = argv[index + 1];
        if (option == "--model") model = value;
        else if (option == "--manifest") manifest = value;
        else if (option == "--payload") payload = value;
        else if (option == "--layout") layout = value;
        else if (option == "--provenance") provenance = value;
        else if (option == "--cache") cache = value;
        else return 2;
    }
    std::string error;
    if (command == "create") {
        astc_vulkan_cache_paths paths;
        if (model.empty() || manifest.empty() || payload.empty() ||
            !astc_vulkan_cache_create(model, manifest, payload, layout, provenance,
                                      cache, paths, error)) {
            std::fprintf(stderr, "astc-cache create failed: %s\n", error.c_str());
            return 1;
        }
        std::printf("astc-cache created source=%s\n", model.c_str());
        print_paths(paths);
        return 0;
    }
    if (command == "inspect") {
        astc_vulkan_cache_validation validation;
        if (model.empty() || !astc_vulkan_cache_validate(model, cache, validation, error)) {
            std::fprintf(stderr, "astc-cache miss: %s\n", error.c_str());
            return 1;
        }
        std::printf("astc-cache hit tensors=%zu paired-d2=%s\n", validation.manifest.tensors.size(),
                    validation.has_paired_d2 ? "true" : "false");
        print_paths(validation.paths);
        return 0;
    }
    std::fprintf(stderr, "unknown astc-cache command: %s\n", command.c_str());
    return 2;
}
