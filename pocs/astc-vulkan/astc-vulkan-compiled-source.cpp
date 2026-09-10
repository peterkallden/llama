#include "astc-vulkan-compiled-source.h"

#include "astc-vulkan-compiled-model.h"
#include "astc-vulkan-provenance.h"

#include <chrono>
#include <fstream>

namespace {
namespace fs = std::filesystem;

bool write_bytes(const fs::path & path, const std::vector<uint8_t> & bytes, std::string & error) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file || (!bytes.empty() && !file.write(
            reinterpret_cast<const char *>(bytes.data()),
            static_cast<std::streamsize>(bytes.size())))) {
        error = "cannot write compiled-model materialization: " + path.string();
        return false;
    }
    return true;
}

bool write_text(const fs::path & path, const std::string & text, std::string & error) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file || !(file << text << '\n')) {
        error = "cannot write compiled-model metadata: " + path.string();
        return false;
    }
    return true;
}

bool write_hashed(const fs::path & path, const std::vector<uint8_t> & bytes,
                 const fs::path & hash_path, std::string & error) {
    if (!write_bytes(path, bytes, error)) return false;
    std::string hash;
    return astc_vulkan_sha256_file_hex(path.string(), hash, error) &&
        write_text(hash_path, hash, error);
}

bool write_optional_hashed(const fs::path & root, const char * name,
                           const std::vector<uint8_t> & bytes, std::string & error) {
    if (bytes.empty()) return true;
    const fs::path path = root / name;
    // Cache validation uses stable sidecar names without the payload file
    // extension (layout-map.sha256, row-scales.sha256, ...).
    std::string hash_name = name;
    const size_t extension = hash_name.rfind(".bin");
    if (extension != std::string::npos && extension + 4 == hash_name.size()) {
        hash_name.erase(extension);
    } else if (hash_name == "catalog.astcc") {
        hash_name = "catalog";
    }
    return write_hashed(path, bytes, root / (hash_name + ".sha256"), error);
}
} // namespace

astc_vulkan_compiled_source::~astc_vulkan_compiled_source() {
    reset();
}

void astc_vulkan_compiled_source::reset() {
    std::error_code ignored;
    if (!root_.empty()) fs::remove_all(root_, ignored);
    root_.clear();
    model_path_.clear();
    cache_path_.clear();
    source_fingerprint_.clear();
}

bool astc_vulkan_compiled_source::open(const std::string & compiled_model_path,
                                       std::string & error) {
    reset();
    astc_vulkan_compiled_model model;
    if (!astc_vulkan_compiled_model_read(compiled_model_path, model, error)) return false;

    const auto nonce = static_cast<uint64_t>(
        std::chrono::steady_clock::now().time_since_epoch().count()) ^
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(this));
    std::error_code ec;
    for (unsigned attempt = 0; attempt != 16; ++attempt) {
        root_ = fs::temp_directory_path() /
            ("astc-vulkan-compiled-source-" + std::to_string(nonce + attempt));
        if (fs::create_directories(root_, ec)) break;
        if (ec) {
            error = "cannot create compiled-model workspace: " + ec.message();
            reset();
            return false;
        }
    }
    if (root_.empty() || !fs::is_directory(root_, ec)) {
        error = "cannot allocate unique compiled-model workspace";
        reset();
        return false;
    }

    const fs::path gguf_path = root_ / "embedded-model.gguf";
    const fs::path cache_root = root_ / "astc-cache";
    if (!fs::create_directories(cache_root, ec) || ec || !write_bytes(gguf_path, model.gguf, error)) {
        if (error.empty()) error = "cannot create compiled-model cache workspace";
        reset();
        return false;
    }

    std::string embedded_hash;
    if (!astc_vulkan_sha256_file_hex(gguf_path.string(), embedded_hash, error) ||
        embedded_hash != model.source_model_fingerprint) {
        error = "compiled-model source fingerprint does not match embedded GGUF";
        reset();
        return false;
    }

    if (!write_hashed(cache_root / "manifest.astcv", model.manifest,
                      cache_root / "manifest.sha256", error) ||
        !write_hashed(cache_root / "payload.astcpack", model.payload,
                      cache_root / "payload.sha256", error) ||
        !write_optional_hashed(cache_root, "layout-map.bin", model.layout, error) ||
        !write_optional_hashed(cache_root, "row-scales.bin", model.row_scales, error) ||
        !write_optional_hashed(cache_root, "pair-map.bin", model.pair_map, error) ||
        !write_optional_hashed(cache_root, "catalog.astcc", model.catalog, error) ||
        (!model.provenance.empty() &&
         !write_bytes(cache_root / "provenance.txt", model.provenance, error)) ||
        !write_text(cache_root / "source.gguf.sha256", embedded_hash, error)) {
        reset();
        return false;
    }

    model_path_ = gguf_path.string();
    cache_path_ = cache_root.string();
    source_fingerprint_ = embedded_hash;
    error.clear();
    return true;
}
