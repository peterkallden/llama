#include "astc-vulkan-cache.h"

#include "astc-vulkan-provenance.h"

#include <algorithm>
#include <fstream>
#include <limits>
#include <utility>

namespace {

const std::string * path_for(const astc_vulkan_cache_paths & paths,
                             astc_vulkan_cache_blob_kind kind) {
    switch (kind) {
        case astc_vulkan_cache_blob_kind::manifest:    return &paths.manifest;
        case astc_vulkan_cache_blob_kind::payload:     return &paths.payload;
        case astc_vulkan_cache_blob_kind::layout:      return &paths.layout;
        case astc_vulkan_cache_blob_kind::row_scales:  return &paths.row_scales;
        case astc_vulkan_cache_blob_kind::pair_map:    return &paths.pair_map;
        case astc_vulkan_cache_blob_kind::provenance:  return &paths.provenance;
        case astc_vulkan_cache_blob_kind::catalog:     return &paths.catalog;
    }
    return nullptr;
}

const astc_vulkan_cache_blob * blob_for(const astc_vulkan_cache_blob_set & blobs,
                                        astc_vulkan_cache_blob_kind kind) {
    switch (kind) {
        case astc_vulkan_cache_blob_kind::manifest:    return &blobs.manifest;
        case astc_vulkan_cache_blob_kind::payload:     return &blobs.payload;
        case astc_vulkan_cache_blob_kind::layout:      return &blobs.layout;
        case astc_vulkan_cache_blob_kind::row_scales:  return &blobs.row_scales;
        case astc_vulkan_cache_blob_kind::pair_map:    return &blobs.pair_map;
        case astc_vulkan_cache_blob_kind::provenance:  return &blobs.provenance;
        case astc_vulkan_cache_blob_kind::catalog:     return &blobs.catalog;
    }
    return nullptr;
}

class file_cache_source final : public astc_vulkan_cache_source {
public:
    explicit file_cache_source(astc_vulkan_cache_paths paths) : paths_(std::move(paths)) {}

    bool blob(astc_vulkan_cache_blob_kind kind, astc_vulkan_cache_blob & result,
              std::string & error) const override {
        result = {};
        const std::string * path = path_for(paths_, kind);
        if (path == nullptr || path->empty()) return true;
        std::ifstream file(*path, std::ios::binary | std::ios::ate);
        if (!file) {
            error = "cannot open ASTC cache blob: " + *path;
            return false;
        }
        const std::streamoff end = file.tellg();
        if (end < 0) {
            error = "cannot determine ASTC cache blob size: " + *path;
            return false;
        }
        result.size = static_cast<uint64_t>(end);
        return true;
    }

    bool read_range(astc_vulkan_cache_blob_kind kind, uint64_t offset, uint64_t size,
                    std::vector<uint8_t> & result, std::string & error) const override {
        result.clear();
        const std::string * path = path_for(paths_, kind);
        if (path == nullptr || path->empty()) {
            error = "ASTC cache blob is unavailable";
            return false;
        }
        if (size > static_cast<uint64_t>(std::numeric_limits<size_t>::max()) ||
            offset > static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max())) {
            error = "ASTC cache blob range is too large";
            return false;
        }
        std::ifstream file(*path, std::ios::binary);
        if (!file) {
            error = "cannot open ASTC cache blob: " + *path;
            return false;
        }
        file.seekg(static_cast<std::streamoff>(offset));
        if (!file) {
            error = "cannot seek ASTC cache blob: " + *path;
            return false;
        }
        result.resize(static_cast<size_t>(size));
        if (size != 0 && !file.read(reinterpret_cast<char *>(result.data()),
                                    static_cast<std::streamsize>(size))) {
            error = "cannot read ASTC cache blob range: " + *path;
            result.clear();
            return false;
        }
        return true;
    }

    bool sha256(astc_vulkan_cache_blob_kind kind, std::string & result,
                std::string & error) const override {
        const std::string * path = path_for(paths_, kind);
        if (path == nullptr || path->empty()) {
            result.clear();
            return true;
        }
        return astc_vulkan_sha256_file_hex(*path, result, error);
    }

private:
    astc_vulkan_cache_paths paths_;
};

class memory_cache_source final : public astc_vulkan_cache_source {
public:
    memory_cache_source(astc_vulkan_cache_blob_set blobs,
                        std::shared_ptr<const void> owner)
        : blobs_(blobs), owner_(std::move(owner)) {}

    bool blob(astc_vulkan_cache_blob_kind kind, astc_vulkan_cache_blob & result,
              std::string & error) const override {
        result = {};
        const astc_vulkan_cache_blob * source = blob_for(blobs_, kind);
        if (source == nullptr) {
            error = "invalid ASTC memory cache blob kind";
            return false;
        }
        result = *source;
        return true;
    }

    bool read_range(astc_vulkan_cache_blob_kind kind, uint64_t offset, uint64_t size,
                    std::vector<uint8_t> & result, std::string & error) const override {
        result.clear();
        astc_vulkan_cache_blob source;
        if (!blob(kind, source, error)) return false;
        if (offset > source.size || size > source.size - offset ||
            size > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
            error = "ASTC memory cache blob range is out of bounds";
            return false;
        }
        result.resize(static_cast<size_t>(size));
        if (size != 0) {
            std::copy_n(source.data + offset, static_cast<size_t>(size), result.data());
        }
        return true;
    }

    bool sha256(astc_vulkan_cache_blob_kind kind, std::string & result,
                std::string & error) const override {
        astc_vulkan_cache_blob source;
        if (!blob(kind, source, error)) return false;
        result = astc_vulkan_sha256_hex(source.data, static_cast<size_t>(source.size));
        return true;
    }

private:
    astc_vulkan_cache_blob_set blobs_;
    std::shared_ptr<const void> owner_;
};

} // namespace

std::shared_ptr<const astc_vulkan_cache_source> astc_vulkan_make_file_cache_source(
        const astc_vulkan_cache_paths & paths) {
    return std::make_shared<file_cache_source>(paths);
}

std::shared_ptr<const astc_vulkan_cache_source> astc_vulkan_make_memory_cache_source(
        const astc_vulkan_cache_blob_set & blobs,
        std::shared_ptr<const void> owner) {
    return std::make_shared<memory_cache_source>(blobs, std::move(owner));
}
