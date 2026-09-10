#include "astc-vulkan-compiled-source.h"

#include "astc-vulkan-compiled-model.h"
#include "astc-vulkan-provenance.h"

#include <chrono>
#include <fstream>

#if defined(__linux__)
#include <cerrno>
#include <fcntl.h>
#include <linux/memfd.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

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

#if defined(__linux__)
int create_memory_file(const char * name) {
#if defined(SYS_memfd_create)
    return static_cast<int>(syscall(SYS_memfd_create, name, MFD_CLOEXEC));
#else
    (void) name;
    return -1;
#endif
}

bool write_memory_file(int fd, const std::vector<uint8_t> & bytes, std::string & error) {
    size_t written = 0;
    while (written < bytes.size()) {
        const ssize_t count = ::write(fd, bytes.data() + written, bytes.size() - written);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) {
            error = "cannot write embedded GGUF memory file";
            return false;
        }
        written += static_cast<size_t>(count);
    }
    if (::lseek(fd, 0, SEEK_SET) < 0) {
        error = "cannot rewind embedded GGUF memory file";
        return false;
    }
    return true;
}
#endif

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
    close_model_file();
    std::error_code ignored;
    if (!root_.empty()) fs::remove_all(root_, ignored);
    root_.clear();
    model_path_.clear();
    cache_path_.clear();
    source_fingerprint_.clear();
    mode_ = astc_vulkan_compiled_source_mode::private_file;
}

void astc_vulkan_compiled_source::close_model_file() {
#if defined(__linux__)
    if (model_fd_ >= 0) {
        ::close(model_fd_);
        model_fd_ = -1;
    }
#else
    model_fd_ = -1;
#endif
}

const char * astc_vulkan_compiled_source::materialization_mode_name() const {
    return mode_ == astc_vulkan_compiled_source_mode::memory_file ?
        "memory-file" : "private-file";
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
    if (!fs::create_directories(cache_root, ec) || ec) {
        if (error.empty()) error = "cannot create compiled-model cache workspace";
        reset();
        return false;
    }

    const std::string embedded_hash = astc_vulkan_sha256_hex(
        model.gguf.data(), model.gguf.size());
    if (embedded_hash != model.source_model_fingerprint) {
        error = "compiled-model source fingerprint does not match embedded GGUF";
        reset();
        return false;
    }

    // The normal model loader is path-based today. On Linux, a memfd keeps
    // the GGUF bytes out of the private workspace and remains valid for the
    // whole server lifetime. If memfd is unavailable, use the portable file
    // materialization below.
#if defined(__linux__)
    model_fd_ = create_memory_file("astc-vulkan-embedded-gguf");
    if (model_fd_ >= 0 && write_memory_file(model_fd_, model.gguf, error)) {
        const std::string proc_path = "/proc/self/fd/" + std::to_string(model_fd_);
        if (::access(proc_path.c_str(), R_OK) == 0) {
            model_path_ = proc_path;
            mode_ = astc_vulkan_compiled_source_mode::memory_file;
        } else {
            close_model_file();
        }
    }
    if (model_path_.empty()) {
        if (model_fd_ >= 0) close_model_file();
        if (!write_bytes(gguf_path, model.gguf, error)) {
            reset();
            return false;
        }
        model_path_ = gguf_path.string();
        mode_ = astc_vulkan_compiled_source_mode::private_file;
    }
#else
    if (!write_bytes(gguf_path, model.gguf, error)) {
        reset();
        return false;
    }
    model_path_ = gguf_path.string();
    mode_ = astc_vulkan_compiled_source_mode::private_file;
#endif

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

    cache_path_ = cache_root.string();
    source_fingerprint_ = embedded_hash;
    error.clear();
    return true;
}
