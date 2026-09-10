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

bool create_private_workspace(fs::path & root, uint64_t nonce, std::string & error) {
    std::error_code ec;
    for (unsigned attempt = 0; attempt != 16; ++attempt) {
        root = fs::temp_directory_path() /
            ("astc-vulkan-compiled-source-" + std::to_string(nonce + attempt));
        if (fs::create_directories(root, ec)) return true;
        if (ec) {
            error = "cannot create compiled-model workspace: " + ec.message();
            return false;
        }
    }
    root.clear();
    error = "cannot allocate unique compiled-model workspace";
    return false;
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
    source_fingerprint_.clear();
    cache_source_.reset();
    model_.reset();
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
    auto model = std::make_shared<astc_vulkan_compiled_model>();
    if (!astc_vulkan_compiled_model_read(compiled_model_path, *model, error)) return false;
    model_ = model;

    const std::string embedded_hash = astc_vulkan_sha256_hex(
        model->gguf.data(), model->gguf.size());
    if (embedded_hash != model->source_model_fingerprint) {
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
    if (model_fd_ >= 0 && write_memory_file(model_fd_, model->gguf, error)) {
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
        const auto nonce = static_cast<uint64_t>(
            std::chrono::steady_clock::now().time_since_epoch().count()) ^
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(this));
        if (!create_private_workspace(root_, nonce, error)) {
            reset();
            return false;
        }
        const fs::path gguf_path = root_ / "embedded-model.gguf";
        if (!write_bytes(gguf_path, model->gguf, error)) {
            reset();
            return false;
        }
        model_path_ = gguf_path.string();
        mode_ = astc_vulkan_compiled_source_mode::private_file;
    }
#else
    const auto nonce = static_cast<uint64_t>(
        std::chrono::steady_clock::now().time_since_epoch().count()) ^
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(this));
    if (!create_private_workspace(root_, nonce, error)) {
        reset();
        return false;
    }
    const fs::path gguf_path = root_ / "embedded-model.gguf";
    if (!write_bytes(gguf_path, model->gguf, error)) {
        reset();
        return false;
    }
    model_path_ = gguf_path.string();
    mode_ = astc_vulkan_compiled_source_mode::private_file;
#endif

    astc_vulkan_cache_blob_set blobs;
    blobs.manifest = {model->manifest.data(), model->manifest.size()};
    blobs.payload = {model->payload.data(), model->payload.size()};
    blobs.layout = {model->layout.data(), model->layout.size()};
    blobs.row_scales = {model->row_scales.data(), model->row_scales.size()};
    blobs.pair_map = {model->pair_map.data(), model->pair_map.size()};
    blobs.provenance = {model->provenance.data(), model->provenance.size()};
    blobs.catalog = {model->catalog.data(), model->catalog.size()};
    cache_source_ = astc_vulkan_make_memory_cache_source(
        blobs, std::static_pointer_cast<const void>(model_));
    source_fingerprint_ = embedded_hash;
    error.clear();
    return true;
}
