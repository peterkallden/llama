#include "astc-vulkan-compiled-source.h"

#include "astc-vulkan-compiled-model.h"
#include "astc-vulkan-provenance.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <chrono>
#include <fstream>
#include <cstring>

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
    native_table_ = {};
    native_index_.clear();
    callback_error_.clear();
    if (metadata_) {
        gguf_free(metadata_);
        metadata_ = nullptr;
    }
    model_.reset();
    mode_ = astc_vulkan_compiled_source_mode::private_file;
}

bool astc_vulkan_compiled_source::initialize_v2(std::string & error) {
    if (!model_ || !model_->is_bootstrap_v2()) { error = "not a bootstrap compiled model"; return false; }
    gguf_init_params params{true, nullptr};
    metadata_ = gguf_init_from_buffer(model_->gguf.data(), model_->gguf.size(), params);
    if (!metadata_) { error = "cannot initialize embedded bootstrap GGUF metadata"; return false; }
    if (!astc_vulkan_decode_native_tensor_table(model_->native_table, native_table_, error)) return false;
    if (native_table_.tensors.size() != static_cast<size_t>(gguf_get_n_tensors(metadata_))) {
        error = "bootstrap table does not cover the embedded GGUF tensor inventory"; return false;
    }
    for (size_t index = 0; index < native_table_.tensors.size(); ++index) {
        const auto & record = native_table_.tensors[index];
        const int64_t tid = gguf_find_tensor(metadata_, record.name.c_str());
        if (tid < 0 || static_cast<uint32_t>(gguf_get_tensor_type(metadata_, tid)) != record.ggml_type ||
            gguf_get_tensor_size(metadata_, tid) != record.native_size &&
            record.storage == astc_vulkan_native_tensor_storage::native) {
            error = "bootstrap table does not match GGUF tensor descriptor: " + record.name;
            return false;
        }
        native_index_.emplace(record.name, index);
    }
    source_fingerprint_ = model_->source_model_fingerprint;
    return true;
}

void astc_vulkan_compiled_source::set_tensor_data_callback(
    struct ggml_tensor * tensor, void * userdata) {
    static_cast<astc_vulkan_compiled_source *>(userdata)->set_tensor_data(tensor);
}

void astc_vulkan_compiled_source::set_tensor_data(struct ggml_tensor * tensor) {
    if (!callback_error_.empty()) return;
    const char * name = ggml_get_name(tensor);
    const auto it = native_index_.find(name ? name : "");
    if (it == native_index_.end()) { callback_error_ = "compiled bootstrap tensor missing from table"; return; }
    const auto & record = native_table_.tensors[it->second];
    if (!tensor->buffer) { callback_error_ = "compiled bootstrap tensor has no target buffer: " + record.name; return; }
    if (record.storage == astc_vulkan_native_tensor_storage::native) {
        if (ggml_nbytes(tensor) != record.native_size ||
            record.native_offset > model_->native_payload.size() ||
            record.native_size > model_->native_payload.size() - record.native_offset) {
            callback_error_ = "compiled native tensor range does not match destination: " + record.name;
            return;
        }
        // The destination may live in device memory when GPU offload is
        // enabled.  Use the backend upload API rather than dereferencing
        // tensor->data directly; the latter is only valid for host buffers.
        ggml_backend_tensor_set(tensor,
                                model_->native_payload.data() + record.native_offset,
                                0, static_cast<size_t>(record.native_size));
        return;
    }
    // ASTC tensors are replaced by the opt-in provider before compute. Zero
    // their conventional ggml allocation so a missed binding is deterministic,
    // never uninitialized memory. Strict provider admission is checked at the
    // runtime seam; this callback is only the model-loader data contract.
    std::vector<uint8_t> zeros(ggml_nbytes(tensor), 0);
    ggml_backend_tensor_set(tensor, zeros.data(), 0, zeros.size());
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

    const std::string embedded_hash = astc_vulkan_sha256_hex(model->gguf.data(), model->gguf.size());
    if (!model->is_bootstrap_v2() && embedded_hash != model->source_model_fingerprint) {
        error = "compiled-model source fingerprint does not match embedded GGUF";
        reset();
        return false;
    }

    if (model->is_bootstrap_v2() && !initialize_v2(error)) {
        reset();
        return false;
    }

    // V1 uses the normal path-based model loader. On Linux, a memfd keeps
    // the GGUF bytes out of the private workspace and remains valid for the
    // whole server lifetime. If memfd is unavailable, use the portable file
    // materialization below.
#if defined(__linux__)
    if (!model->is_bootstrap_v2()) {
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
    }
#else
    if (!model->is_bootstrap_v2()) {
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
    }
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
    blobs.embedding_metadata = {model->embedding_metadata.data(), model->embedding_metadata.size()};
    blobs.embedding_payload = {model->embedding_payload.data(), model->embedding_payload.size()};
    blobs.embedding_affine = {model->embedding_affine.data(), model->embedding_affine.size()};
    cache_source_ = astc_vulkan_make_memory_cache_source(
        blobs, std::static_pointer_cast<const void>(model_));
    if (!model->is_bootstrap_v2()) source_fingerprint_ = embedded_hash;
    error.clear();
    return true;
}
