#include "agent-resource-store.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <unordered_map>

namespace {

struct sha256_state {
    std::array<uint32_t, 8> h = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    };
    std::array<uint8_t, 64> block = {};
    uint64_t bit_count = 0;
    size_t block_size = 0;
};

inline uint32_t rotr(uint32_t value, uint32_t bits) {
    return (value >> bits) | (value << (32 - bits));
}

void sha256_transform(sha256_state & state, const uint8_t * data) {
    static const uint32_t k[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    };

    uint32_t w[64];
    for (size_t i = 0; i < 16; ++i) {
        const size_t base = i * 4;
        w[i] = (static_cast<uint32_t>(data[base]) << 24) |
               (static_cast<uint32_t>(data[base + 1]) << 16) |
               (static_cast<uint32_t>(data[base + 2]) << 8) |
               static_cast<uint32_t>(data[base + 3]);
    }
    for (size_t i = 16; i < 64; ++i) {
        const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
        const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }

    uint32_t a = state.h[0];
    uint32_t b = state.h[1];
    uint32_t c = state.h[2];
    uint32_t d = state.h[3];
    uint32_t e = state.h[4];
    uint32_t f = state.h[5];
    uint32_t g = state.h[6];
    uint32_t h = state.h[7];

    for (size_t i = 0; i < 64; ++i) {
        const uint32_t s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
        const uint32_t ch = (e & f) ^ ((~e) & g);
        const uint32_t temp1 = h + s1 + ch + k[i] + w[i];
        const uint32_t s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
        const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        const uint32_t temp2 = s0 + maj;

        h = g;
        g = f;
        f = e;
        e = d + temp1;
        d = c;
        c = b;
        b = a;
        a = temp1 + temp2;
    }

    state.h[0] += a;
    state.h[1] += b;
    state.h[2] += c;
    state.h[3] += d;
    state.h[4] += e;
    state.h[5] += f;
    state.h[6] += g;
    state.h[7] += h;
}

void sha256_update(sha256_state & state, const std::string & bytes) {
    for (unsigned char ch : bytes) {
        state.block[state.block_size++] = ch;
        state.bit_count += 8;
        if (state.block_size == state.block.size()) {
            sha256_transform(state, state.block.data());
            state.block_size = 0;
        }
    }
}

std::string sha256_hex(const std::string & bytes) {
    sha256_state state;
    sha256_update(state, bytes);

    state.block[state.block_size++] = 0x80;
    if (state.block_size > 56) {
        while (state.block_size < 64) {
            state.block[state.block_size++] = 0;
        }
        sha256_transform(state, state.block.data());
        state.block_size = 0;
    }
    while (state.block_size < 56) {
        state.block[state.block_size++] = 0;
    }
    for (int i = 7; i >= 0; --i) {
        state.block[state.block_size++] = static_cast<uint8_t>((state.bit_count >> (i * 8)) & 0xff);
    }
    sha256_transform(state, state.block.data());

    std::ostringstream out;
    out.fill('0');
    out << std::hex;
    for (uint32_t value : state.h) {
        out.width(8);
        out << value;
    }
    return out.str();
}

int64_t current_time_seconds() {
    return static_cast<int64_t>(std::time(nullptr));
}

bool resource_expired(const agent_resource_descriptor & descriptor, int64_t now) {
    return descriptor.expires_at > 0 && now > 0 && now >= descriptor.expires_at;
}

bool authority_allows(
    const agent_resource_descriptor & descriptor,
    const agent_resource_read_authority & authority,
    std::string & error) {
    if (descriptor.namespace_id != authority.namespace_id) {
        error = "resource authority namespace mismatch";
        return false;
    }
    if (resource_expired(descriptor, authority.now)) {
        error = "resource has expired";
        return false;
    }
    switch (descriptor.scope) {
        case common_runtime_resource_scope::turn:
            if (descriptor.session_id != authority.session_id || descriptor.turn_id != authority.turn_id) {
                error = "resource authority turn mismatch";
                return false;
            }
            return true;
        case common_runtime_resource_scope::session:
            if (descriptor.session_id != authority.session_id) {
                error = "resource authority session mismatch";
                return false;
            }
            return true;
        case common_runtime_resource_scope::project:
            if (descriptor.project_id != authority.project_id) {
                error = "resource authority project mismatch";
                return false;
            }
            return true;
    }
    error = "resource scope is invalid";
    return false;
}

std::filesystem::path blob_layout_path(
        const std::filesystem::path & root,
        const std::string & sha256) {
    if (sha256.size() < 4) {
        return root / "blobs" / "sha256" / sha256;
    }
    return root / "blobs" / "sha256" / sha256.substr(0, 2) / sha256.substr(2, 2) / sha256;
}

} // namespace

bool validate_agent_resource_store_config(
    const agent_resource_store_config & config,
    std::string & error) {
    agent_resource_blob_backend blob_backend = agent_resource_blob_backend::auto_;
    if (!parse_agent_resource_blob_backend(config.blob_backend, blob_backend)) {
        error = "unknown resource blob backend: " + config.blob_backend;
        return false;
    }

    agent_resource_metadata_backend metadata_backend = agent_resource_metadata_backend::auto_;
    if (!parse_agent_resource_metadata_backend(config.metadata_backend, metadata_backend)) {
        error = "unknown resource metadata backend: " + config.metadata_backend;
        return false;
    }

    if (blob_backend == agent_resource_blob_backend::in_memory && !config.blob_root.empty()) {
        error = "--resource-blob-root requires --resource-blob-backend fs or the default auto backend";
        return false;
    }
    if (blob_backend == agent_resource_blob_backend::s3) {
        error = "resource blob backend s3 is not implemented yet";
        return false;
    }
    if (metadata_backend == agent_resource_metadata_backend::in_memory && !config.metadata_db.empty()) {
        error = "--resource-metadata-db requires --resource-metadata-backend cozo or the default auto backend";
        return false;
    }
#ifndef LLAMA_MEMORY_USE_COZO
    if (metadata_backend == agent_resource_metadata_backend::cozo) {
        error = "this binary was built without LLAMA_MEMORY_COZO";
        return false;
    }
#endif

    error.clear();
    return true;
}

bool agent_in_memory_blob_store::put_bytes(
    const std::string & bytes,
    agent_blob_descriptor & out,
    std::string & error) {
    (void) error;
    const std::string sha256 = sha256_hex(bytes);
    std::lock_guard<std::mutex> lock(mutex_);
    blobs_[sha256] = bytes;
    out.sha256 = sha256;
    out.size_bytes = bytes.size();
    return true;
}

bool agent_in_memory_blob_store::get_bytes(
    const std::string & sha256,
    size_t max_bytes,
    std::string & out,
    std::string & error) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = blobs_.find(sha256);
    if (it == blobs_.end()) {
        error = "blob was not found";
        return false;
    }
    if (it->second.size() > max_bytes) {
        error = "blob exceeds read limit";
        return false;
    }
    out = it->second;
    return true;
}

bool agent_in_memory_blob_store::get_bytes_range(
        const std::string & sha256,
        size_t offset,
        size_t max_bytes,
        std::string & out,
        std::string & error) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = blobs_.find(sha256);
    if (it == blobs_.end()) {
        error = "blob was not found";
        return false;
    }
    if (offset > it->second.size()) {
        error = "blob range offset is out of bounds";
        return false;
    }
    const size_t length = std::min(max_bytes, it->second.size() - offset);
    out.assign(it->second.data() + offset, length);
    error.clear();
    return true;
}

bool agent_in_memory_blob_store::exists_sha256(const std::string & sha256) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return blobs_.find(sha256) != blobs_.end();
}

agent_filesystem_blob_store::agent_filesystem_blob_store(std::string root) :
    root_(std::move(root)) {
}

std::filesystem::path agent_filesystem_blob_store::blob_path_for_sha256(const std::string & sha256) const {
    return blob_layout_path(root_, sha256);
}

bool agent_filesystem_blob_store::put_bytes(
    const std::string & bytes,
    agent_blob_descriptor & out,
    std::string & error) {
    const std::string sha256 = sha256_hex(bytes);
    const auto final_path = blob_path_for_sha256(sha256);

    std::error_code ec;
    std::filesystem::create_directories(final_path.parent_path(), ec);
    if (ec) {
        error = "failed to create blob directory: " + ec.message();
        return false;
    }

    if (!std::filesystem::exists(final_path)) {
        std::ofstream file(final_path, std::ios::binary | std::ios::trunc);
        if (!file) {
            error = "failed to open blob file for write";
            return false;
        }
        file.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        if (!file.good()) {
            error = "failed to write blob file";
            return false;
        }
    }

    out.sha256 = sha256;
    out.size_bytes = bytes.size();
    error.clear();
    return true;
}

bool agent_filesystem_blob_store::get_bytes(
    const std::string & sha256,
    size_t max_bytes,
    std::string & out,
    std::string & error) const {
    const auto path = blob_path_for_sha256(sha256);
    if (!std::filesystem::exists(path) || !std::filesystem::is_regular_file(path)) {
        error = "blob was not found";
        return false;
    }

    const auto file_size = std::filesystem::file_size(path);
    if (file_size > max_bytes) {
        error = "blob exceeds read limit";
        return false;
    }

    std::ifstream file(path, std::ios::binary);
    if (!file) {
        error = "failed to open blob file for read";
        return false;
    }

    out.assign(static_cast<size_t>(file_size), '\0');
    file.read(out.data(), static_cast<std::streamsize>(file_size));
    if (!file.good() && !file.eof()) {
        error = "failed to read blob file";
        return false;
    }
    error.clear();
    return true;
}

bool agent_filesystem_blob_store::get_bytes_range(
        const std::string & sha256,
        size_t offset,
        size_t max_bytes,
        std::string & out,
        std::string & error) const {
    const auto path = blob_path_for_sha256(sha256);
    if (!std::filesystem::exists(path) || !std::filesystem::is_regular_file(path)) {
        error = "blob was not found";
        return false;
    }
    const auto file_size = std::filesystem::file_size(path);
    if (offset > file_size) {
        error = "blob range offset is out of bounds";
        return false;
    }
    const size_t length = std::min(max_bytes, static_cast<size_t>(file_size - offset));
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        error = "failed to open blob file for read";
        return false;
    }
    file.seekg(static_cast<std::streamoff>(offset));
    out.assign(length, '\0');
    file.read(out.data(), static_cast<std::streamsize>(length));
    if (!file.good() && !file.eof()) {
        error = "failed to read blob range";
        return false;
    }
    error.clear();
    return true;
}

bool agent_filesystem_blob_store::exists_sha256(const std::string & sha256) const {
    const auto path = blob_path_for_sha256(sha256);
    return std::filesystem::exists(path) && std::filesystem::is_regular_file(path);
}

agent_catalogued_resource_store::agent_catalogued_resource_store(
    std::shared_ptr<agent_blob_store> blob_store,
    std::unique_ptr<agent_resource_catalog> catalog) :
    blob_store_(std::move(blob_store)),
    catalog_(std::move(catalog)) {
    if (!blob_store_) {
        blob_store_ = std::make_shared<agent_in_memory_blob_store>();
    }
    if (!catalog_) {
        catalog_ = std::make_unique<agent_in_memory_resource_catalog>();
    }
}

bool agent_catalogued_resource_store::put_text(
        const agent_resource_put_request & request,
        agent_resource_descriptor & out,
        std::string & error) {
    agent_resource_put_request bytes_request = request;
    bytes_request.bytes = request.text;
    return put_bytes(bytes_request, out, error);
}

bool agent_catalogued_resource_store::put_bytes(
        const agent_resource_put_request & request,
        agent_resource_descriptor & out,
        std::string & error) {
    agent_blob_descriptor blob;
    if (!blob_store_->put_bytes(request.bytes, blob, error)) {
        return false;
    }

    agent_resource_descriptor descriptor;
    if (!catalog_->next_resource_id(descriptor.resource_id, error)) {
        return false;
    }
    descriptor.uri = "agent-resource://resource/" + descriptor.resource_id;
    descriptor.name = request.name;
    descriptor.description = request.description;
    descriptor.mime_type = request.mime_type;
    descriptor.size_bytes = request.bytes.size();
    descriptor.scope = request.scope;
    descriptor.sha256 = blob.sha256;
    descriptor.namespace_id = request.namespace_id;
    descriptor.session_id = request.session_id;
    descriptor.project_id = request.project_id;
    descriptor.turn_id = request.turn_id;
    descriptor.tool_call_id = request.tool_call_id;
    descriptor.source_provider = request.source_provider;
    descriptor.source_tool = request.source_tool;
    descriptor.created_at = request.created_at > 0 ? request.created_at : current_time_seconds();
    descriptor.expires_at = request.expires_at;
    descriptor.metadata = request.metadata;
    descriptor.lineage = request.lineage;

    if (!catalog_->put_descriptor(descriptor, error)) {
        return false;
    }
    out = descriptor;
    error.clear();
    return true;
}

bool agent_catalogued_resource_store::read_bytes(
        const std::string & uri,
    const agent_resource_read_authority & authority,
    size_t max_bytes,
    std::string & out,
    std::string & error) const {
    agent_resource_descriptor descriptor;
    if (!catalog_->find_descriptor(uri, descriptor, error)) {
        return false;
    }
    if (!authority_allows(descriptor, authority, error)) {
        return false;
    }
    return blob_store_->get_bytes(descriptor.sha256, max_bytes, out, error);
}

bool agent_catalogued_resource_store::read_bytes_range(
        const std::string & uri,
        const agent_resource_read_authority & authority,
        size_t offset,
        size_t max_bytes,
        std::string & out,
        std::string & error) const {
    agent_resource_descriptor descriptor;
    if (!catalog_->find_descriptor(uri, descriptor, error)) {
        return false;
    }
    if (!authority_allows(descriptor, authority, error)) {
        return false;
    }
    if (offset > descriptor.size_bytes) {
        error = "resource range offset is out of bounds";
        return false;
    }
    return blob_store_->get_bytes_range(descriptor.sha256, offset, max_bytes, out, error);
}

bool agent_catalogued_resource_store::read_text(
        const std::string & uri,
        const agent_resource_read_authority & authority,
        size_t max_bytes,
        std::string & out,
        std::string & error) const {
    return read_bytes(uri, authority, max_bytes, out, error);
}

bool agent_catalogued_resource_store::read_text_range(
        const std::string & uri,
        const agent_resource_read_authority & authority,
        size_t offset,
        size_t max_bytes,
        std::string & out,
        std::string & error) const {
    return read_bytes_range(uri, authority, offset, max_bytes, out, error);
}

bool agent_catalogued_resource_store::stat(
    const std::string & uri,
    const agent_resource_read_authority & authority,
    agent_resource_descriptor & out,
    std::string & error) const {
    if (!catalog_->find_descriptor(uri, out, error)) {
        return false;
    }
    return authority_allows(out, authority, error);
}

bool agent_catalogued_resource_store::list(
    const agent_resource_read_authority & authority,
    std::vector<agent_resource_descriptor> & out,
    std::string & error) const {
    std::vector<agent_resource_descriptor> descriptors;
    if (!catalog_->list_descriptors(descriptors, error)) {
        return false;
    }

    out.clear();
    for (const auto & descriptor : descriptors) {
        std::string filter_error;
        if (authority_allows(descriptor, authority, filter_error)) {
            out.push_back(descriptor);
        }
    }
    std::sort(out.begin(), out.end(), [](const auto & lhs, const auto & rhs) {
        if (lhs.created_at != rhs.created_at) {
            return lhs.created_at > rhs.created_at;
        }
        return lhs.uri < rhs.uri;
    });
    error.clear();
    return true;
}

std::shared_ptr<agent_blob_store> make_agent_blob_store(
    const agent_resource_store_config & config,
    std::string & error) {
    if (!validate_agent_resource_store_config(config, error)) {
        return nullptr;
    }

    std::string backend = config.blob_backend;
    if (backend == "auto") {
        backend = "fs";
    }

    std::string blob_root = config.blob_root;
    if (blob_root.empty() && backend == "fs") {
        if (!config.metadata_db.empty()) {
            std::filesystem::path metadata_path(config.metadata_db);
            auto parent = metadata_path.parent_path();
            blob_root = ((parent.empty() ? std::filesystem::path(".") : parent) / "resource-blobs").string();
        } else {
            blob_root = "agent-resources";
        }
    }

    if (backend == "in-memory") {
        error.clear();
        return std::make_shared<agent_in_memory_blob_store>();
    }
    if (backend == "fs") {
        error.clear();
        return std::make_shared<agent_filesystem_blob_store>(blob_root);
    }
    if (backend == "s3") {
        error = "resource blob backend s3 is not implemented yet";
        return nullptr;
    }

    error = "unknown resource blob backend: " + backend;
    return nullptr;
}

std::unique_ptr<agent_resource_store> make_agent_resource_store(
    const agent_resource_store_config & config,
    std::string & error) {
    if (!validate_agent_resource_store_config(config, error)) {
        return nullptr;
    }

    std::string metadata_backend = config.metadata_backend;
    if (metadata_backend == "auto") {
        metadata_backend = config.metadata_db.empty() ? "in-memory" : "cozo";
    }

    std::shared_ptr<agent_blob_store> blob_store = make_agent_blob_store(config, error);
    if (!blob_store) {
        return nullptr;
    }

    if (metadata_backend == "in-memory") {
        error.clear();
        return std::make_unique<agent_catalogued_resource_store>(
            std::move(blob_store),
            std::make_unique<agent_in_memory_resource_catalog>());
    }
    if (metadata_backend == "cozo") {
#ifdef LLAMA_MEMORY_USE_COZO
        auto catalog = std::make_unique<agent_cozo_resource_catalog>();
        const std::string metadata_db = config.metadata_db.empty() ? "resource-metadata.cozo" : config.metadata_db;
        if (!catalog->open(metadata_db, error)) {
            return nullptr;
        }
        error.clear();
        return std::make_unique<agent_catalogued_resource_store>(
            std::move(blob_store),
            std::move(catalog));
#else
        error = "this binary was built without LLAMA_MEMORY_COZO";
        return nullptr;
#endif
    }

    error = "unknown resource metadata backend: " + metadata_backend;
    return nullptr;
}
