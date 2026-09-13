#include "astc-vulkan-native-tensor-table.h"

#include <array>
#include <limits>
#include <unordered_set>

namespace {
constexpr std::array<char, 8> kMagic = {'A','S','T','C','N','0','0','1'};
constexpr uint32_t kMaxRecords = 1u << 20;
constexpr uint32_t kMaxName = 1u << 20;

void put_u32(std::vector<uint8_t> & out, uint32_t v) {
    for (unsigned s = 0; s < 32; s += 8) out.push_back((v >> s) & 0xffu);
}
void put_u64(std::vector<uint8_t> & out, uint64_t v) {
    for (unsigned s = 0; s < 64; s += 8) out.push_back((v >> s) & 0xffu);
}
bool get_u32(const std::vector<uint8_t> & in, size_t & p, uint32_t & v) {
    if (in.size() - p < 4) return false; v = 0;
    for (unsigned s = 0; s < 32; s += 8) v |= uint32_t(in[p++]) << s;
    return true;
}
bool get_u64(const std::vector<uint8_t> & in, size_t & p, uint64_t & v) {
    if (in.size() - p < 8) return false; v = 0;
    for (unsigned s = 0; s < 64; s += 8) v |= uint64_t(in[p++]) << s;
    return true;
}
}

bool astc_vulkan_validate_native_tensor_table(
    const astc_vulkan_native_tensor_table & table, std::string & error) {
    if (table.version != astc_vulkan_native_tensor_table::kVersion ||
        table.tensors.empty() || table.tensors.size() > kMaxRecords) {
        error = "unsupported or empty native tensor table"; return false;
    }
    std::unordered_set<std::string> names;
    for (const auto & r : table.tensors) {
        if (r.name.empty() || r.name.size() > kMaxName || r.n_dims == 0 || r.n_dims > 4 ||
            (r.storage != astc_vulkan_native_tensor_storage::native &&
             r.storage != astc_vulkan_native_tensor_storage::astc) || !names.insert(r.name).second) {
            error = "invalid or duplicate native tensor table record"; return false;
        }
        if (r.storage == astc_vulkan_native_tensor_storage::native && r.native_size == 0) {
            error = "native tensor record has no data"; return false;
        }
        if (r.storage == astc_vulkan_native_tensor_storage::astc && r.native_size != 0) {
            error = "ASTC tensor record unexpectedly owns native data"; return false;
        }
    }
    error.clear(); return true;
}

bool astc_vulkan_encode_native_tensor_table(
    const astc_vulkan_native_tensor_table & table, std::vector<uint8_t> & bytes,
    std::string & error) {
    if (!astc_vulkan_validate_native_tensor_table(table, error)) return false;
    bytes.assign(kMagic.begin(), kMagic.end());
    put_u32(bytes, table.version); put_u32(bytes, uint32_t(table.tensors.size()));
    for (const auto & r : table.tensors) {
        put_u32(bytes, uint32_t(r.name.size()));
        bytes.insert(bytes.end(), r.name.begin(), r.name.end());
        bytes.push_back(uint8_t(r.storage)); bytes.insert(bytes.end(), 3, 0);
        put_u32(bytes, r.ggml_type); put_u32(bytes, r.n_dims);
        for (uint64_t n : r.ne) put_u64(bytes, n);
        put_u64(bytes, r.native_offset); put_u64(bytes, r.native_size); put_u64(bytes, r.native_hash64);
    }
    error.clear(); return true;
}

bool astc_vulkan_decode_native_tensor_table(
    const std::vector<uint8_t> & bytes, astc_vulkan_native_tensor_table & table,
    std::string & error) {
    table = {};
    if (bytes.size() < 16 || !std::equal(kMagic.begin(), kMagic.end(), bytes.begin())) {
        error = "invalid native tensor table magic"; return false;
    }
    size_t p = 8; uint32_t count = 0;
    if (!get_u32(bytes,p,table.version) || !get_u32(bytes,p,count) || count == 0 || count > kMaxRecords) {
        error = "invalid native tensor table header"; return false;
    }
    table.tensors.reserve(count);
    for (uint32_t i=0;i<count;++i) {
        uint32_t len=0, storage=0;
        if (!get_u32(bytes,p,len) || len == 0 || len > kMaxName || bytes.size()-p < len+4) {
            error="truncated native tensor table name"; return false;
        }
        astc_vulkan_native_tensor_record r;
        r.name.assign(reinterpret_cast<const char *>(bytes.data()+p), len); p += len;
        storage=bytes[p++]; p += 3; r.storage=static_cast<astc_vulkan_native_tensor_storage>(storage);
        if (!get_u32(bytes,p,r.ggml_type) || !get_u32(bytes,p,r.n_dims)) { error="truncated native tensor metadata"; return false; }
        for (auto & n:r.ne) if (!get_u64(bytes,p,n)) { error="truncated native tensor shape"; return false; }
        if (!get_u64(bytes,p,r.native_offset) || !get_u64(bytes,p,r.native_size) || !get_u64(bytes,p,r.native_hash64)) { error="truncated native tensor range"; return false; }
        table.tensors.push_back(std::move(r));
    }
    if (p != bytes.size()) { error="native tensor table has trailing bytes"; return false; }
    return astc_vulkan_validate_native_tensor_table(table,error);
}
