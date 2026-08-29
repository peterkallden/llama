#include "agent-learning-adapter-import.h"

#include "hash/hash.h"

#include <fstream>
#include <iterator>
#include <vector>

bool common_learning_verify_adapter_artifact(
        const common_learning_adapter_manifest & manifest,
        const std::filesystem::path & artifact_root,
        const size_t max_bytes,
        std::filesystem::path & resolved_path,
        std::string & error) {
    error.clear();
    resolved_path.clear();
    if (!common_learning_validate_adapter_manifest(manifest, error)) return false;
    if (artifact_root.empty() || max_bytes == 0) {
        error = "adapter artifact import requires a root and positive byte bound";
        return false;
    }

    std::error_code ec;
    const auto root = std::filesystem::canonical(artifact_root, ec);
    if (ec) { error = "adapter artifact root is not available"; return false; }
    const auto candidate = std::filesystem::canonical(root / manifest.artifact_path, ec);
    if (ec || !std::filesystem::is_regular_file(candidate, ec) || ec) {
        error = "adapter artifact is not a regular file";
        return false;
    }
    const auto relative = candidate.lexically_relative(root);
    if (relative.empty() || relative.is_absolute() || relative.string().find("..") == 0) {
        error = "adapter artifact escapes its root";
        return false;
    }
    const auto size = std::filesystem::file_size(candidate, ec);
    if (ec) { error = "could not inspect adapter artifact size"; return false; }
    if (size > max_bytes) {
        error = "adapter artifact exceeds import byte bound";
        return false;
    }
    std::ifstream input(candidate, std::ios::binary);
    if (!input) { error = "could not read adapter artifact"; return false; }
    const std::vector<char> bytes((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    if (!input.good() && !input.eof()) { error = "could not read adapter artifact"; return false; }
    if (bytes.size() != size || "sha256:" + hash_sha256_hex(bytes.data(), bytes.size()) != manifest.artifact_sha256) {
        error = "adapter artifact SHA-256 does not match manifest";
        return false;
    }
    resolved_path = candidate;
    return true;
}
