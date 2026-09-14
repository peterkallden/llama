#pragma once

#include "agent/adaptation/flydelta/flydelta.h"

#include <cstddef>
#include <filesystem>
#include <string>

struct common_flydelta_artifact_store_config {
    size_t max_weights = 1U << 20;
    size_t max_serialized_bytes = 4U * 1024U * 1024U;
};

// Host-owned immutable storage for versioned .flyd JSON artifacts. Paths are
// relative to root; the store never follows symlinks or accepts traversal.
// It does not change registry lifecycle state or activate an artifact.
class common_flydelta_artifact_store final {
public:
    common_flydelta_artifact_store(
            std::filesystem::path root,
            common_flydelta_artifact_store_config config = {});

    // A retry of the same artifact at the same path is idempotent. A different
    // artifact cannot replace an existing immutable path.
    bool write(
            const std::filesystem::path & relative_path,
            const common_flydelta_artifact & artifact,
            std::string & error) const;

    bool read(
            const std::filesystem::path & relative_path,
            common_flydelta_artifact & artifact,
            std::string & error) const;

private:
    bool resolve(
            const std::filesystem::path & relative_path,
            std::filesystem::path & resolved,
            std::string & error) const;

    std::filesystem::path root;
    common_flydelta_artifact_store_config config;
};
