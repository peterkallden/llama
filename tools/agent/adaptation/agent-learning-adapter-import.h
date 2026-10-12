#pragma once

#include "agent/adaptation/adapter-registry.h"

#include <cstddef>
#include <filesystem>
#include <string>

// Verifies a candidate artifact against the host-owned artifact root. This
// imports no state and never activates an adapter; it only returns the safe,
// canonical path after checking bounds and SHA-256.
bool common_learning_verify_adapter_artifact(
        const common_learning_adapter_manifest & manifest,
        const std::filesystem::path & artifact_root,
        size_t max_bytes,
        std::filesystem::path & resolved_path,
        std::string & error);
