#pragma once

#include "agent/adaptation/flydelta/flydelta-artifact-store.h"
#include "agent/adaptation/flydelta/flydelta-sideband-registry.h"

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

// Builds the immutable schema-v2 artifact for an experimental search result.
// The artifact is not a candidate/active model and contains no raw prompts or
// verifier payloads.
bool common_flydelta_build_experimental_artifact(
        const std::string & id,
        uint64_t generation,
        const common_flydelta_encoder_config & encoder,
        const common_flydelta_memory_config & memory,
        const common_flydelta_compatibility & compatibility,
        const std::vector<float> & weights,
        size_t model_n_embd,
        size_t model_n_layers,
        int32_t il_start,
        int32_t il_end,
        const std::vector<common_flydelta_artifact_direction> & steering_basis,
        common_flydelta_artifact & artifact,
        std::string & error);

bool common_flydelta_experimental_manifest_from_artifact(
        const common_flydelta_artifact & artifact,
        const std::filesystem::path & relative_path,
        const std::string & namespace_id,
        const std::string & project_id,
        uint64_t expires_at_epoch_ms,
        common_flydelta_sideband_manifest & manifest,
        std::string & error);

// Persists the immutable artifact and admits only an experimental registry
// entry. A later explicit review must promote it before canary/activation.
bool common_flydelta_persist_experimental_artifact(
        const common_flydelta_artifact_store & store,
        const std::filesystem::path & relative_path,
        common_flydelta_sideband_registry & registry,
        const std::string & namespace_id,
        const std::string & project_id,
        uint64_t expires_at_epoch_ms,
        common_flydelta_artifact & artifact,
        common_flydelta_sideband_manifest & manifest,
        std::string & error);
