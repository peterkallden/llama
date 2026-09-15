#pragma once

#include "agent/adaptation/flydelta/flydelta-contracts.h"
#include "agent/runtime/model-profile.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>

enum class common_flydelta_sideband_status {
    // Immutable offline-search revision. It may contain a basis and an
    // experimental DeltaMemory, but can never be selected by a model profile.
    experimental,
    candidate,
    canary,
    active,
    retired,
    rejected,
    revoked,
};
const char * common_flydelta_sideband_status_name(common_flydelta_sideband_status status);

// Registry metadata is separate from the activation payload. The host
// verifies the artifact hash and identity here, then loads the payload and
// passes it through common_flydelta_prepare_activation().
struct common_flydelta_sideband_manifest {
    int schema_version = 1;
    std::string id;
    common_flydelta_sideband_status status = common_flydelta_sideband_status::candidate;
    std::string artifact_path;
    std::string artifact_hash;
    std::string namespace_id = "local";
    std::string project_id = "default";
    uint64_t expires_at_epoch_ms = 0;
    std::string revocation_reason;
    common_flydelta_compatibility compatibility;
    size_t model_n_embd = 0;
    size_t model_n_layers = 0;
    int32_t il_start = 1;
    int32_t il_end = 0;
    std::string evaluation_revision;
    bool evaluation_passed = false;
};

bool common_flydelta_sideband_manifest_validate(
        const common_flydelta_sideband_manifest & manifest,
        std::string & error);
std::string common_flydelta_sideband_manifest_to_json(
        const common_flydelta_sideband_manifest & manifest);
bool common_flydelta_sideband_manifest_from_json(
        const std::string & text,
        common_flydelta_sideband_manifest & manifest,
        std::string & error);

class common_flydelta_sideband_registry {
public:
    bool admit(const common_flydelta_sideband_manifest & manifest, std::string & error);
    // Explicitly graduates an offline-search revision into the normal
    // candidate lifecycle. Canary evaluation is still required afterwards.
    bool promote_experimental(const std::string & id, const std::string & evaluation_revision,
            std::string & error);
    bool stage_canary(const std::string & id, const std::string & evaluation_revision,
            std::string & error);
    bool activate(const std::string & id, std::string & error);
    bool retire(const std::string & id, std::string & error);
    bool revoke(const std::string & id, const std::string & reason, std::string & error);

    // Resolve only an explicitly named sideband from a profile. The registry
    // never guesses between multiple sidebands and never reads artifact data.
    bool resolve(
            const common_agent_model_profile & profile,
            const std::string & sideband_id,
            const common_flydelta_compatibility & expected,
            size_t model_n_embd,
            size_t model_n_layers,
            common_flydelta_sideband_manifest & manifest,
            double & profile_scale,
            std::string & error) const;

    const std::map<std::string, common_flydelta_sideband_manifest> & list() const { return manifests; }

private:
    std::map<std::string, common_flydelta_sideband_manifest> manifests;
};
