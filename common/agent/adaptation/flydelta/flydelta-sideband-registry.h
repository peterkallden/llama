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

// Active revisions are the only normal runtime authority.  Canary revisions
// require an explicit, host-validated request-scoped authority and can never
// be reached through the ordinary resolve()/resolve_bound() path.
enum class common_flydelta_runtime_authority {
    active_only,
    canary_evaluation,
};

// Immutable operator decision carried by a review-journal event.  It is not
// artifact metadata: closing or expiring it leaves the candidate revision and
// the default active binding untouched.
struct common_flydelta_canary_envelope {
    int schema_version = 1;
    std::string binding_key;
    std::string candidate_revision_id;
    std::string behavior_key;
    std::string scope_fingerprint;
    uint32_t traffic_basis_points = 0;
    uint64_t expires_at_epoch_ms = 0;
    size_t max_observations = 0;
    float max_scale = 0.0f;
    common_flydelta_compatibility compatibility;
    std::string oracle_revision;
    std::string policy_revision;
    std::string baseline_deployment_fingerprint;
    std::string rollback_revision_id;
};

bool common_flydelta_canary_envelope_validate(
        const common_flydelta_canary_envelope & envelope, std::string & error);

// Stable host-only cohort selection. A missing allocation key, an expired
// envelope or an exhausted observation budget always returns false.
bool common_flydelta_canary_select(
        const common_flydelta_canary_envelope & envelope,
        const std::string & canary_event_id,
        const std::string & allocation_key,
        size_t observed_count,
        uint64_t now_epoch_ms);

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
    common_flydelta_applicability applicability;
    size_t model_n_embd = 0;
    size_t model_n_layers = 0;
    int32_t il_start = 1;
    int32_t il_end = 0;
    std::string evaluation_revision;
    bool evaluation_passed = false;
    // True only when a bounded canary was admitted from iterative semantic
    // progress rather than promotion evidence. Such a canary is never
    // eligible for activation until a later normal evaluation clears it.
    bool canary_progress_only = false;
    // Optional immutable lineage/provenance metadata. Kept at the end of the
    // aggregate to preserve source compatibility for older manifest literals.
    std::string parent_revision_id;
    std::string binding_key;
    std::string oracle_ref;
    std::string oracle_revision;
    std::string policy_revision;
    std::string fixture_set_revision;
};

// The registry owns the replayed projection of a logical runtime selection.
// The binding is mutable history; the sideband manifest and artifact remain
// immutable revisions.
struct common_flydelta_activation_binding {
    int schema_version = 1;
    std::string binding_key;
    std::string selected_revision_id;
    std::string previous_revision_id;
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
    // Admission is idempotent for an identical manifest. A retry with the
    // same id but different metadata is rejected so immutable lifecycle
    // identity cannot be overwritten.
    bool admit(const common_flydelta_sideband_manifest & manifest, std::string & error);
    // Explicit entry point for a search artifact. It keeps callers from
    // accidentally registering an experimental revision as a normal
    // candidate. The artifact remains unresolvable until promoted explicitly.
    bool admit_experimental(const common_flydelta_sideband_manifest & manifest,
            std::string & error);
    // Explicitly graduates an offline-search revision into the normal
    // candidate lifecycle. Canary evaluation is still required afterwards.
    bool promote_experimental(const std::string & id, const std::string & evaluation_revision,
            std::string & error, bool explicit_host_approval = false);
    bool stage_canary(const std::string & id, const std::string & evaluation_revision,
            std::string & error, bool canary_progress_only = false);
    bool activate(const std::string & id, std::string & error);
    bool bind_revision(
            const std::string & binding_key,
            const std::string & revision_id,
            const std::string & expected_current_revision_id,
            std::string & error);
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

    // Callers should use this overload when host-owned behavior/scope/verifier
    // identity is available in addition to model compatibility.
    bool resolve(
            const common_agent_model_profile & profile,
            const std::string & sideband_id,
            const common_flydelta_compatibility & expected,
            const common_flydelta_applicability & expected_applicability,
            size_t model_n_embd,
            size_t model_n_layers,
            common_flydelta_sideband_manifest & manifest,
            double & profile_scale,
            std::string & error) const;

    // Resolve the physical revision selected by a logical profile binding.
    // The profile must contain an overlay with this binding_key. Legacy
    // direct sideband-id resolution remains available through the overloads
    // above.
    bool resolve_bound(
            const common_agent_model_profile & profile,
            const std::string & binding_key,
            const common_flydelta_compatibility & expected,
            size_t model_n_embd,
            size_t model_n_layers,
            common_flydelta_sideband_manifest & manifest,
            double & profile_scale,
            std::string & error) const;

    bool resolve_bound(
            const common_agent_model_profile & profile,
            const std::string & binding_key,
            const common_flydelta_compatibility & expected,
            const common_flydelta_applicability & expected_applicability,
            size_t model_n_embd,
            size_t model_n_layers,
            common_flydelta_sideband_manifest & manifest,
            double & profile_scale,
            std::string & error) const;

    // Deliberately separate from normal binding resolution.  The caller must
    // have selected a still-open review envelope for this exact request.
    bool resolve_canary_bound(
            const common_agent_model_profile & profile,
            const common_flydelta_canary_envelope & envelope,
            common_flydelta_runtime_authority authority,
            const common_flydelta_applicability & expected_applicability,
            size_t model_n_embd,
            size_t model_n_layers,
            common_flydelta_sideband_manifest & manifest,
            double & profile_scale,
            std::string & error) const;

    bool binding(
            const std::string & binding_key,
            common_flydelta_activation_binding & result,
            std::string & error) const;

    const std::map<std::string, common_flydelta_sideband_manifest> & list() const { return manifests; }
    const std::map<std::string, common_flydelta_activation_binding> & bindings() const {
        return active_bindings;
    }

private:
    std::map<std::string, common_flydelta_sideband_manifest> manifests;
    std::map<std::string, common_flydelta_activation_binding> active_bindings;
};
