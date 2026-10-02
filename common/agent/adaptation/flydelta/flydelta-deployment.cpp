#include "agent/adaptation/flydelta/flydelta-deployment.h"

#include "hash/hash.h"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace {

bool finite_scale(double value) {
    return std::isfinite(value) && value > 0.0 && value <= 4.0;
}

bool same_layout(
        const common_flydelta_activation_result & left,
        const common_flydelta_activation_result & right) {
    return left.overlay.n_embd == right.overlay.n_embd &&
        left.overlay.il_start == right.overlay.il_start &&
        left.overlay.il_end == right.overlay.il_end &&
        left.overlay.data.size() == right.overlay.data.size();
}

bool compose_activations(
        const std::vector<common_flydelta_activation_result> & parts,
        const common_flydelta_deployment_request & request,
        common_flydelta_activation_result & result,
        std::string & error) {
    error.clear();
    result = {};
    if (parts.empty()) return true;

    size_t first_active = parts.size();
    for (size_t i = 0; i < parts.size(); ++i) {
        if (parts[i].gate.apply && parts[i].overlay.enabled) {
            first_active = i;
            break;
        }
    }
    if (first_active == parts.size()) return true;

    const auto & first = parts[first_active];
    if (!common_flydelta_activation_result_validate(
            first, request.model_n_embd, request.model_n_layers,
            request.max_overlay_bytes, error)) return false;
    result = first;
    result.overlay.artifact_id.clear();
    result.overlay.scale = 1.0f;
    for (float & value : result.overlay.data) value = 0.0f;
    result.sparse_overlay.artifact_id.clear();
    for (const auto & part : parts) {
        if (!part.gate.apply || !part.overlay.enabled) continue;
        if (!common_flydelta_activation_result_validate(
                part, request.model_n_embd, request.model_n_layers,
                request.max_overlay_bytes, error)) return false;
        if (!same_layout(first, part)) {
            error = "FlyDelta deployment overlays have incompatible layouts";
            return false;
        }
        if (part.overlay.data.size() != result.overlay.data.size()) {
            error = "FlyDelta deployment overlays have incompatible data sizes";
            return false;
        }
        for (size_t i = 0; i < result.overlay.data.size(); ++i) {
            result.overlay.data[i] += part.overlay.data[i];
            if (!std::isfinite(result.overlay.data[i])) {
                error = "FlyDelta deployment overlay composition is non-finite";
                return false;
            }
        }
    }

    result.gate.apply = true;
    result.gate.scale = 1.0f;
    result.gate.reason = "resolved_deployment_composition";
    result.overlay.enabled = true;
    result.overlay.artifact_id = "flydelta://deployment/pending";
    // The artifact identity is replaced by the caller with the canonical
    // deployment fingerprint after the ordered revision list is known.
    result.sparse_overlay = {};
    // Materialize the merged dense result back to the established sparse
    // contract. Keeping every in-range layer is conservative but exact: it
    // preserves the existing validation/device seam while avoiding a second
    // composition rule for mixed dense/sparse source revisions.
    result.sparse_overlay.enabled = true;
    result.sparse_overlay.n_embd = first.overlay.n_embd;
    result.sparse_overlay.il_start = first.overlay.il_start;
    result.sparse_overlay.il_end = first.overlay.il_end;
    result.sparse_overlay.artifact_id = result.overlay.artifact_id;
    for (int32_t layer = first.overlay.il_start; layer <= first.overlay.il_end; ++layer) {
        result.sparse_overlay.layer_indices.push_back(static_cast<uint32_t>(layer));
        const size_t offset = request.model_n_embd * (static_cast<size_t>(layer) - 1);
        result.sparse_overlay.data.insert(
            result.sparse_overlay.data.end(), result.overlay.data.begin() + offset,
            result.overlay.data.begin() + offset + request.model_n_embd);
    }
    if (!common_flydelta_activation_result_validate(
            result, request.model_n_embd, request.model_n_layers,
            request.max_overlay_bytes, error)) return false;
    return true;
}

bool load_entries(
        const common_flydelta_deployment_factory_config & config,
        const common_flydelta_deployment_request & request,
        const std::vector<common_flydelta_resolved_deployment_entry> & entries,
        const std::vector<common_flydelta_sideband_manifest> & manifests,
        std::shared_ptr<const common_flydelta_activation_result> & activation,
        std::string & error) {
    std::vector<common_flydelta_activation_result> parts;
    parts.reserve(manifests.size());
    for (size_t i = 0; i < manifests.size(); ++i) {
        common_flydelta_activation_result part;
        if (!config.load_activation(
                manifests[i], entries[i].scale, request, part, error)) return false;
        parts.push_back(std::move(part));
    }
    common_flydelta_activation_result composed;
    if (!compose_activations(parts, request, composed, error)) return false;
    activation = std::make_shared<const common_flydelta_activation_result>(std::move(composed));
    return true;
}

} // namespace

std::string common_flydelta_deployment_fingerprint(
        const common_flydelta_deployment_request & request,
        const std::vector<common_flydelta_resolved_deployment_entry> & entries) {
    std::ostringstream canonical;
    canonical << request.profile.base_model_id << '\n'
              << request.profile.base_model_fingerprint << '\n'
              << request.profile.tokenizer_fingerprint << '\n'
              << request.profile.chat_template_fingerprint << '\n'
              << request.compatibility.inference_layout_revision << '\n'
              << request.model_n_embd << ':' << request.model_n_layers << '\n';
    for (const auto & entry : entries) {
        canonical << entry.binding_key << '\n' << entry.revision_id << '\n'
                  << entry.scale << '\n';
    }
    const std::string value = canonical.str();
    return "sha256:" + hash_sha256_hex(value.data(), value.size());
}

bool common_flydelta_resolve_deployment(
        const common_flydelta_deployment_factory_config & config,
        const common_flydelta_deployment_request & request,
        common_flydelta_deployment_result & result,
        std::string & error) {
    error.clear();
    result = {};
    result.active_only = request.authority != common_flydelta_runtime_authority::canary_evaluation;
    if (!config.registry || !config.load_activation) {
        error = "FlyDelta deployment factory requires registry and activation loader";
        return false;
    }
    if (request.profile.sidebands.size() > 4 ||
            request.model_n_embd == 0 || request.model_n_layers < 2 ||
            !finite_scale(request.gate_request.requested_scale)) {
        error = "FlyDelta deployment request is outside bounds";
        return false;
    }
    if (request.authority == common_flydelta_runtime_authority::canary_evaluation &&
            request.allocation_key.empty()) {
        result.active_only = true;
        result.fallback_to_active = true;
        result.fallback_reason = "missing_host_allocation_key";
    }

    std::vector<common_flydelta_sideband_manifest> active_manifests;
    std::vector<common_flydelta_resolved_deployment_entry> active_entries;
    for (const auto & configured : request.profile.sidebands) {
        common_flydelta_sideband_manifest manifest;
        double scale = configured.scale;
        const bool resolved = configured.binding_key.empty()
            ? config.registry->resolve(
                request.profile, configured.sideband_id, request.compatibility,
                request.applicability, request.model_n_embd, request.model_n_layers,
                manifest, scale, error)
            : config.registry->resolve_bound(
                request.profile, configured.binding_key, request.compatibility,
                request.applicability, request.model_n_embd, request.model_n_layers,
                manifest, scale, error);
        if (!resolved) return false;
        active_manifests.push_back(manifest);
        active_entries.push_back({
            configured.binding_key, manifest.id, scale, false, {}});
    }
    result.baseline = active_entries;
    result.effective = active_entries;
    result.baseline_deployment_fingerprint =
        common_flydelta_deployment_fingerprint(request, result.baseline);

    std::vector<common_flydelta_sideband_review> reviews;
    if (request.authority == common_flydelta_runtime_authority::canary_evaluation &&
            !request.allocation_key.empty() && config.open_canaries) {
        if (!config.open_canaries(reviews, error)) return false;
        result.canary_considered = !reviews.empty();
        std::sort(reviews.begin(), reviews.end(), [](const auto & left, const auto & right) {
            return left.event_id < right.event_id;
        });
        for (const auto & review : reviews) {
            if (!review.has_canary_envelope || review.action != common_flydelta_review_action::stage_canary ||
                    (!review.canary_envelope_event_id.empty() &&
                     review.canary_envelope_event_id != review.event_id) ||
                    review.canary_envelope.behavior_key != request.applicability.behavior_key ||
                    review.canary_envelope.scope_fingerprint != request.applicability.scope_fingerprint) {
                continue;
            }
            const auto & envelope = review.canary_envelope;
            const size_t observed = config.observation_counter
                ? config.observation_counter(envelope.binding_key, review.event_id)
                : request.observed_default;
            if (!common_flydelta_canary_select(
                    envelope, review.event_id, request.allocation_key, observed,
                    request.now_epoch_ms == 0 ? envelope.expires_at_epoch_ms - 1 : request.now_epoch_ms)) {
                continue;
            }
            if (envelope.baseline_deployment_fingerprint !=
                    result.baseline_deployment_fingerprint) {
                result.fallback_to_active = true;
                result.fallback_reason = "canary_baseline_fingerprint_mismatch";
                continue;
            }
            common_flydelta_sideband_manifest canary_manifest;
            double canary_scale = 0.0;
            std::string canary_error;
            if (!config.registry->resolve_canary_bound(
                    request.profile, envelope,
                    common_flydelta_runtime_authority::canary_evaluation,
                    request.applicability, request.model_n_embd, request.model_n_layers,
                    canary_manifest, canary_scale, canary_error)) {
                result.fallback_to_active = true;
                result.fallback_reason = canary_error;
                continue;
            }
            const auto existing = std::find_if(result.effective.begin(), result.effective.end(),
                [&](const auto & entry) { return entry.binding_key == envelope.binding_key; });
            const auto active_manifest = std::find_if(active_entries.begin(), active_entries.end(),
                [&](const auto & entry) { return entry.binding_key == envelope.binding_key; });
            if (active_manifest == active_entries.end() ||
                    envelope.rollback_revision_id != active_manifest->revision_id) {
                result.fallback_to_active = true;
                result.fallback_reason = "canary_rollback_revision_mismatch";
                continue;
            }
            result.canary_selected = true;
            result.active_only = false;
            if (existing != result.effective.end()) {
                existing->revision_id = canary_manifest.id;
                existing->scale = canary_scale;
                existing->canary = true;
                existing->canary_event_id = review.event_id;
                const auto active_index = std::distance(active_entries.begin(), active_manifest);
                active_manifests[active_index] = canary_manifest;
            } else {
                result.effective.push_back({envelope.binding_key, canary_manifest.id,
                    canary_scale, true, review.event_id});
                active_manifests.push_back(canary_manifest);
            }
            break;
        }
    }

    if (!result.canary_selected) {
        result.active_only = true;
    }

    if (result.canary_selected) {
        // Re-resolve manifests in effective order so replacement and additive
        // composition share one deterministic loader path.
        std::vector<common_flydelta_sideband_manifest> effective_manifests;
        effective_manifests.reserve(result.effective.size());
        for (const auto & entry : result.effective) {
            auto manifest = std::find_if(active_manifests.begin(), active_manifests.end(),
                [&](const auto & value) { return value.id == entry.revision_id; });
            if (manifest == active_manifests.end()) {
                error = "FlyDelta effective deployment manifest is missing";
                return false;
            }
            effective_manifests.push_back(*manifest);
        }
        if (!load_entries(config, request, result.effective, effective_manifests,
                result.activation, error)) return false;
        result.candidate_deployment_fingerprint =
            common_flydelta_deployment_fingerprint(request, result.effective);
    } else {
        if (!load_entries(config, request, result.baseline, active_manifests,
                result.activation, error)) return false;
        result.candidate_deployment_fingerprint = result.baseline_deployment_fingerprint;
    }
    if (result.activation && result.activation->overlay.enabled) {
        auto mutable_activation = std::make_shared<common_flydelta_activation_result>(*result.activation);
        mutable_activation->overlay.artifact_id = "flydelta://deployment/" +
            result.candidate_deployment_fingerprint.substr(7, 32);
        if (mutable_activation->sparse_overlay.enabled) {
            mutable_activation->sparse_overlay.artifact_id = mutable_activation->overlay.artifact_id;
        }
        result.activation = std::move(mutable_activation);
    }
    return true;
}
