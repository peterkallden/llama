#pragma once

#include "agent/adaptation/flydelta/flydelta-basis.h"
#include "agent/adaptation/flydelta/flydelta-gate.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct common_flydelta_static_overlay;

// Layer-sparse overlay material for device-aware backends. Data is stored
// layer-major only for the listed layers; it is already scaled by the gate.
// A legacy dense cvec can be produced explicitly at the model boundary.
struct common_flydelta_sparse_overlay {
    bool enabled = false;
    std::string artifact_id;
    int32_t n_embd = 0;
    int32_t il_start = 1;
    int32_t il_end = 0;
    std::vector<uint32_t> layer_indices;
    std::vector<float> data;
};

// Device-facing batch material. Each entry is a separate sequence overlay;
// entries must retain distinct artifact identities so a backend cannot
// accidentally reuse prompt/KV state for a different intervention.
struct common_flydelta_sparse_overlay_batch {
    int schema_version = 1;
    std::vector<common_flydelta_sparse_overlay> overlays;
};

bool common_flydelta_sparse_overlay_validate(
        const common_flydelta_sparse_overlay & overlay,
        size_t model_n_embd,
        size_t model_n_layers,
        size_t max_bytes,
        std::string & error);

bool common_flydelta_sparse_overlay_batch_validate(
        const common_flydelta_sparse_overlay_batch & batch,
        size_t model_n_embd,
        size_t model_n_layers,
        size_t max_bytes_per_overlay,
        std::string & error);

// Compatibility fallback for a backend that accepts only the legacy dense
// cvec representation. Expansion is performed per sequence and preserves
// artifact identity; it must not merge overlays or share prompt/KV state.
bool common_flydelta_expand_sparse_overlay_batch(
        const common_flydelta_sparse_overlay_batch & batch,
        size_t model_n_embd,
        size_t model_n_layers,
        size_t max_bytes_per_overlay,
        std::vector<common_flydelta_static_overlay> & dense,
        std::string & error);

// Produces a layer-sparse overlay from the same basis and gate inputs as the
// existing dense composer. The selected layers are sorted and duplicate basis
// contributions for one layer are accumulated.
bool common_flydelta_compose_sparse_overlay(
        const std::string & artifact_id,
        size_t model_n_embd,
        size_t model_n_layers,
        int32_t il_start,
        int32_t il_end,
        const std::vector<common_flydelta_basis_direction> & directions,
        const std::vector<float> & coefficients,
        const common_flydelta_gate_decision & gate,
        size_t max_bytes,
        common_flydelta_sparse_overlay & overlay,
        std::string & error);

// Compatibility bridge for the current llama.cpp cvec path. This is the only
// step that expands sparse material to all model layers.
bool common_flydelta_expand_sparse_overlay(
        const common_flydelta_sparse_overlay & sparse,
        size_t model_n_embd,
        size_t model_n_layers,
        size_t max_bytes,
        common_flydelta_static_overlay & dense,
        std::string & error);

// Composes host-selected basis directions into llama.cpp's cvec layout. The
// buffer starts at layer 1 and contains n_embd values for every model layer
// except layer 0. No model or context is touched here.
bool common_flydelta_compose_static_overlay(
        const std::string & artifact_id,
        size_t model_n_embd,
        size_t model_n_layers,
        int32_t il_start,
        int32_t il_end,
        const std::vector<common_flydelta_basis_direction> & directions,
        const std::vector<float> & coefficients,
        const common_flydelta_gate_decision & gate,
        size_t max_bytes,
        common_flydelta_static_overlay & overlay,
        std::string & error);
