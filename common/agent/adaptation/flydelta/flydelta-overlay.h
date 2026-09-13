#pragma once

#include "agent/adaptation/flydelta/flydelta-basis.h"
#include "agent/adaptation/flydelta/flydelta-gate.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

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
