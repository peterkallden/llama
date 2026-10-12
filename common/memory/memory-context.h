#pragma once

#include "memory/memory-policy-pack.h"
#include "memory/memory-types.h"

#include <string>
#include <vector>

struct common_memory_context_config {
    size_t char_budget = 4096;
    size_t per_memory_char_budget = 1200;
};

struct common_memory_symbolic_overlay_config {
    size_t char_budget = 1600;
    size_t per_item_char_budget = 240;
    size_t max_constraints = 4;
    size_t max_decisions = 4;
    size_t max_procedures = 3;
    size_t max_facts = 2;
    bool include_facts = true;
};

struct common_memory_policy_pack_render_config {
    size_t char_budget = 1200;
    size_t max_constraints = 6;
    size_t max_decisions = 6;
    size_t max_preferred_procedures = 4;
    size_t per_item_char_budget = 240;
};

enum class common_memory_overlay_stage {
    planning,
    reasoning,
    reflection,
    memory_learning,
    general,
};

std::string common_memory_render_context(
    const std::vector<common_memory_hit> & hits,
    const common_memory_context_config & config = {});

std::vector<common_memory_hit> common_memory_select_symbolic_overlay_hits(
    const std::vector<common_memory_hit> & hits,
    common_memory_overlay_stage stage,
    size_t max_hits = 8);

common_memory_policy_pack common_memory_compact_policy_pack(
    const common_memory_policy_pack & policy_pack,
    const common_memory_policy_pack_render_config & config = {});

bool common_memory_policy_pack_needs_compaction(
    const common_memory_policy_pack & policy_pack,
    const common_memory_policy_pack_render_config & config = {});

std::vector<common_memory_hit> common_memory_compact_symbolic_overlay_hits(
    const std::vector<common_memory_hit> & hits,
    const common_memory_symbolic_overlay_config & config = {});

std::string common_memory_render_policy_pack(
    const common_memory_policy_pack & policy_pack,
    const common_memory_policy_pack_render_config & config = {});

std::string common_memory_render_symbolic_overlay(
    const std::vector<common_memory_hit> & hits,
    const common_memory_symbolic_overlay_config & config = {});

std::string common_memory_escape_context_text(const std::string & text);
