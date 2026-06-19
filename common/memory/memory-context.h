#pragma once

#include "memory/memory-types.h"

#include <string>
#include <vector>

struct common_memory_context_config {
    size_t char_budget = 4096;
    size_t per_memory_char_budget = 1200;
};

std::string common_memory_render_context(
    const std::vector<common_memory_hit> & hits,
    const common_memory_context_config & config = {});

std::string common_memory_escape_context_text(const std::string & text);
