#pragma once

#include "agent/adaptation/flydelta/flydelta-teaching-material.h"

#include <memory>
#include <string>

// Creates the host-side teaching-material index. With a path, `auto` selects
// the compiled Cozo backend; without a path it remains an in-memory index.
// The returned runtime loads its reference-only snapshot before use.
std::shared_ptr<common_flydelta_teaching_material_runtime>
make_agent_flydelta_teaching_material_runtime(
        const std::string & backend,
        const std::string & path,
        common_flydelta_teaching_material_identity identity,
        std::string & error);
