#pragma once

#include "agent/adaptation/flydelta/flydelta-teaching-material.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// Host-owned, reference-only work item for materialising one matched
// baseline/conditioned/control concept trajectory. It contains no prompts,
// tensors or model contexts; the model host resolves the refs and executes
// one bounded capture slice.
struct common_flydelta_concept_capture_plan {
    int schema_version = 1;
    std::string id;
    std::string group_ref;
    std::string relation_ref;
    std::string baseline_ref;
    std::string conditioned_ref;
    std::string control_ref;
    std::string semantic_anchor;
    int32_t layer_index = -1;
    common_flydelta_teaching_material_identity identity;
};

bool common_flydelta_concept_capture_plan_validate(
        const common_flydelta_concept_capture_plan & plan,
        std::string & error);
std::string common_flydelta_concept_capture_plan_to_json(
        const common_flydelta_concept_capture_plan & plan);

bool common_flydelta_concept_capture_plan_from_json(
        const std::string & text,
        common_flydelta_concept_capture_plan & plan,
        std::string & error);
