#pragma once

#include "plan/plan-types.h"
#include <nlohmann/json.hpp>

// One codec for package import and both persistent plan backends. Omit empty
// additions so legacy package fingerprints remain stable.
inline nlohmann::ordered_json common_plan_workflow_binding_to_json(
        const common_plan_workflow_binding & binding) {
    nlohmann::ordered_json value = {{"workflow_ref", binding.workflow_ref},
        {"workflow_revision", binding.workflow_revision}};
    if (!binding.phase_bindings.empty()) value["phase_bindings"] = binding.phase_bindings;
    return value;
}

inline common_plan_workflow_binding common_plan_workflow_binding_from_json(
        const nlohmann::ordered_json & value) {
    return {value.at("workflow_ref").get<std::string>(), value.at("workflow_revision").get<std::string>(),
        value.value("phase_bindings", std::map<std::string, std::vector<std::string>>{})};
}
