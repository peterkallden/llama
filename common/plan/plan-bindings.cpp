#include "plan/plan-bindings.h"

#include <nlohmann/json.hpp>

using json = nlohmann::ordered_json;

namespace {

const common_plan_step * find_step(const common_plan_state & plan, const std::string & id) {
    for (const auto & step : plan.steps) if (step.id == id) return &step;
    return nullptr;
}

const common_plan_observation * find_step_observation(const common_plan_state & plan, const std::string & step_id) {
    const std::string prefix = "tool:" + step_id + ":";
    const std::string reasoning = "reasoning:" + step_id;
    for (auto it = plan.observations.rbegin(); it != plan.observations.rend(); ++it) {
        if (it->id.rfind(prefix, 0) == 0 || it->id == reasoning || it->source.rfind(prefix, 0) == 0 || it->source == reasoning) return &*it;
    }
    return nullptr;
}

bool resolve_value(const common_plan_state & plan, json & value, size_t depth, std::string & error) {
    if (depth > 16) { error = "tool argument binding nesting is too deep"; return false; }
    if (value.is_array()) {
        for (auto & item : value) if (!resolve_value(plan, item, depth + 1, error)) return false;
        return true;
    }
    if (!value.is_object()) return true;
    if (value.contains("$from_step") || value.contains("$json_pointer")) {
        if (value.size() != 2 || !value["$from_step"].is_string() || !value["$json_pointer"].is_string()) {
            error = "tool argument binding must contain only string $from_step and $json_pointer";
            return false;
        }
        const auto step_id = value["$from_step"].get<std::string>();
        const auto pointer = value["$json_pointer"].get<std::string>();
        if (step_id.empty() || step_id.size() > 64 || pointer.size() > 256 || pointer.empty() || pointer.front() != '/') {
            error = "tool argument binding is out of bounds";
            return false;
        }
        const auto * source_step = find_step(plan, step_id);
        if (!source_step || source_step->status != common_plan_step_status::completed) {
            error = "tool argument binding requires a completed source step";
            return false;
        }
        const auto * observation = find_step_observation(plan, step_id);
        if (!observation) { error = "tool argument binding source has no observation"; return false; }
        const auto source = json::parse(observation->summary, nullptr, false);
        if (source.is_discarded()) { error = "tool argument binding source is not JSON"; return false; }
        try {
            value = source.at(json::json_pointer(pointer));
        } catch (const json::exception &) {
            error = "tool argument binding JSON pointer was not found";
            return false;
        }
        return true;
    }
    for (auto & item : value.items()) if (!resolve_value(plan, item.value(), depth + 1, error)) return false;
    return true;
}

} // namespace

bool common_plan_materialize_tool_arguments(const common_plan_state & plan, const common_plan_step & step,
        const std::string & arguments_json, std::string & materialized_arguments_json, std::string & error) {
    if (arguments_json.size() > 4096) { error = "tool arguments are too large to materialize"; return false; }
    auto arguments = json::parse(arguments_json, nullptr, false);
    if (!arguments.is_object()) { error = "tool arguments must be a JSON object"; return false; }
    if (!resolve_value(plan, arguments, 0, error)) return false;
    materialized_arguments_json = arguments.dump();
    if (materialized_arguments_json.size() > 4096) { error = "materialized tool arguments are too large"; return false; }
    error.clear();
    return true;
}
