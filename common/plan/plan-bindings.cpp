#include "plan/plan-bindings.h"

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

bool accepts_implicit_dataset(const std::string & tool_name) {
    return tool_name == "dataset.inspect" || tool_name == "dataset.schema" ||
        tool_name == "dataset.sample" || tool_name == "dataset.validate" ||
        tool_name == "data.query" || tool_name == "data.filter" ||
        tool_name == "data.aggregate" || tool_name == "data.transform" ||
        tool_name == "statistics.describe" || tool_name == "statistics.outliers" ||
        tool_name == "statistics.value_counts";
}

void add_unambiguous_dataset_binding(
        const common_plan_state & plan,
        const common_plan_step & step,
        nlohmann::ordered_json & arguments) {
    if (arguments.contains("dataset") || step.depends_on.size() != 1 ||
            !step.tool_call || !accepts_implicit_dataset(step.tool_call->name)) return;
    const auto * source_step = find_step(plan, step.depends_on.front());
    if (!source_step || source_step->status != common_plan_step_status::completed) return;
    const auto * observation = find_step_observation(plan, source_step->id);
    if (!observation) return;
    const auto source = nlohmann::ordered_json::parse(observation->summary, nullptr, false);
    if (!source.is_object() || !source.contains("dataset") || !source["dataset"].is_string() ||
            source["dataset"].get<std::string>().empty()) return;
    arguments["dataset"] = nlohmann::ordered_json{
        {"$from_step", source_step->id},
        {"$json_pointer", "/dataset"},
    };
}

bool resolve_value(
        const common_plan_state & plan,
        nlohmann::ordered_json & value,
        size_t depth,
        std::string & error) {
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
        const auto source = nlohmann::ordered_json::parse(observation->summary, nullptr, false);
        if (source.is_discarded()) { error = "tool argument binding source is not JSON"; return false; }
        try {
            value = source.at(nlohmann::ordered_json::json_pointer(pointer));
        } catch (const nlohmann::ordered_json::exception &) {
            error = "tool argument binding JSON pointer was not found";
            return false;
        }
        return true;
    }
    for (auto & item : value.items()) if (!resolve_value(plan, item.value(), depth + 1, error)) return false;
    return true;
}

} // namespace

bool common_plan_materialize_tool_arguments_contract(
        const common_plan_state & plan,
        const common_plan_step & step,
        const common_plan_tool_arguments_contract & contract,
        common_plan_tool_arguments_contract & materialized_contract,
        std::string & error) {
    (void) step;
    materialized_contract = contract;
    if (!materialized_contract.value.is_object()) {
        error = "tool arguments must be a JSON object";
        return false;
    }
    // Safe typed autowiring: one completed predecessor with one compatible
    // dataset output may fill one omitted dataset input. Ambiguous or missing
    // dataflow remains an ordinary validation failure.
    add_unambiguous_dataset_binding(plan, step, materialized_contract.value);
    if (!resolve_value(plan, materialized_contract.value, 0, error)) {
        return false;
    }
    error.clear();
    return true;
}

bool common_plan_materialize_tool_arguments(
        const common_plan_state & plan,
        const common_plan_step & step,
        const std::string & arguments_json,
        std::string & materialized_arguments_json,
        std::string & error) {
    if (arguments_json.size() > 4096) {
        error = "tool arguments are too large to materialize";
        return false;
    }

    const std::string tool_name = step.tool_call ? step.tool_call->name : std::string();
    common_plan_tool_arguments_contract contract;
    if (!common_plan_parse_tool_arguments_contract_json(
            tool_name,
            arguments_json,
            contract,
            error)) {
        return false;
    }

    common_plan_tool_arguments_contract materialized_contract;
    if (!common_plan_materialize_tool_arguments_contract(
            plan,
            step,
            contract,
            materialized_contract,
            error)) {
        return false;
    }

    if (!common_plan_serialize_tool_arguments_contract_json(
            tool_name,
            materialized_contract,
            materialized_arguments_json,
            error)) {
        return false;
    }

    if (materialized_arguments_json.size() > 4096) {
        error = "materialized tool arguments are too large";
        return false;
    }
    error.clear();
    return true;
}
