#include "plan/plan-bindings.h"
#include "plan/plan-contract.h"

#include <nlohmann/json.hpp>


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

using json = nlohmann::ordered_json;

json observation_payload(const common_plan_observation & observation) {
    const auto value = json::parse(observation.summary, nullptr, false);
    if (value.is_object() && value.value("ok", false) && value.contains("result")) return value["result"];
    return value;
}

const common_plan_tool_field_contract * find_field(
        const std::vector<common_plan_tool_field_contract> & fields,
        const std::string & name) {
    for (const auto & field : fields) if (field.name == name) return &field;
    return nullptr;
}

bool observation_has_field(const common_plan_observation & observation, const std::string & field) {
    const auto value = observation_payload(observation);
    return value.is_object() && value.contains(field) && !value[field].is_null();
}

bool validate_explicit_binding_types(
        const common_plan_state & plan,
        const json & arguments,
        const common_plan_tool_dataflow_contract & target,
        const common_plan_tool_dataflow_contract_resolver & resolver,
        std::string & error) {
    if (!resolver) return true;
    for (const auto & input : target.inputs) {
        if (!arguments.contains(input.name) || !arguments[input.name].is_object()) continue;
        const auto & value = arguments[input.name];
        if (!value.contains("$from_step") || !value["$from_step"].is_string() ||
                !value.contains("$json_pointer") || !value["$json_pointer"].is_string()) continue;
        const auto * source_step = find_step(plan, value["$from_step"].get<std::string>());
        if (!source_step || !source_step->tool_call) continue;
        common_plan_tool_dataflow_contract source;
        std::string contract_error;
        if (!resolver(source_step->tool_call->name, source, contract_error)) continue;
        const auto pointer = value["$json_pointer"].get<std::string>();
        if (pointer.size() < 2 || pointer.front() != '/') continue;
        const auto end = pointer.find('/', 1);
        const auto field_name = pointer.substr(1, end == std::string::npos ? std::string::npos : end - 1);
        const auto * output = find_field(source.outputs, field_name);
        if (!output || input.semantic_type.empty() || output->semantic_type.empty()) continue;
        if (output->collection && end == std::string::npos) {
            error = "plan.binding.collection_requires_index: collection output '" + field_name + "' requires an array index";
            return false;
        }
        if (!common_plan_semantic_types_compatible(output->semantic_type, input.semantic_type)) {
            error = "plan.binding.incompatible_types: '" + field_name + "' has type " +
                output->semantic_type + ", but '" + input.name + "' requires " + input.semantic_type;
            return false;
        }
    }
    return true;
}

bool add_unambiguous_typed_binding(
        const common_plan_state & plan,
        const common_plan_step & step,
        nlohmann::ordered_json & arguments,
        const common_plan_tool_dataflow_contract & target,
        const common_plan_tool_dataflow_contract_resolver & resolver,
        std::string & error) {
    if (!resolver || step.depends_on.size() != 1 || !step.tool_call) return false;
    const auto * source_step = find_step(plan, step.depends_on.front());
    if (!source_step || source_step->status != common_plan_step_status::completed) return false;
    const auto * observation = find_step_observation(plan, source_step->id);
    if (!observation) return false;
    common_plan_tool_dataflow_contract source_contract;
    std::string contract_error;
    if (!resolver(source_step->tool_call ? source_step->tool_call->name : std::string(), source_contract, contract_error)) return false;

    struct candidate { std::string input; std::string output; };
    std::vector<candidate> candidates;
    for (const auto & input : target.inputs) {
        if (arguments.contains(input.name) || input.semantic_type.empty()) continue;
        for (const auto & output : source_contract.outputs) {
            if (!output.collection && common_plan_semantic_types_compatible(output.semantic_type, input.semantic_type) &&
                    observation_has_field(*observation, output.name)) {
                candidates.push_back({input.name, output.name});
            }
        }
    }
    if (candidates.size() != 1) {
        if (candidates.size() > 1) {
            bool required_ambiguity = false;
            for (const auto & candidate : candidates) {
                const auto * input = find_field(target.inputs, candidate.input);
                required_ambiguity = required_ambiguity || (input != nullptr && input->required);
            }
            if (required_ambiguity) {
                std::string details;
                for (const auto & item : candidates) {
                    if (!details.empty()) details += ", ";
                    details += item.input + "<-" + source_step->id + "." + item.output;
                }
                error = "plan.binding.ambiguous_autowire: tool '" + step.tool_call->name +
                    "' has multiple unresolved inputs compatible with completed outputs (" + details +
                    "); explicit bindings are required";
                return false;
            }
        }
        return false;
    }
    arguments[candidates.front().input] = nlohmann::ordered_json{
        {"$from_step", source_step->id},
        {"$json_pointer", "/" + candidates.front().output},
    };
    return true;
}

bool resolve_bare_alias_bindings(
        const common_plan_state & plan,
        nlohmann::ordered_json & arguments,
        const common_plan_tool_dataflow_contract & target,
        const common_plan_tool_dataflow_contract_resolver & resolver,
        std::string & error) {
    if (!resolver || !arguments.is_object()) return true;
    for (const auto & input : target.inputs) {
        if (!arguments.contains(input.name) || !arguments[input.name].is_object()) continue;
        auto & value = arguments[input.name];
        if (!value.contains("$from_step") || !value.contains("$json_pointer") ||
                !value["$from_step"].is_string() || !value["$json_pointer"].is_string() ||
                !value["$json_pointer"].get<std::string>().empty()) continue;
        const auto * source_step = find_step(plan, value["$from_step"].get<std::string>());
        if (!source_step || !source_step->tool_call) {
            error = "plan.binding.bare_alias_source_unavailable: bare alias source is not a tool step";
            return false;
        }
        common_plan_tool_dataflow_contract source;
        std::string contract_error;
        if (!resolver(source_step->tool_call->name, source, contract_error)) {
            error = "plan.binding.bare_alias_source_contract_unavailable: cannot inspect bare alias source";
            return false;
        }
        const auto * observation = find_step_observation(plan, source_step->id);
        if (!observation) {
            error = "plan.binding.bare_alias_source_unavailable: bare alias source has no completed observation";
            return false;
        }
        std::vector<std::string> candidates;
        for (const auto & output : source.outputs) {
            if (!output.collection && common_plan_semantic_types_compatible(output.semantic_type, input.semantic_type) &&
                    observation_has_field(*observation, output.name)) candidates.push_back(output.name);
        }
        if (candidates.empty()) {
            error = "plan.binding.bare_alias_no_compatible_output: bare alias '$" +
                value["$from_step"].get<std::string>() + "' has no compatible output for '" + input.name + "'";
            return false;
        }
        if (candidates.size() != 1) {
            error = "plan.binding.ambiguous_bare_alias: bare alias '$" +
                value["$from_step"].get<std::string>() + "' has multiple compatible outputs for '" + input.name + "'";
            return false;
        }
        value["$json_pointer"] = "/" + candidates.front();
    }
    return true;
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
        const auto source = observation_payload(*observation);
        if (source.is_discarded()) { error = "tool argument binding source is not JSON"; return false; }
        try {
            value = source.at(nlohmann::ordered_json::json_pointer(pointer));
        } catch (const nlohmann::ordered_json::exception &) {
            error = "tool argument binding JSON pointer was not found: step=" + step_id + " pointer=" + pointer;
            return false;
        }
        return true;
    }
    for (auto & item : value.items()) if (!resolve_value(plan, item.value(), depth + 1, error)) return false;
    return true;
}

} // namespace

bool common_plan_semantic_types_compatible(
        const std::string & source_type,
        const std::string & target_type) {
    return !source_type.empty() && !target_type.empty() && source_type == target_type;
}

bool common_plan_materialize_tool_arguments_contract(
        const common_plan_state & plan,
        const common_plan_step & step,
        const common_plan_tool_arguments_contract & contract,
        common_plan_tool_arguments_contract & materialized_contract,
        std::string & error,
        const common_plan_tool_dataflow_contract_resolver & dataflow_resolver) {
    materialized_contract = contract;
    if (!materialized_contract.value.is_object()) {
        error = "tool arguments must be a JSON object";
        return false;
    }
    common_plan_tool_dataflow_contract target;
    if (dataflow_resolver && step.tool_call &&
            dataflow_resolver(step.tool_call->name, target, error)) {
        if (!resolve_bare_alias_bindings(
                plan, materialized_contract.value, target, dataflow_resolver, error)) return false;
        if (!validate_explicit_binding_types(
                plan, materialized_contract.value, target, dataflow_resolver, error)) return false;
        if (!add_unambiguous_typed_binding(
                plan, step, materialized_contract.value, target, dataflow_resolver, error)) {
            if (!error.empty()) return false;
        }
    }
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
        std::string & error,
        const common_plan_tool_dataflow_contract_resolver & dataflow_resolver) {
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
            error,
            dataflow_resolver)) {
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

bool common_plan_dataflow_contract_from_schemas(
        const std::string & tool_name,
        const std::string & input_schema_json,
        const std::string & result_schema_json,
        common_plan_tool_dataflow_contract & contract,
        std::string & error) {
    std::vector<common_plan_schema_field> input_fields;
    std::vector<common_plan_schema_field> result_fields;
    if (!common_plan_extract_schema_fields(input_schema_json, input_fields, error) ||
            !common_plan_extract_schema_fields(result_schema_json, result_fields, error)) return false;
    contract = {};
    contract.tool_name = tool_name;
    const auto collect = [](const std::vector<common_plan_schema_field> & source,
            std::vector<common_plan_tool_field_contract> & fields) {
        for (const auto & item : source) {
            if (item.semantic_type.empty()) continue;
            fields.push_back({item.name, item.semantic_type, item.required});
        }
    };
    collect(input_fields, contract.inputs);
    collect(result_fields, contract.outputs);
    error.clear();
    return true;
}
