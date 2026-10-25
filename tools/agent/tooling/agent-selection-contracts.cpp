#include "agent-selection-contracts.h"
#include "agent/tooling/schema/tool-dsl-grammar.h"
#include "agent/tooling/schema/tool-schema-compact.h"
#include "agent/tooling/contracts/schema-contract.h"

using json = nlohmann::ordered_json;

namespace {

bool parse_choice_contract_json(
        const std::string & text,
        const char * id_field,
        std::string & decision,
        std::string & id,
        float & confidence,
        std::string & error) {
    const auto parsed = json::parse(text, nullptr, false);
    if (!parsed.is_object()) {
        error = "selection contract returned invalid JSON";
        return false;
    }

    decision = parsed.value("decision", std::string());
    id = parsed.value(id_field, std::string());
    confidence = parsed.value("confidence", 0.0f);
    error.clear();
    return true;
}

} // namespace

json make_agent_blueprint_selection_schema_json(
        const std::vector<std::string> & blueprint_ids) {
    json ids = json::array({""});
    for (const auto & id : blueprint_ids) {
        ids.push_back(id);
    }
    return {
        {"type", "object"},
        {"additionalProperties", false},
        {"required", {"decision", "blueprint_id", "confidence"}},
        {"properties", {
            {"decision", {{"enum", {"instantiate", "none"}}}},
            {"blueprint_id", {{"enum", ids}}},
            {"confidence", {{"type", "number"}, {"minimum", 0}, {"maximum", 1}}},
        }},
    };
}

std::string make_agent_blueprint_selection_schema_json_string(
        const std::vector<std::string> & blueprint_ids) {
    return make_agent_blueprint_selection_schema_json(blueprint_ids).dump();
}

json make_agent_plan_selection_schema_json(
        const std::vector<std::string> & plan_ids) {
    json ids = json::array({""});
    for (const auto & id : plan_ids) {
        ids.push_back(id);
    }
    return {
        {"type", "object"},
        {"additionalProperties", false},
        {"required", {"decision", "plan_id", "confidence"}},
        {"properties", {
            {"decision", {{"enum", {"resume", "new"}}}},
            {"plan_id", {{"enum", ids}}},
            {"confidence", {{"type", "number"}, {"minimum", 0}, {"maximum", 1}}},
        }},
    };
}

std::string make_agent_plan_selection_schema_json_string(
        const std::vector<std::string> & plan_ids) {
    return make_agent_plan_selection_schema_json(plan_ids).dump();
}

bool parse_agent_blueprint_selection_contract_json(
        const std::string & text,
        agent_blueprint_selection_contract & contract,
        std::string & error) {
    contract = {};
    return parse_choice_contract_json(
        text,
        "blueprint_id",
        contract.decision,
        contract.blueprint_id,
        contract.confidence,
        error);
}

bool parse_agent_plan_selection_contract_json(
        const std::string & text,
        agent_plan_selection_contract & contract,
        std::string & error) {
    contract = {};
    return parse_choice_contract_json(
        text,
        "plan_id",
        contract.decision,
        contract.plan_id,
        contract.confidence,
        error);
}

bool parse_agent_blueprint_binding_contract_json(
        const std::string & text,
        std::vector<agent_blueprint_binding_contract_entry> & bindings,
        std::string & error) {
    bindings.clear();
    const auto parsed = json::parse(text, nullptr, false);
    if (!parsed.is_object() || !parsed.contains("bindings") || !parsed["bindings"].is_array()) {
        error = "blueprint binding returned invalid JSON";
        return false;
    }

    for (const auto & binding : parsed["bindings"]) {
        if (!binding.is_object() ||
                !binding.contains("step_id") || !binding["step_id"].is_string() ||
                !binding.contains("tool") || !binding["tool"].is_object()) {
            error = "invalid blueprint binding";
            return false;
        }

        const auto & tool = binding["tool"];
        if (!tool.contains("name") || !tool["name"].is_string() ||
                !tool.contains("arguments") || !tool["arguments"].is_object()) {
            error = "invalid or duplicate blueprint binding";
            return false;
        }

        bindings.push_back({
            binding["step_id"].get<std::string>(),
            tool["name"].get<std::string>(),
            common_plan_tool_arguments_contract{tool["arguments"]},
        });
    }

    error.clear();
    return true;
}

bool make_agent_selection_presentation(common_agent_tool_output_format format,
        const std::string & name, const json & schema,
        agent_selection_presentation & out, std::string & error) {
    out = {}; error.clear();
    if (format != common_agent_tool_output_format::compact_dsl) {
        out.instructions = "Return only JSON matching the selection contract.";
        out.json_schema = schema.dump();
        return true;
    }
    const common_chat_tool contract{name, "Return the host selection decision.", schema.dump()};
    out.grammar = common_build_tool_dsl_grammar({contract}, common_tool_dsl_grammar_form::action, error);
    if (out.grammar.empty()) return false;
    const auto fields = common_render_compact_tool_description(name, contract.description,
        contract.parameters, "{}", error);
    if (!error.empty()) return false;
    out.instructions = "Return flat DSL only: open! " + name +
        " followed by one field per line. Quote string values. Contract:\n" + fields;
    return true;
}

bool parse_agent_selection_response(common_agent_tool_output_format format,
        const std::string & name, const json & schema, const std::string & text,
        json & value, std::string & error) {
    value = json(); error.clear();
    std::string canonical = text;
    if (format == common_agent_tool_output_format::compact_dsl) {
        common_agent_tool_call call;
        if (!common_parse_model_tool_call(format, text,
                {common_chat_tool{name, "Selection decision", schema.dump()}}, call, error)) return false;
        if (call.name != name) { error = "selection returned an unexpected decision contract"; return false; }
        canonical = call.arguments_json;
    }
    std::string normalized;
    if (!common_schema_normalize_and_validate_object(canonical, schema.dump(), normalized, error)) return false;
    value = json::parse(normalized);
    return true;
}
