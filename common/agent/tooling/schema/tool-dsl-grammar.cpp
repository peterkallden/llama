#include "agent/tooling/schema/tool-dsl-grammar.h"

#include <nlohmann/json.hpp>
#include <set>
#include <sstream>

namespace {
using json = nlohmann::ordered_json;

std::string literal(const std::string & text) {
    return json(text).dump();
}

bool collect_fields(const json & schema, const std::string & path, size_t depth,
        std::set<std::string> & fields, std::string & error) {
    if (depth > 12 || fields.size() > 2048) {
        error = "DSL schema projection exceeds its depth or field limit";
        return false;
    }
    if (!path.empty()) fields.insert(path);
    if (!schema.is_object()) return true;
    for (const auto * combinator : {"oneOf", "anyOf", "allOf"}) {
        const auto alternatives = schema.find(combinator);
        if (alternatives == schema.end()) continue;
        if (!alternatives->is_array()) {
            error = "DSL schema alternatives must be arrays";
            return false;
        }
        for (const auto & alternative : *alternatives) {
            if (!collect_fields(alternative, path, depth + 1, fields, error)) return false;
        }
    }
    const auto properties = schema.find("properties");
    if (properties != schema.end()) {
        if (!properties->is_object()) {
            error = "DSL schema properties must be an object";
            return false;
        }
        for (const auto & property : properties->items()) {
            if (property.key().empty() || property.key().find_first_not_of(
                    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-") != std::string::npos) {
                error = "DSL schema property is not a field identifier";
                return false;
            }
            const auto child = path.empty() ? literal(property.key()) : path + " " + literal("." + property.key());
            if (!collect_fields(property.value(), child, depth + 1, fields, error)) return false;
            if (path.empty() && property.value().is_object()) {
                const auto annotation = property.value().find("x-agent-flat");
                if (annotation != property.value().end() && annotation->is_object()) {
                    const auto directive = annotation->find("directive");
                    if (directive != annotation->end() && directive->is_string() && !directive->get_ref<const std::string &>().empty()) {
                        fields.insert(literal(directive->get<std::string>()));
                    }
                }
            }
        }
    }
    const auto type = schema.find("type");
    const bool object = (type != schema.end() && *type == "object") || properties != schema.end();
    if (!path.empty() && object) {
        const auto additional = schema.find("additionalProperties");
        if (additional == schema.end() || *additional != false) fields.insert(path + " dynamic-path");
    }
    const auto items = schema.find("items");
    if (!path.empty() && items != schema.end()) {
        if (!collect_fields(*items, path + " \"[\" index \"]\"", depth + 1, fields, error)) return false;
    }
    return true;
}
} // namespace

std::string common_build_tool_dsl_grammar(
        const std::vector<common_chat_tool> & tools, common_tool_dsl_grammar_form form,
        std::string & error, size_t max_steps, size_t max_goal_chars) {
    error.clear();
    if (tools.empty() || tools.size() > 256 || max_steps == 0 || max_steps > 6 ||
            max_goal_chars == 0 || max_goal_chars > 256 ||
            (form == common_tool_dsl_grammar_form::arguments && tools.size() != 1)) {
        error = "invalid DSL grammar catalog, form or plan bounds";
        return {};
    }
    std::ostringstream rules;
    std::set<std::string> names;
    for (size_t index = 0; index < tools.size(); ++index) {
        const auto & tool = tools[index];
        const auto schema = json::parse(tool.parameters, nullptr, false);
        if (tool.name.empty() || !names.insert(tool.name).second || !schema.is_object() ||
                !schema.contains("type") || schema["type"] != "object") {
            error = "invalid DSL input contract for tool: " + tool.name;
            return {};
        }
        std::set<std::string> fields;
        if (!collect_fields(schema, {}, 0, fields, error)) {
            error += " (tool: " + tool.name + ")";
            return {};
        }
        const auto suffix = std::to_string(index);
        rules << "call-" << suffix << " ::= " << literal("open! " + tool.name);
        if (!fields.empty()) rules << " (\"\\n\" field-" << suffix << ")*";
        rules << '\n';
        if (!fields.empty()) {
            rules << "field-" << suffix << " ::= (";
            bool first = true;
            for (const auto & field : fields) {
                if (!first) rules << " | ";
                first = false;
                rules << field;
            }
            rules << ") \": \" value\n";
        }
        if (form == common_tool_dsl_grammar_form::arguments) {
            rules << "root ::= ";
            if (fields.empty()) rules << "\"\"\n";
            else rules << "field-0 (\"\\n\" field-0)* \"\\n\"?\n";
        }
    }
    if (form != common_tool_dsl_grammar_form::arguments) {
        rules << "call ::= ";
        for (size_t index = 0; index < tools.size(); ++index) {
            if (index) rules << " | ";
            rules << "call-" << index;
        }
        rules << '\n';
        if (form == common_tool_dsl_grammar_form::action) {
            rules << "root ::= call \"\\n\"?\n";
        } else {
            rules << "root ::= \"plan goal=\\\"\" char{1," << max_goal_chars << "} \"\\\"\\n\" step";
            if (max_steps > 1) rules << " (\"\\n\" step){0," << max_steps - 1 << "}";
            rules << " \"\\n\"?\nstep ::= \"step\" (\" as=\" alias)? \" | \" call\n";
        }
    }
    rules << R"gbnf(alias ::= [a-zA-Z0-9_.-]{1,64}
index ::= [0-9]+
dynamic-path ::= "." [a-zA-Z0-9_-]+ ("." [a-zA-Z0-9_-]+ | "[" index "]")*
value ::= [^\n\r]+
char ::= [^"\\\x00-\x1f] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F]{4})
)gbnf";
    return rules.str();
}
