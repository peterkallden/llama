#include "agent/tooling/schema/tool-dsl-grammar.h"

#include <nlohmann/json.hpp>
#include <algorithm>
#include <map>
#include <set>
#include <sstream>

namespace {
using json = nlohmann::ordered_json;

std::string literal(const std::string & text) {
    return json(text).dump();
}

// Bare strings must not lex as a boolean/null. A small prefix trie excludes
// those tokens without changing the codec or restricting ordinary identifiers.
std::string plain_string_rules() {
    const std::vector<std::string> reserved = {"true", "false", "null"};
    const std::string initial = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_./:";
    const std::string rest = initial + "0123456789-";
    std::ostringstream out;
    const auto visit = [&](const auto & self, const std::string & prefix) -> void {
        const auto name = prefix.empty() ? std::string("plain-string") : "plain-" + prefix;
        std::set<char> next;
        bool terminal = false;
        for (const auto & word : reserved) {
            if (word == prefix) terminal = true;
            if (word.size() > prefix.size() && word.compare(0, prefix.size(), prefix) == 0) next.insert(word[prefix.size()]);
        }
        std::string safe;
        for (char ch : prefix.empty() ? initial : rest) {
            if (!next.count(ch)) safe += ch == '-' ? "\\-" : std::string(1, ch);
        }
        out << name << " ::= [" << safe << "] plain-tail*";
        if (!prefix.empty() && !terminal) out << " | \"\"";
        for (char ch : next) out << " | " << literal(std::string(1, ch)) << " plain-" << prefix << ch;
        out << '\n';
        for (char ch : next) self(self, prefix + ch);
    };
    visit(visit, {});
    out << "plain-tail ::= [a-zA-Z0-9_./:\\-]\n";
    return out.str();
}

// A field group is emitted in schema order. Only arrays and dynamic paths may
// repeat; ordinary scalar properties never share the old unrestricted loop.
struct field_group {
    std::string expression;
    bool required = false;
};

class contract_grammar {
public:
    std::ostringstream rules;
    std::string error;
    explicit contract_grammar(std::string prefix) : prefix(std::move(prefix)) {}

    std::string first(const std::string & name) const { return name + "-first"; }

    std::string rule(const std::string & expression, const std::string & initial = {}) {
        if (++count > 12000) { error = "DSL schema projection exceeds its rule limit"; return "invalid"; }
        const auto name = prefix + "contract-" + std::to_string(count);
        const auto normal = expression.empty() ? "\"\"" : expression;
        rules << name << " ::= " << normal << '\n';
        rules << first(name) << " ::= " << (initial.empty() ? normal : initial) << '\n';
        return name;
    }

    std::string line(const std::string & content) {
        return rule("\"\\n\" " + content, content);
    }

    std::string repeat(const std::string & value) {
        return rule(value + " (" + value + ")*", first(value) + " (" + value + ")*");
    }

    std::string scalar(const json & schema) {
        std::string value;
        if (schema.contains("const") || schema.contains("enum")) {
            const auto values = schema.contains("const") ? json::array({schema["const"]}) : schema["enum"];
            if (!values.is_array() || values.empty() || values.size() > 2048) {
                error = "DSL enum must be a bounded nonempty array";
                return "invalid";
            }
            for (const auto & item : values) {
                if (item.is_object() || item.is_array()) { error = "DSL enum requires scalar values"; return "invalid"; }
                if (!value.empty()) value += " | ";
                // Always accept the unambiguous quoted representation of a
                // string; plain enum tokens are allowed only when the codec
                // cannot interpret them as numbers, booleans or null.
                value += literal(item.dump());
                if (item.is_string()) {
                    const auto text = item.get<std::string>();
                    if (!text.empty() && text.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_./:-") == std::string::npos &&
                            json::parse(text, nullptr, false).is_discarded()) value += " | " + literal(text);
                }
            }
        } else {
            auto types = schema.value("type", json());
            if (types.is_string()) types = json::array({types});
            if (types.is_null()) return "scalar-value";
            if (!types.is_array() || types.empty()) { error = "invalid DSL scalar type"; return "invalid"; }
            for (const auto & type : types) {
                if (!type.is_string()) { error = "invalid DSL type alternative"; return "invalid"; }
                std::string next;
                if (type == "string") next = "string-value";
                else if (type == "boolean") next = "boolean-value";
                else if (type == "integer") next = "integer-value";
                else if (type == "number") next = "number-value";
                else if (type == "null") next = "\"null\"";
                else { error = "unsupported DSL scalar type: " + type.dump(); return "invalid"; }
                if (!value.empty()) value += " | ";
                value += next;
            }
        }
        // The planner's existing model-validation projection defers string
        // references. It does not defer numeric, enum or whole-array values;
        // advertising those bindings would emit calls the host cannot admit.
        const auto type = schema.value("type", json());
        const bool string_binding = !schema.contains("enum") && !schema.contains("const") &&
            (type == "string" || (type.is_array() && std::find(type.begin(), type.end(), json("string")) != type.end()));
        return rule("(" + value + ")" + (string_binding ? " | binding" : ""));
    }

    // Keep the sequence linear rather than enumerating all field permutations.
    // Every expression in the returned sequence starts with a newline.
    std::string sequence(const std::vector<field_group> & fields, bool nonempty) {
        std::vector<std::string> suffix(fields.size() + 1);
        suffix.back() = rule("\"\"");
        for (size_t i = fields.size(); i-- > 0;) {
            suffix[i] = rule((fields[i].required ? fields[i].expression : "(" + fields[i].expression + ")?") + " " + suffix[i + 1]);
        }
        std::string initial, selected;
        bool nullable = true;
        for (size_t i = 0; i < fields.size(); ++i) {
            if (!initial.empty()) { initial += " | "; selected += " | "; }
            initial += first(fields[i].expression) + " " + suffix[i + 1];
            selected += fields[i].expression + " " + suffix[i + 1];
            if (fields[i].required) { nullable = false; break; }
        }
        if (nullable && !nonempty) {
            if (!initial.empty()) initial += " | ";
            initial += "\"\"";
        }
        if (initial.empty()) { error = "DSL object has no representable fields"; return "invalid"; }
        return rule(nonempty ? selected : suffix.front(), initial);
    }

    std::string dictionary_key(const json & properties) {
        std::vector<std::string> excluded;
        for (const auto & property : properties.items()) excluded.push_back(property.key());
        const auto cache_key = json(excluded).dump();
        const auto cached = dictionary_keys.find(cache_key);
        if (cached != dictionary_keys.end()) return cached->second;
        const std::string alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-";
        const auto visit = [&](const auto & self, const std::string & prefix) -> std::string {
            if (prefix.size() > 256) { error = "DSL dictionary key exceeds its projection limit"; return "invalid"; }
            std::set<char> next;
            bool terminal = false;
            for (const auto & name : excluded) {
                if (name == prefix) terminal = true;
                if (name.size() > prefix.size() && name.compare(0, prefix.size(), prefix) == 0 &&
                        alphabet.find(name[prefix.size()]) != std::string::npos) next.insert(name[prefix.size()]);
            }
            std::string safe;
            for (char ch : alphabet) if (!next.count(ch)) safe += ch == '-' ? "\\-" : std::string(1, ch);
            std::string choices = safe.empty() ? "" : "[" + safe + "] dictionary-char*";
            if (!prefix.empty() && !terminal) {
                if (!choices.empty()) choices += " | ";
                choices += "\"\"";
            }
            for (char ch : next) {
                const auto tail = self(self, prefix + ch);
                if (!choices.empty()) choices += " | ";
                choices += literal(std::string(1, ch)) + " " + tail;
            }
            return rule(choices);
        };
        const auto result = visit(visit, {});
        dictionary_keys.emplace(cache_key, result);
        return result;
    }

    std::string object(const json & schema, const std::string & path, size_t depth = 0, bool nonempty = false) {
        if (!schema.is_object() || depth > 12) { error = "invalid or excessively nested DSL object schema"; return "invalid"; }
        // Branch-specific field sets and requiredness are not flattened into a
        // permissive union. allOf is merged only for compatible object members.
        for (const char * key : {"oneOf", "anyOf", "allOf"}) {
            if (!schema.contains(key)) continue;
            const auto & branches = schema[key];
            if (!branches.is_array() || branches.empty() || branches.size() > 64) {
                error = "DSL alternatives must be bounded nonempty arrays"; return "invalid";
            }
            auto base = schema;
            base.erase(key);
            if (std::string(key) == "allOf") {
                for (const auto & branch : branches) if (!merge(base, branch)) return "invalid";
                return object(base, path, depth + 1, nonempty);
            }
            std::string choices, initial;
            for (const auto & branch : branches) {
                auto candidate = base;
                if (!merge(candidate, branch)) return "invalid";
                const auto branch_rule = object(candidate, path, depth + 1, nonempty);
                if (!choices.empty()) { choices += " | "; initial += " | "; }
                choices += branch_rule;
                initial += first(branch_rule);
            }
            return rule(choices, initial);
        }
        const auto properties = schema.value("properties", json::object());
        const auto required = schema.value("required", json::array());
        if (!properties.is_object() || !required.is_array()) { error = "invalid DSL properties or required list"; return "invalid"; }
        std::set<std::string> mandatory;
        for (const auto & name : required) {
            if (!name.is_string() || !properties.contains(name.get<std::string>())) {
                error = "DSL required field has no declared property"; return "invalid";
            }
            mandatory.insert(name.get<std::string>());
        }
        std::vector<field_group> fields;
        for (const auto & property : properties.items()) {
            if (property.key().empty() || property.key().find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-") != std::string::npos) {
                error = "DSL schema property is not a field identifier"; return "invalid";
            }
            const auto child = path.empty() ? literal(property.key()) : path + " " + literal("." + property.key());
            fields.push_back({field(property.value(), child, depth + 1), mandatory.count(property.key()) != 0});
        }
        // Root additions were never model-visible fields; nested dictionaries
        // remain supported and their keys/duplicates are validated by the codec.
        if (!path.empty() && schema.value("additionalProperties", json(true)) != false) {
            const auto additional = schema.value("additionalProperties", json::object());
            const auto value = additional.is_object() ? scalar(additional) : "scalar-value";
            const auto key = dictionary_key(properties);
            fields.push_back({repeat(line(path + " \".\" " + key + " dynamic-tail \": \" " + value)), false});
        }
        return sequence(fields, nonempty);
    }

    std::string field(const json & schema, const std::string & path, size_t depth) {
        if (!schema.is_object() || depth > 12) { error = "invalid or excessively nested DSL field schema"; return "invalid"; }
        if (schema.contains("type") && schema["type"].is_array()) {
            if (schema["type"].empty() || schema["type"].size() > 7) { error = "invalid DSL type union"; return "invalid"; }
            std::string choices, initial;
            for (const auto & type : schema["type"]) {
                if (!type.is_string()) { error = "invalid DSL type alternative"; return "invalid"; }
                auto candidate = schema; candidate["type"] = type;
                // An object/array's member annotations do not change the null
                // alternative into a structured field.
                if (type != "object") candidate.erase("properties");
                if (type != "array") candidate.erase("items");
                const auto branch = field(candidate, path, depth + 1);
                if (!choices.empty()) { choices += " | "; initial += " | "; }
                choices += branch; initial += first(branch);
            }
            return rule(choices, initial);
        }
        if (schema.contains("allOf")) {
            if (!schema["allOf"].is_array() || schema["allOf"].empty()) { error = "invalid DSL field allOf"; return "invalid"; }
            auto combined = schema; combined.erase("allOf");
            for (const auto & branch : schema["allOf"]) if (!merge(combined, branch)) return "invalid";
            return field(combined, path, depth + 1);
        }
        if (schema.value("type", json()) == "object" || schema.contains("properties")) return object(schema, path, depth, true);
        if (schema.value("type", json()) == "array" || (!schema.contains("type") && schema.contains("items"))) {
            const auto item = schema.value("items", json::object());
            const auto annotation = schema.value("x-agent-flat", json::object());
            if (!annotation.is_object()) { error = "invalid DSL flat annotation"; return "invalid"; }
            std::string directive;
            if (annotation.contains("directive")) {
                if (!annotation["directive"].is_string() || annotation["directive"].get<std::string>().empty()) {
                    error = "invalid DSL flat directive"; return "invalid";
                }
                // Positional/named directive parsing remains owned by the codec.
                directive = repeat(line(literal(annotation["directive"].get<std::string>()) + " \": \" directive-value"));
            }
            if (item.value("type", json()) == "object" || item.contains("properties") || item.contains("oneOf") || item.contains("anyOf")) {
                std::string indexed;
                // Decimal index continuity is not context-free in general.
                // For modest declared bounds, generate a linear chain with
                // literal indices instead of a repeatable generic index.
                // Nested, larger and unbounded lists retain codec-checked
                // indexing to avoid multiplying schema expansion costs.
                const auto maximum = schema.value("maxItems", json());
                const auto minimum = schema.value("minItems", json(0));
                if (array_expansion_depth == 0 && maximum.is_number_integer() && maximum >= 0 && maximum.get<size_t>() <= 64) {
                    const size_t count = maximum.get<size_t>();
                    if (!minimum.is_number_integer() || minimum < 0 || minimum.get<size_t>() > count) {
                        error = "invalid DSL object-array bounds"; return "invalid";
                    }
                    const size_t required = std::max<size_t>(minimum.get<size_t>(), 1);
                    std::vector<std::string> entries;
                    ++array_expansion_depth;
                    for (size_t i = 0; i < count; ++i) {
                        entries.push_back(object(item, path + " " + literal("[" + std::to_string(i) + "]"), depth + 1, true));
                    }
                    --array_expansion_depth;
                    std::string tail, initial;
                    for (size_t i = count; i-- > 0;) {
                        const std::string rest = tail.empty() ? "" :
                            (i + 1 < required ? " " + tail : " (" + tail + ")?");
                        initial = first(entries[i]) + rest;
                        tail = rule(entries[i] + rest, initial);
                    }
                    if (minimum == 0) {
                        const auto empty = line(path + " \": []\"");
                        if (tail.empty()) tail = empty;
                        else tail = rule(tail + " | " + empty, first(tail) + " | " + first(empty));
                    }
                    indexed = tail;
                } else {
                    indexed = repeat(object(item, path + " \"[\" index \"]\"", depth + 1, true));
                }
                return directive.empty() ? indexed :
                    rule(indexed + " | " + directive, first(indexed) + " | " + first(directive));
            }
            if (!directive.empty()) { error = "DSL flat directive requires object array items"; return "invalid"; }
            const auto value = scalar(item);
            const auto list = rule("\"[\" (" + value + " (\", \" " + value + ")*)? \"]\"");
            const auto bracketed = line(path + " \": \" " + list);
            const auto repeated = repeat(line(path + " \": \" " + value));
            const auto comma = line(path + " \": \" " + value + " (\", \" " + value + ")+");
            return rule(bracketed + " | " + repeated + " | " + comma,
                first(bracketed) + " | " + first(repeated) + " | " + first(comma));
        }
        if (schema.contains("oneOf") || schema.contains("anyOf")) {
            const char * key = schema.contains("oneOf") ? "oneOf" : "anyOf";
            if (!schema[key].is_array() || schema[key].empty()) { error = "invalid DSL field alternatives"; return "invalid"; }
            std::string choices, initial;
            for (const auto & branch : schema[key]) {
                auto candidate = schema; candidate.erase(key);
                if (!merge(candidate, branch)) return "invalid";
                const auto branch_rule = field(candidate, path, depth + 1);
                if (!choices.empty()) { choices += " | "; initial += " | "; }
                choices += branch_rule; initial += first(branch_rule);
            }
            return rule(choices, initial);
        }
        return line(path + " \": \" " + scalar(schema));
    }

private:
    size_t count = 0;
    size_t array_expansion_depth = 0;
    std::string prefix;
    std::map<std::string, std::string> dictionary_keys;
    bool merge(json & base, const json & other) {
        if (!other.is_object()) { error = "DSL alternative must be an object"; return false; }
        for (const auto & entry : other.items()) {
            if (entry.key() == "required") {
                if (!entry.value().is_array()) { error = "invalid DSL required list"; return false; }
                if (!base.contains("required")) base["required"] = json::array();
                if (!base["required"].is_array()) { error = "invalid DSL required list"; return false; }
                for (const auto & name : entry.value()) {
                    if (std::find(base["required"].begin(), base["required"].end(), name) == base["required"].end()) base["required"].push_back(name);
                }
            } else if (entry.key() == "properties") {
                if (!entry.value().is_object()) { error = "invalid DSL properties"; return false; }
                if (!base.contains("properties")) base["properties"] = json::object();
                for (const auto & member : entry.value().items()) {
                    if (base["properties"].contains(member.key())) {
                        if (!merge(base["properties"][member.key()], member.value())) return false;
                    } else base["properties"][member.key()] = member.value();
                }
            } else if (entry.key() == "enum" && base.contains("enum")) {
                if (!base["enum"].is_array() || !entry.value().is_array()) { error = "invalid DSL enum intersection"; return false; }
                json values = json::array();
                for (const auto & value : base["enum"]) {
                    if (std::find(entry.value().begin(), entry.value().end(), value) != entry.value().end()) values.push_back(value);
                }
                if (values.empty()) { error = "DSL enum intersection is empty"; return false; }
                base["enum"] = std::move(values);
            } else if (!base.contains(entry.key())) base[entry.key()] = entry.value();
            else if (base[entry.key()] != entry.value() && entry.key().rfind("x-", 0) != 0 && entry.key() != "description" && entry.key() != "title") {
                error = "DSL intersection contains conflicting constraints: " + entry.key(); return false;
            }
        }
        return true;
    }
};

} // namespace

static std::string build_tool_dsl_grammar(
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
        contract_grammar projection("tool-" + std::to_string(index) + "-");
        const auto body = projection.object(schema, {});
        if (!projection.error.empty()) {
            error = projection.error + " (tool: " + tool.name + ")";
            return {};
        }
        rules << projection.rules.str();
        rules << "call-" << index << " ::= " << literal("open! " + tool.name) << " " << body << '\n';
        if (form == common_tool_dsl_grammar_form::arguments) {
            rules << "root ::= " << projection.first(body) << " \"\\n\"?\n";
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
    rules << plain_string_rules();
    rules << R"gbnf(alias ::= [a-zA-Z0-9_.-]{1,64}
index ::= "0" | [1-9] [0-9]*
dictionary-char ::= [a-zA-Z0-9_\-]
dynamic-tail ::= ("." dictionary-char+ | "[" index "]")*
binding ::= "$" [a-zA-Z_] [a-zA-Z0-9_.-]*
quoted-string ::= "\"" char* "\""
string-value ::= quoted-string | plain-string
boolean-value ::= "true" | "false"
integer-value ::= "-"? ("0" | [1-9] [0-9]*)
number-value ::= integer-value ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
scalar-value ::= string-value | number-value | boolean-value | "null" | binding
directive-value ::= [^\n\r]+
char ::= [^"\\\x00-\x1f] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F]{4})
)gbnf";
    return rules.str();
}

std::string common_build_tool_dsl_grammar(
        const std::vector<common_chat_tool> & tools, common_tool_dsl_grammar_form form,
        std::string & error, size_t max_steps, size_t max_goal_chars) {
    try {
        auto grammar = build_tool_dsl_grammar(tools, form, error, max_steps, max_goal_chars);
        if (grammar.size() > 4 * 1024 * 1024) {
            error = "DSL grammar exceeds its serialized size limit";
            return {};
        }
        return grammar;
    } catch (const json::exception & failure) {
        error = std::string("invalid DSL schema: ") + failure.what();
        return {};
    }
}
