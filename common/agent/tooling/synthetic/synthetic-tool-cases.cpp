#include "agent/tooling/synthetic/synthetic-tool-cases.h"

#include "agent/tooling/contracts/schema-contract.h"
#include "agent/tooling/schema/tool-schema-compact.h"

#include <nlohmann/json.hpp>

#include <fstream>
#include <iomanip>
#include <set>
#include <sstream>

using json = nlohmann::ordered_json;

namespace {

bool bounded_nonempty(const std::string & value, size_t max_size) {
    return !value.empty() && value.size() <= max_size;
}

uint64_t fnv1a(const std::string & value) {
    uint64_t hash = 1469598103934665603ULL;
    for (const unsigned char byte : value) {
        hash ^= byte;
        hash *= 1099511628211ULL;
    }
    return hash;
}

std::string fingerprint(const std::string & value) {
    std::ostringstream out;
    out << "synthetic:fnv1a64:" << std::hex << std::setw(16) << std::setfill('0') << fnv1a(value);
    return out.str();
}

json sample_for_schema(const json & schema, const std::string & field, size_t depth = 0) {
    if (schema.contains("default")) return schema["default"];
    if (schema.contains("enum") && schema["enum"].is_array() && !schema["enum"].empty()) {
        return schema["enum"].front();
    }
    const auto type = schema.value("type", "string");
    if (type == "string") return field + "-sample";
    if (type == "integer") return schema.contains("minimum") ? schema["minimum"] : json(1);
    if (type == "number") return schema.contains("minimum") ? schema["minimum"] : json(1.0);
    if (type == "boolean") return true;
    if (type == "array") {
        json result = json::array();
        if (depth < 2 && schema.contains("items")) result.push_back(sample_for_schema(schema["items"], field, depth + 1));
        return result;
    }
    if (type == "object") {
        json result = json::object();
        const auto properties = schema.value("properties", json::object());
        const auto required = schema.value("required", json::array());
        if (properties.is_object() && required.is_array()) {
            for (const auto & item : required) {
                if (!item.is_string() || !properties.contains(item.get<std::string>())) continue;
                result[item.get<std::string>()] = sample_for_schema(
                    properties[item.get<std::string>()], item.get<std::string>(), depth + 1);
            }
        }
        return result;
    }
    return "sample";
}

bool parse_object(const std::string & text, json & value) {
    value = json::parse(text, nullptr, false);
    return value.is_object();
}

std::vector<common_synthetic_tool_mutation> supported_mutations(const json & schema) {
    std::vector<common_synthetic_tool_mutation> result;
    const auto properties = schema.value("properties", json::object());
    const auto required = schema.value("required", json::array());
    bool has_enum = false;
    if (properties.is_object()) {
        for (auto it = properties.begin(); it != properties.end(); ++it) {
            has_enum = has_enum || (it.value().contains("enum") && it.value()["enum"].is_array());
        }
    }
    if (required.is_array() && !required.empty()) result.push_back(common_synthetic_tool_mutation::missing_required);
    if (properties.is_object() && !properties.empty()) result.push_back(common_synthetic_tool_mutation::wrong_type);
    if (has_enum) result.push_back(common_synthetic_tool_mutation::invalid_enum);
    if (schema.value("additionalProperties", true) == false) result.push_back(common_synthetic_tool_mutation::unexpected_property);
    result.push_back(common_synthetic_tool_mutation::malformed_json);
    return result;
}

bool mutate_arguments(
        const json & schema,
        const json & baseline,
        common_synthetic_tool_mutation mutation,
        json & mutated,
        std::string & error) {
    mutated = baseline;
    const auto properties = schema.value("properties", json::object());
    const auto required = schema.value("required", json::array());
    if (!properties.is_object()) { error = "synthetic tool schema properties are invalid"; return false; }
    if (mutation == common_synthetic_tool_mutation::missing_required) {
        if (!required.is_array() || required.empty() || !required.front().is_string()) {
            error = "synthetic missing-required mutation has no required field";
            return false;
        }
        mutated.erase(required.front().get<std::string>());
        return true;
    }
    if (mutation == common_synthetic_tool_mutation::unexpected_property) {
        mutated["__synthetic_extra"] = true;
        return true;
    }
    if (mutation == common_synthetic_tool_mutation::malformed_json) return true;
    for (auto it = properties.begin(); it != properties.end(); ++it) {
        if (mutation == common_synthetic_tool_mutation::wrong_type) {
            const auto type = it.value().value("type", "string");
            if (type == "string") mutated[it.key()] = 17;
            else if (type == "integer" || type == "number") mutated[it.key()] = "wrong-type";
            else if (type == "boolean") mutated[it.key()] = "wrong-type";
            else mutated[it.key()] = "wrong-type";
            return true;
        }
        if (mutation == common_synthetic_tool_mutation::invalid_enum &&
                it.value().contains("enum") && it.value()["enum"].is_array()) {
            const auto type = it.value().value("type", "string");
            mutated[it.key()] = type == "integer" ? json(9223372036854775807LL) : json("__synthetic_invalid_enum__");
            return true;
        }
    }
    error = "synthetic mutation could not find a suitable field";
    return false;
}

} // namespace

const char * common_synthetic_tool_mutation_name(common_synthetic_tool_mutation mutation) {
    switch (mutation) {
        case common_synthetic_tool_mutation::missing_required: return "missing_required";
        case common_synthetic_tool_mutation::wrong_type: return "wrong_type";
        case common_synthetic_tool_mutation::invalid_enum: return "invalid_enum";
        case common_synthetic_tool_mutation::unexpected_property: return "unexpected_property";
        case common_synthetic_tool_mutation::malformed_json: return "malformed_json";
    }
    return "missing_required";
}

bool common_synthetic_tool_case_validate(
        const common_synthetic_tool_case & value,
        size_t max_text_size,
        std::string & error) {
    error.clear();
    if (value.schema_version != 1 || value.id.empty() || value.tool_name.empty() ||
            value.tool_family.empty() || value.provider_kind.empty() || value.seed == 0 ||
            !bounded_nonempty(value.host_input_schema_json, max_text_size) ||
            !bounded_nonempty(value.baseline_arguments_json, max_text_size) ||
            !bounded_nonempty(value.mutated_arguments_json, max_text_size) ||
            !bounded_nonempty(value.expected_arguments_json, max_text_size) ||
            !bounded_nonempty(value.model_input_schema_json, max_text_size) ||
            !bounded_nonempty(value.model_facing_contract, max_text_size) ||
            !bounded_nonempty(value.model_facing_prompt, max_text_size) ||
            !bounded_nonempty(value.schema_fingerprint, 256) ||
            !bounded_nonempty(value.verifier_revision, 256)) {
        error = "synthetic tool case identity or bounds are invalid";
        return false;
    }
    if (value.mutation == common_synthetic_tool_mutation::malformed_json) {
        if (value.mutated_arguments_json != "{\"name\":\"" + value.tool_name + "\",\"arguments\":") {
            error = "malformed synthetic tool case has unexpected mutation";
            return false;
        }
    } else {
        json parsed;
        if (!parse_object(value.baseline_arguments_json, parsed) || !parse_object(value.expected_arguments_json, parsed) ||
                !parse_object(value.mutated_arguments_json, parsed)) {
            error = "synthetic tool case arguments are not JSON objects";
            return false;
        }
    }
    if (!value.host_verified || value.mutated_valid || !value.repair_valid) {
        error = "synthetic tool case lacks host verification";
        return false;
    }
    return true;
}

bool common_synthetic_tool_cases_generate(
        const common_tool_definition & definition,
        const std::string & tool_family,
        const std::string & provider_kind,
        uint64_t seed,
        size_t max_cases,
        std::vector<common_synthetic_tool_case> & cases,
        std::string & error) {
    error.clear();
    cases.clear();
    if (definition.name.empty() || definition.input_schema_json.empty() || tool_family.empty() ||
            provider_kind.empty() || seed == 0 || max_cases == 0) {
        error = "synthetic tool generation identity or bounds are invalid";
        return false;
    }
    const auto schema = json::parse(definition.input_schema_json, nullptr, false);
    if (!schema.is_object() || schema.value("type", "") != "object") {
        error = "synthetic tool generation requires an object input schema";
        return false;
    }
    const auto baseline = sample_for_schema(schema, definition.name);
    std::string compact_error;
    const auto model_schema = common_tool_model_input_schema(definition);
    const auto compact = common_render_compact_tool_description(
        definition.name, definition.description, model_schema,
        common_tool_model_result_schema(definition), compact_error);
    if (!compact_error.empty() || compact.empty()) { error = compact_error; return false; }
    for (const auto mutation : supported_mutations(schema)) {
        if (cases.size() >= max_cases) break;
        common_synthetic_tool_case value;
        value.seed = seed;
        value.tool_name = definition.name;
        value.tool_family = tool_family;
        value.provider_kind = provider_kind;
        value.mutation = mutation;
        value.id = "synthetic://tool/" + definition.name + "/" + common_synthetic_tool_mutation_name(mutation);
        value.host_input_schema_json = schema.dump();
        value.baseline_arguments_json = baseline.dump();
        value.expected_arguments_json = baseline.dump();
        value.model_input_schema_json = model_schema;
        value.model_facing_contract = compact;
        value.model_facing_prompt = "Tool contract:\n" + compact +
            "\nRepair the invalid tool call using the contract.";
        value.schema_fingerprint = fingerprint(value.host_input_schema_json);
        json mutated;
        if (!mutate_arguments(schema, baseline, mutation, mutated, error)) return false;
        value.mutated_arguments_json = mutation == common_synthetic_tool_mutation::malformed_json
            ? "{\"name\":\"" + definition.name + "\",\"arguments\":" : mutated.dump();
        if (!common_synthetic_tool_case_verify(value, error)) return false;
        cases.push_back(std::move(value));
    }
    if (cases.empty()) { error = "synthetic tool schema produced no supported mutations"; return false; }
    return true;
}

bool common_synthetic_tool_case_verify(
        common_synthetic_tool_case & value,
        std::string & error) {
    error.clear();
    const auto schema = json::parse(value.host_input_schema_json, nullptr, false);
    if (!schema.is_object()) { error = "synthetic host schema is invalid"; return false; }
    std::string baseline_normalized;
    if (!common_schema_normalize_and_validate_object(
            value.baseline_arguments_json, value.host_input_schema_json, baseline_normalized, error)) {
        return false;
    }
    std::string expected_normalized;
    if (!common_schema_normalize_and_validate_object(
            value.expected_arguments_json, value.host_input_schema_json, expected_normalized, error)) {
        return false;
    }
    if (baseline_normalized != expected_normalized) {
        error = "synthetic repair does not preserve the valid baseline arguments";
        return false;
    }
    value.repair_valid = true;
    if (value.mutation == common_synthetic_tool_mutation::malformed_json) {
        value.mutated_valid = false;
    } else {
        std::string mutated_normalized;
        value.mutated_valid = common_schema_normalize_and_validate_object(
            value.mutated_arguments_json, value.host_input_schema_json, mutated_normalized, error);
        error.clear();
    }
    value.host_verified = value.repair_valid && !value.mutated_valid;
    if (!value.host_verified) { error = "synthetic mutation did not produce the expected host verdict"; return false; }
    return true;
}

std::string common_synthetic_tool_case_to_json(const common_synthetic_tool_case & value) {
    return json{
        {"schema_version", value.schema_version}, {"id", value.id}, {"seed", value.seed},
        {"tool_name", value.tool_name}, {"tool_family", value.tool_family},
        {"provider_kind", value.provider_kind},
        {"mutation", common_synthetic_tool_mutation_name(value.mutation)},
        {"host_input_schema", value.host_input_schema_json},
        {"baseline_arguments", value.baseline_arguments_json},
        {"mutated_arguments", value.mutated_arguments_json},
        {"expected_arguments", value.expected_arguments_json},
        {"model_input_schema", value.model_input_schema_json},
        {"model_facing_contract", value.model_facing_contract},
        {"model_facing_prompt", value.model_facing_prompt},
        {"schema_fingerprint", value.schema_fingerprint},
        {"verifier_revision", value.verifier_revision},
        {"mutated_valid", value.mutated_valid}, {"repair_valid", value.repair_valid},
        {"host_verified", value.host_verified},
    }.dump();
}

bool common_synthetic_tool_case_from_json(
        const std::string & text,
        size_t max_text_size,
        common_synthetic_tool_case & value,
        std::string & error) {
    error.clear();
    const auto parsed = json::parse(text, nullptr, false);
    if (!parsed.is_object()) { error = "synthetic tool case is not an object"; return false; }
    value = {};
    value.schema_version = parsed.value("schema_version", 0);
    value.id = parsed.value("id", "");
    value.seed = parsed.value("seed", 0ULL);
    value.tool_name = parsed.value("tool_name", "");
    value.tool_family = parsed.value("tool_family", "");
    value.provider_kind = parsed.value("provider_kind", "");
    const auto mutation = parsed.value("mutation", "");
    const std::vector<common_synthetic_tool_mutation> mutations = {
        common_synthetic_tool_mutation::missing_required,
        common_synthetic_tool_mutation::wrong_type,
        common_synthetic_tool_mutation::invalid_enum,
        common_synthetic_tool_mutation::unexpected_property,
        common_synthetic_tool_mutation::malformed_json,
    };
    bool found = false;
    for (const auto candidate : mutations) if (mutation == common_synthetic_tool_mutation_name(candidate)) {
        value.mutation = candidate; found = true; break;
    }
    if (!found) { error = "synthetic tool case mutation is unknown"; return false; }
    value.host_input_schema_json = parsed.value("host_input_schema", "");
    value.baseline_arguments_json = parsed.value("baseline_arguments", "");
    value.mutated_arguments_json = parsed.value("mutated_arguments", "");
    value.expected_arguments_json = parsed.value("expected_arguments", "");
    value.model_input_schema_json = parsed.value("model_input_schema", "");
    value.model_facing_contract = parsed.value("model_facing_contract", "");
    value.model_facing_prompt = parsed.value("model_facing_prompt", "");
    value.schema_fingerprint = parsed.value("schema_fingerprint", "");
    value.verifier_revision = parsed.value("verifier_revision", "");
    value.mutated_valid = parsed.value("mutated_valid", false);
    value.repair_valid = parsed.value("repair_valid", false);
    value.host_verified = parsed.value("host_verified", false);
    if (!common_synthetic_tool_case_verify(value, error)) return false;
    return common_synthetic_tool_case_validate(value, max_text_size, error);
}

bool common_synthetic_tool_cases_export_jsonl(
        const std::vector<common_synthetic_tool_case> & cases,
        const std::filesystem::path & path,
        size_t max_bytes,
        std::string & error) {
    error.clear();
    if (path.empty() || max_bytes == 0 || cases.empty()) { error = "synthetic JSONL export bounds are invalid"; return false; }
    std::ostringstream buffer;
    for (const auto & source : cases) {
        auto value = source;
        if (!common_synthetic_tool_case_verify(value, error) ||
                !common_synthetic_tool_case_validate(value, 64 * 1024, error)) return false;
        buffer << common_synthetic_tool_case_to_json(value) << '\n';
    }
    const auto text = buffer.str();
    if (text.size() > max_bytes) { error = "synthetic JSONL export exceeds byte bound"; return false; }
    std::error_code filesystem_error;
    if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path(), filesystem_error);
    if (filesystem_error) { error = filesystem_error.message(); return false; }
    std::ofstream output(path, std::ios::binary);
    if (!output) { error = "could not open synthetic JSONL export"; return false; }
    output << text;
    if (!output) { error = "could not write synthetic JSONL export"; return false; }
    return true;
}
