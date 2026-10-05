#include "agent/adaptation/flydelta/oracles/tool-contract.h"

#include "agent/tooling/contracts/schema-contract.h"

#include <algorithm>
#include <nlohmann/json.hpp>
#include <utility>

using json = nlohmann::ordered_json;

namespace {

void add_check(
        common_flydelta_oracle_result & result,
        const char * code,
        common_flydelta_oracle_verdict verdict,
        const char * dimension) {
    common_flydelta_oracle_check check;
    check.code = code;
    check.verdict = verdict;
    if (dimension != nullptr && *dimension != '\0') check.dimensions.emplace_back(dimension);
    result.checks.push_back(std::move(check));
}

void set_violation(
        common_flydelta_oracle_result & result,
        const char * code,
        const char * dimension) {
    result.known = true;
    result.verdict = common_flydelta_oracle_verdict::violated;
    result.violation_kind = common_flydelta_oracle_violation_kind::contract_violation;
    result.violation_code = code;
    if (dimension != nullptr && *dimension != '\0') {
        result.violation_dimensions.emplace_back(dimension);
    }
    add_check(result, code, common_flydelta_oracle_verdict::violated, dimension);
}

const char * argument_violation_code(const std::string & error) {
    if (error.find("required contract field is missing") != std::string::npos) {
        return "tool.missing_required_parameter";
    }
    if (error.find("unexpected contract field") != std::string::npos) {
        return "tool.unknown_argument";
    }
    if (error.find("invalid type") != std::string::npos) {
        return "tool.invalid_argument_type";
    }
    if (error.find("allowed value") != std::string::npos ||
            error.find("below its minimum") != std::string::npos ||
            error.find("exceeds its maximum") != std::string::npos ||
            error.find("too short") != std::string::npos ||
            error.find("too long") != std::string::npos ||
            error.find("too few items") != std::string::npos ||
            error.find("too many items") != std::string::npos) {
        return "tool.invalid_argument_value";
    }
    return "tool.invalid_arguments";
}

void initialize_result(
        const common_flydelta_model_tool_contract & contract,
        common_flydelta_oracle_result & result) {
    result = {};
    result.oracle_ref = "flydelta://oracle/tool-contract";
    result.oracle_revision = contract.contract_revision.empty()
        ? "v1" : contract.contract_revision;
    result.evaluator_ref = "flydelta://evaluator/tool-contract";
    result.evaluator_revision = "v1";
}

} // namespace

bool common_flydelta_parse_model_tool_text(
        common_agent_tool_output_format format,
        const std::string & observed,
        common_agent_tool_call & call,
        std::string & error) {
    error.clear();
    if (format == common_agent_tool_output_format::native) {
        error = "native calls must be supplied as parsed agent_tool_call values";
        return false;
    }
    std::string parse_error;
    if (common_parse_model_tool_call(format, observed, call, parse_error)) return true;

    // Server-context verification canonicalizes native parser output as
    // {"name":...,"arguments":...}; this is a bounded transport form, not
    // a second model-facing contract.
    if (format == common_agent_tool_output_format::jsonl) {
        const auto value = json::parse(observed, nullptr, false);
        if (value.is_object() && value.contains("name") && value["name"].is_string() &&
                value.contains("arguments") && value["arguments"].is_object()) {
            call.name = value["name"].get<std::string>();
            call.arguments_json = value["arguments"].dump();
            return true;
        }
    }
    error = parse_error;
    return false;
}

bool common_flydelta_validate_model_tool_call(
        const common_flydelta_model_tool_contract & contract,
        const common_agent_tool_call & call,
        common_flydelta_oracle_result & result,
        std::string & error) {
    error.clear();
    initialize_result(contract, result);
    result.confidence = 1.0f;

    if (contract.exposed_tool_name.empty() || contract.model_input_schema_json.empty()) {
        result.verdict = common_flydelta_oracle_verdict::unknown;
        result.reason = "model-facing tool contract is incomplete";
        add_check(result, "tool.contract_available",
            common_flydelta_oracle_verdict::unknown, "contract");
        return true;
    }

    if (call.name != contract.exposed_tool_name) {
        set_violation(result, "tool.unknown_tool", "tool_identity");
        result.reason = "observed tool is not the exposed model-facing tool";
        return true;
    }
    add_check(result, "tool.resolve", common_flydelta_oracle_verdict::satisfied, "tool_identity");

    std::string normalized_model_arguments;
    std::string validation_error;
    if (!common_schema_normalize_and_validate_object(
            call.arguments_json, contract.model_input_schema_json,
            normalized_model_arguments, validation_error)) {
        set_violation(result, argument_violation_code(validation_error), "arguments");
        result.reason = validation_error;
        return true;
    }

    std::string normalized_arguments = normalized_model_arguments;
    if (!contract.host_input_schema_json.empty() &&
            !common_schema_normalize_and_validate_object(
                normalized_model_arguments, contract.host_input_schema_json,
                normalized_arguments, validation_error)) {
        const char * code = validation_error.find("required contract field is missing") !=
                std::string::npos
            ? "tool.host_requirement_missing" : "tool.host_validation";
        set_violation(result, code, "host_contract");
        result.reason = validation_error;
        return true;
    }
    result.normalized_arguments_json = normalized_arguments;
    add_check(result, "tool.arguments", common_flydelta_oracle_verdict::satisfied, "arguments");

    if (contract.uses_network && !contract.host_allows_network) {
        set_violation(result, "tool.policy_rejected", "policy");
        result.reason = "host policy does not allow this network tool";
        return true;
    }
    if (contract.requires_confirmation && !contract.confirmation_satisfied) {
        set_violation(result, "tool.confirmation_required", "policy");
        result.reason = "tool requires host confirmation before execution";
        return true;
    }
    add_check(result, "tool.policy", common_flydelta_oracle_verdict::satisfied, "policy");

    result.known = true;
    result.verdict = common_flydelta_oracle_verdict::satisfied;
    result.reason = "model-facing tool call satisfies the host contract";
    return true;
}

bool common_flydelta_validate_model_tool_text(
        const common_flydelta_model_tool_contract & contract,
        common_agent_tool_output_format format,
        const std::string & observed,
        common_flydelta_oracle_result & result,
        std::string & error) {
    error.clear();
    if (format == common_agent_tool_output_format::native) {
        initialize_result(contract, result);
        result.confidence = 1.0f;
        result.known = false;
        result.verdict = common_flydelta_oracle_verdict::unknown;
        result.reason = "native calls must use the parsed-call validator";
        add_check(result, "tool.native_call_binding",
            common_flydelta_oracle_verdict::unknown, "protocol");
        return true;
    }

    common_agent_tool_call call;
    std::string parse_error;
    if (!common_flydelta_parse_model_tool_text(format, observed, call, parse_error)) {
        initialize_result(contract, result);
        result.known = true;
        result.confidence = 1.0f;
        set_violation(result, "tool.malformed_call", "protocol");
        result.reason = parse_error;
        return true;
    }
    return common_flydelta_validate_model_tool_call(contract, call, result, error);
}

common_flydelta_oracle_evaluator common_flydelta_make_tool_contract_oracle(
        common_flydelta_model_tool_contract contract) {
    return [contract = std::move(contract)](
            const common_flydelta_oracle_request & request,
            const std::string & observed,
            common_flydelta_oracle_result & result,
            std::string & error) {
        if (request.semantic_kind != "tool_contract" &&
                request.semantic_kind != "tool") return false;
        if (!request.expected_contract_ref.empty() &&
                request.expected_contract_ref != contract.contract_ref) return false;
        if (!request.expected_contract_revision.empty() &&
                request.expected_contract_revision != contract.contract_revision) return false;
        common_agent_tool_output_format format;
        if (!common_parse_agent_tool_output_format(
                request.observed_format.empty() ? "jsonl" : request.observed_format,
                format, error)) {
            result = {};
            result.oracle_ref = request.oracle_ref.empty()
                ? "flydelta://oracle/tool-contract" : request.oracle_ref;
            result.oracle_revision = request.oracle_revision.empty()
                ? contract.contract_revision : request.oracle_revision;
            result.policy_revision = request.policy_revision;
            result.known = false;
            result.verdict = common_flydelta_oracle_verdict::unknown;
            result.reason = error;
            error.clear();
            return true;
        }
        if (!common_flydelta_validate_model_tool_text(contract, format, observed, result, error)) {
            return false;
        }
        result.oracle_ref = request.oracle_ref.empty()
            ? "flydelta://oracle/tool-contract" : request.oracle_ref;
        result.oracle_revision = request.oracle_revision.empty()
            ? contract.contract_revision : request.oracle_revision;
        result.policy_revision = request.policy_revision;
        return true;
    };
}
