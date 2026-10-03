#include "../agent-flydelta-repair-smoke-support.h"

namespace agent_flydelta_repair_smoke_support {

bool parse_model_tool_call(
        const common_agent_generation_result & generation,
        json & parsed) {
    parsed = json::parse(generation.content, nullptr, false);
    if (parsed.is_discarded() && generation.chat_params) {
        common_chat_parser_params parser_params(*generation.chat_params);
        parser_params.parse_tool_calls = true;
        if (!generation.chat_params->parser.empty()) {
            parser_params.parser.load(generation.chat_params->parser);
        }
        const auto assistant = common_chat_parse(generation.content, false, parser_params);
        if (!assistant.tool_calls.empty()) {
            const auto & call = assistant.tool_calls.front();
            const auto arguments = json::parse(call.arguments, nullptr, false);
            if (!arguments.is_discarded()) {
                parsed = json{{"name", call.name}, {"arguments", arguments}};
            }
        }
    }
    if (parsed.is_discarded()) {
        const size_t first = generation.content.find('{');
        const size_t last = generation.content.rfind('}');
        if (first == std::string::npos || last <= first) return false;
        parsed = json::parse(generation.content.substr(first, last - first + 1), nullptr, false);
    }
    if (!parsed.is_object() || !parsed.contains("name") ||
            !parsed["name"].is_string()) return false;
    if (!parsed.contains("arguments") && parsed.contains("args") &&
            parsed["args"].is_object()) {
        parsed["arguments"] = parsed["args"];
    }
    return parsed.contains("arguments") && parsed["arguments"].is_object();
}

bool verify_model_tool_contract(
        const common_agent_generation_result & generation,
        const std::string & expected_tool,
        const json & expected_arguments,
        agent_flydelta_dataset_repair_host & host,
        host_tool_verdict & verdict,
        std::string & error) {
    error.clear();
    verdict = {};
    const std::string & canonical_expected_tool = expected_tool;
    common_flydelta_oracle_request request;
    request.oracle_ref = "flydelta://oracle/host-tool-contract";
    request.oracle_revision = "v1";
    request.policy_revision = "policy:canonical-tool-call-v1";
    request.phase = common_flydelta_oracle_phase::synthesis;
    request.concept_key = "structured_tool_selection";
    request.behavior_key = "structured_tool_selection";
    request.semantic_kind = "host_tool_contract";
    request.expected_decision_available = true;
    common_flydelta_oracle_evaluator_chain evaluators;
    evaluators.host_supported = [&, canonical_expected_tool, expected_arguments](
            const common_flydelta_oracle_request & oracle_request,
            const std::string &,
            common_flydelta_oracle_result & result,
            std::string & evaluator_error) {
        evaluator_error.clear();
        if (oracle_request.semantic_kind != "host_tool_contract") return false;
        result = {};
        result.strength = common_flydelta_oracle_strength::host_supported;
        result.oracle_ref = oracle_request.oracle_ref;
        result.oracle_revision = oracle_request.oracle_revision;
        result.policy_revision = oracle_request.policy_revision;
        result.evidence_ref = "evidence:host-tool-contract";
        json parsed;
        if (!common_agent_generation_succeeded(generation) ||
                !parse_model_tool_call(generation, parsed)) {
            result.known = true;
            result.verdict = common_flydelta_oracle_verdict::violated;
            result.confidence = 1.0f;
            result.reason = "host rejected a non-canonical tool call";
            return true;
        }
        const std::string selected_tool = parsed["name"].get<std::string>();
        json normalized_actual;
        std::string host_error;
        if (!host.normalize_call(selected_tool, parsed["arguments"], normalized_actual, host_error)) {
            result.known = true;
            result.verdict = common_flydelta_oracle_verdict::violated;
            result.confidence = 1.0f;
            result.reason = "host rejected tool arguments: " + host_error;
            return true;
        }
        common_tool_execution_result execution;
        if (!host.execute_normalized_call(selected_tool, normalized_actual, execution, host_error)) {
            result.known = true;
            result.verdict = common_flydelta_oracle_verdict::violated;
            result.confidence = 1.0f;
            result.reason = "host rejected tool execution: " + host_error;
            return true;
        }
        json normalized_expected;
        if (!host.normalize_call(canonical_expected_tool, expected_arguments,
                normalized_expected, host_error)) {
            evaluator_error = "host canonical tool contract is invalid: " + host_error;
            return false;
        }
        result.known = true;
        result.confidence = 1.0f;
        if (selected_tool != canonical_expected_tool || normalized_actual != normalized_expected) {
            result.verdict = common_flydelta_oracle_verdict::violated;
            result.reason = "host executed a different tool or canonical argument set";
            return true;
        }
        result.verdict = common_flydelta_oracle_verdict::satisfied;
        result.reason = "host executed the expected canonical tool call";
        return true;
    };
    common_flydelta_oracle_result oracle_result;
    if (!common_flydelta_oracle_evaluate(
            evaluators, request,
            common_agent_generation_succeeded(generation) ? generation.content : "",
            oracle_result, error)) return false;
    verdict.known = oracle_result.known;
    verdict.passed = oracle_result.known &&
        oracle_result.verdict == common_flydelta_oracle_verdict::satisfied;
    verdict.reason = oracle_result.reason;
    return true;
}

common_flydelta_counterfactual_outcome classify_host_verdicts(
        const host_tool_verdict & baseline,
        const host_tool_verdict & candidate) {
    if (!baseline.known || !candidate.known) {
        return common_flydelta_counterfactual_outcome::unknown;
    }
    if (!baseline.passed && candidate.passed) return common_flydelta_counterfactual_outcome::helped;
    if (baseline.passed && !candidate.passed) return common_flydelta_counterfactual_outcome::harmed;
    if (baseline.passed && candidate.passed) return common_flydelta_counterfactual_outcome::neutral;
    return common_flydelta_counterfactual_outcome::unknown;
}

std::string output_preview(const common_agent_generation_result & result) {
    if (!common_agent_generation_succeeded(result)) {
        return std::string("<generation-failed: ") + result.error_message + ">";
    }
    std::string preview = result.content;
    for (char & value : preview) {
        if (value == '\n' || value == '\r' || value == '\t') value = ' ';
    }
    constexpr size_t max_preview = 512;
    if (preview.size() > max_preview) preview.resize(max_preview);
    return preview;
}

} // namespace agent_flydelta_repair_smoke_support
