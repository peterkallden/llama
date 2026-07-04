#include "memory/memory-tool-service.h"

#include "memory/memory-context.h"
#include "memory/memory-retrieval.h"

#include <algorithm>
#include <ctime>

#include <nlohmann/json.hpp>

using json = nlohmann::ordered_json;

namespace {

bool parse_object_arguments(
        const std::string & arguments_json,
        json & arguments,
        std::string & error) {
    arguments = json::parse(arguments_json, nullptr, false);
    if (!arguments.is_object()) {
        error = "arguments must be a JSON object";
        return false;
    }
    error.clear();
    return true;
}

} // namespace

common_memory_tool_service::common_memory_tool_service(common_memory_store & store)
    : store(store) {}

bool common_memory_tool_service::search(
        const common_memory_tool_context & context,
        const std::string & arguments_json,
        common_memory_tool_search_result & result,
        std::string & error) const {
    result = {};

    json arguments;
    if (!parse_object_arguments(arguments_json, arguments, error)) {
        return false;
    }
    for (const auto & item : arguments.items()) {
        if (item.key() != "query" && item.key() != "limit") {
            error = "unsupported argument: " + item.key();
            return false;
        }
    }
    if (!arguments.contains("query") || !arguments.at("query").is_string()) {
        error = "query must be a string";
        return false;
    }

    result.query = arguments.at("query").get<std::string>();
    if (result.query.empty() || result.query.size() > context.max_query_chars) {
        error = "query must contain between 1 and 1024 characters";
        return false;
    }

    common_memory_query query = context.query_defaults;
    query.text = result.query;

    size_t limit = std::clamp(query.limit, size_t(1), context.max_search_limit);
    if (arguments.contains("limit")) {
        if (!arguments.at("limit").is_number_unsigned() && !arguments.at("limit").is_number_integer()) {
            error = "limit must be an integer";
            return false;
        }
        const int requested_limit = arguments.at("limit").get<int>();
        if (requested_limit < 1 || requested_limit > (int) context.max_search_limit) {
            error = "limit must be between 1 and 8";
            return false;
        }
        limit = (size_t) requested_limit;
    }
    query.limit = limit;

    if (context.embed) {
        query.embedding.clear();
        if (!context.embed(query.text, query.embedding, error)) {
            return false;
        }
    }

    common_memory_retrieval retrieval(store);
    result.hits = retrieval.retrieve(query, error);
    if (!error.empty()) {
        return false;
    }

    common_memory_context_config context_config;
    context_config.char_budget = query.token_budget * 4;
    result.context = common_memory_render_context(result.hits, context_config);
    error.clear();
    return true;
}

bool common_memory_tool_service::remember_proposal(
        const common_memory_tool_context & context,
        const std::string & arguments_json,
        common_memory_tool_remember_result & result,
        std::string & error) const {
    result = {};

    if (!context.allow_write_proposals) {
        error = "memory write proposals are disabled in this runtime context";
        return false;
    }

    json arguments;
    if (!parse_object_arguments(arguments_json, arguments, error)) {
        return false;
    }
    for (const auto & item : arguments.items()) {
        if (item.key() != "kind" && item.key() != "content" && item.key() != "importance" &&
                item.key() != "confidence" && item.key() != "rationale") {
            error = "unsupported argument: " + item.key();
            return false;
        }
    }
    if (!arguments.contains("kind") || !arguments.at("kind").is_string()) {
        error = "kind must be a string";
        return false;
    }
    if (!arguments.contains("content") || !arguments.at("content").is_string()) {
        error = "content must be a string";
        return false;
    }

    common_memory_remember_request proposal;
    if (!common_memory_kind_parse(arguments.at("kind").get<std::string>(), proposal.kind)) {
        error = "unsupported memory kind";
        return false;
    }
    proposal.content = arguments.at("content").get<std::string>();
    proposal.scope = context.query_defaults.scope;
    proposal.namespace_id = context.query_defaults.namespace_id;
    proposal.session_id = context.query_defaults.session_id;
    proposal.project_id = context.query_defaults.project_id;
    proposal.turn_id = context.query_defaults.turn_id;
    proposal.global_opt_in = context.query_defaults.global_opt_in;
    if (proposal.content.empty() || proposal.content.size() > context.max_content_chars) {
        error = "content must contain between 1 and 512 characters";
        return false;
    }

    if (arguments.contains("importance")) {
        if (!arguments.at("importance").is_number()) {
            error = "importance must be a number";
            return false;
        }
        proposal.importance = arguments.at("importance").get<float>();
    }
    if (arguments.contains("confidence")) {
        if (!arguments.at("confidence").is_number()) {
            error = "confidence must be a number";
            return false;
        }
        proposal.confidence = arguments.at("confidence").get<float>();
    }
    if (arguments.contains("rationale")) {
        if (!arguments.at("rationale").is_string()) {
            error = "rationale must be a string";
            return false;
        }
        proposal.rationale = arguments.at("rationale").get<std::string>();
        if (proposal.rationale.size() > context.max_rationale_chars) {
            error = "rationale must contain at most 240 characters";
            return false;
        }
    }

    std::vector<float> embedding;
    if (context.embed && !context.embed(proposal.content, embedding, error)) {
        return false;
    }

    result.proposal = proposal;
    const int64_t now = context.now != 0 ? context.now : std::time(nullptr);
    result.decision = common_memory_evaluate_remember_request(
        store,
        proposal,
        embedding,
        now,
        error);
    return error.empty();
}
