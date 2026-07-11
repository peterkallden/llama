#pragma once

#include "memory/memory-policy.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

struct common_memory_tool_context {
    common_memory_query query_defaults;
    size_t max_search_limit = 8;
    size_t max_query_chars = 1024;
    size_t max_content_chars = 512;
    size_t max_rationale_chars = 240;
    bool allow_write_proposals = false;
    int64_t now = 0;
    std::function<bool(const std::string & text, std::vector<float> & embedding, std::string & error)> embed;
};

struct common_memory_tool_search_result {
    std::string query;
    std::vector<common_memory_hit> hits;
    std::string context;
};

struct common_memory_tool_search_arguments_contract {
    std::string query;
    std::optional<size_t> limit;
};

struct common_memory_tool_remember_arguments_contract {
    std::string kind;
    std::string content;
    std::optional<float> importance;
    std::optional<float> confidence;
    std::optional<std::string> rationale;
};

struct common_memory_tool_remember_result {
    common_memory_remember_request proposal;
    common_memory_remember_result decision;
};

bool common_memory_parse_tool_search_arguments_json(
    const std::string & arguments_json,
    common_memory_tool_search_arguments_contract & contract,
    std::string & error);
bool common_memory_parse_tool_search_arguments_value(
    const nlohmann::ordered_json & arguments,
    common_memory_tool_search_arguments_contract & contract,
    std::string & error);

bool common_memory_parse_tool_remember_arguments_json(
    const std::string & arguments_json,
    common_memory_tool_remember_arguments_contract & contract,
    std::string & error);
bool common_memory_parse_tool_remember_arguments_value(
    const nlohmann::ordered_json & arguments,
    common_memory_tool_remember_arguments_contract & contract,
    std::string & error);

class common_memory_tool_service {
public:
    explicit common_memory_tool_service(common_memory_store & store);

    bool search(
        const common_memory_tool_context & context,
        const std::string & arguments_json,
        common_memory_tool_search_result & result,
        std::string & error) const;

    bool remember_proposal(
        const common_memory_tool_context & context,
        const std::string & arguments_json,
        common_memory_tool_remember_result & result,
        std::string & error) const;

private:
    common_memory_store & store;
};
