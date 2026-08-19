#pragma once

#include "agent/tooling/contracts/tool-runtime-contract.h"
#include "memory/memory-tool-service.h"
#include "plan/plan-types.h"
#include "resource/resource-contract.h"

#include <nlohmann/json.hpp>

#include <optional>
#include <string>
#include <vector>

struct common_tool_resource_read_result {
    agent_resource_descriptor resource;
    std::string representation = "text";
    std::string content;
    std::string content_encoding;
};

struct common_tool_resource_inspect_result {
    agent_resource_descriptor resource;
    std::vector<std::string> available_representations = {"text"};
};

struct common_tool_web_search_result {
    nlohmann::ordered_json results;
    std::string provider;
    std::optional<bool> truncated;
    std::optional<size_t> total_results;
};

struct common_tool_web_fetch_result {
    std::string url;
    std::string final_url;
    int status = 0;
    std::string content_type;
    std::string title;
    std::string text;
    bool truncated = false;
};

struct common_tool_web_fetch_inline_result {
    std::string url;
    std::string final_url;
    int status = 0;
    std::string content_type;
    std::string title;
    std::string text_excerpt;
    size_t text_length = 0;
    bool truncated = false;
};

struct common_tool_memory_search_payload {
    std::vector<common_memory_hit> hits;
};

struct common_tool_memory_remember_related_entry {
    std::string id;
    common_memory_kind kind = common_memory_kind::episode;
    float score = 0.0f;
    std::string content;
};

struct common_tool_memory_remember_payload {
    bool ok = true;
    std::string decision;
    std::string reason;
    common_memory_kind kind = common_memory_kind::episode;
    common_memory_scope scope = common_memory_scope::session;
    std::string content;
    size_t related_count = 0;
    std::vector<common_tool_memory_remember_related_entry> related;
    std::optional<std::string> id;
    std::optional<std::string> error;
};

struct common_tool_plan_get_step_payload {
    std::string id;
    std::string title;
    std::string objective;
    int status = 0;
    std::optional<std::string> selected_tool;
};

struct common_tool_plan_get_payload {
    std::string plan_id;
    uint64_t version = 0;
    std::string goal;
    std::optional<std::string> active_step;
    std::optional<std::string> next_action;
    std::vector<common_tool_plan_get_step_payload> steps;
    std::optional<size_t> history_count;
};

nlohmann::ordered_json common_tool_resource_descriptor_to_json(
    const agent_resource_descriptor & descriptor);

nlohmann::ordered_json common_tool_resource_read_result_to_json(
        const common_tool_resource_read_result & result);

nlohmann::ordered_json common_tool_resource_inspect_result_to_json(
        const common_tool_resource_inspect_result & result);

nlohmann::ordered_json common_tool_web_search_result_to_json(
    const common_tool_web_search_result & result);

nlohmann::ordered_json common_tool_web_fetch_result_to_json(
    const common_tool_web_fetch_result & result);

nlohmann::ordered_json common_tool_web_fetch_inline_result_to_json(
    const common_tool_web_fetch_inline_result & result);

nlohmann::ordered_json common_tool_memory_record_to_json(
    const common_memory_record & memory);

nlohmann::ordered_json common_tool_memory_search_result_to_json(
    const common_tool_memory_search_payload & result);

nlohmann::ordered_json common_tool_memory_get_result_to_json(
    const common_memory_record & memory);

nlohmann::ordered_json common_tool_memory_remember_result_to_json(
    const common_tool_memory_remember_payload & result);

nlohmann::ordered_json common_tool_plan_get_result_to_json(
    const common_tool_plan_get_payload & result);

nlohmann::ordered_json common_tool_chat_failure_payload_to_json(
    const std::string & code,
    const std::string & message,
    bool retryable,
    common_tool_failure_class failure_class);

nlohmann::ordered_json common_tool_chat_success_payload_to_json(
    const std::string & output_json_or_text);
