#pragma once

#include "../tooling/agent-tool-provider.h"

#include <nlohmann/json.hpp>

#include <string>
#include <vector>

nlohmann::ordered_json render_agent_tool_resource_ref_json(
    const common_runtime_resource_ref & resource);

void attach_agent_tool_resource_refs_json(
    const std::vector<common_runtime_resource_ref> & resources,
    nlohmann::ordered_json & payload);

nlohmann::ordered_json make_agent_tool_failure_payload_json(
    const std::string & code,
    const std::string & message,
    bool retryable,
    common_tool_failure_class failure_class,
    const std::vector<common_runtime_resource_ref> & resources);

nlohmann::ordered_json make_agent_tool_success_payload_json(
    const std::string & output_json_or_text,
    const std::string & summary,
    const std::vector<common_runtime_resource_ref> & resources);

nlohmann::ordered_json make_agent_tool_structured_success_payload_json(
    const std::string & structured_content_json,
    const std::string & summary,
    const std::vector<common_runtime_resource_ref> & resources);

nlohmann::ordered_json make_agent_tool_text_success_payload_json(
    const std::string & text,
    const std::string & summary,
    const std::vector<common_runtime_resource_ref> & resources);

agent_tool_result make_agent_tool_failure_result(
    const agent_tool_call & call,
    const std::string & failure_code,
    common_tool_failure_class failure_class,
    bool retryable,
    const std::string & safe_summary,
    const std::string & raw_diagnostic = {},
    std::vector<common_runtime_resource_ref> resources = {});

agent_tool_result make_agent_tool_json_success_result(
    const agent_tool_call & call,
    const std::string & output_json_or_text,
    const std::string & summary,
    std::vector<common_runtime_resource_ref> resources = {});

agent_tool_result make_agent_tool_structured_success_result(
    const agent_tool_call & call,
    const std::string & structured_content_json,
    const std::string & summary,
    std::vector<common_runtime_resource_ref> resources = {});

agent_tool_result make_agent_tool_text_success_result(
    const agent_tool_call & call,
    const std::string & text,
    const std::string & summary,
    std::vector<common_runtime_resource_ref> resources = {});
