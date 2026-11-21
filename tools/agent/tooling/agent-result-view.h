#pragma once
#include <string>

// Validate a host-owned result view and render it against an unchanged JSON source.
bool agent_result_view_validate(const std::string & policy, std::string & error);
bool agent_result_view_project(const std::string & policy_json, const std::string & source_json,
    std::string & output_json, std::string & error);
