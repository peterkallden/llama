#pragma once

#include <string>

// Adds bounded, deterministic convenience fields to JSON results shown to the
// model. The fetched/provider body remains unchanged in its host resource.
std::string agent_openapi_project_model_result(
    const std::string & raw_json,
    bool openalex_result = false);
