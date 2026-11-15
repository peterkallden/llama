#pragma once

#include "agent/agent-inference.h"
#include "agent/runtime/model-catalog.h"

#include <memory>
#include <functional>

// Optional host transport for platforms which own TLS and credential storage.
// The callback returns an HTTP status and response body, or a transport error.
using common_agent_http_post = std::function<bool(const std::string &, const std::string &,
    uint32_t, uint32_t, int &, std::string &, std::string &)>;

std::unique_ptr<common_agent_inference> make_agent_http_inference(
    const common_agent_model_selection & selection,
    common_agent_http_post post = {});
