#pragma once

#include <jni.h>

#include <memory>

#include "tools/agent/mcp/agent-mcp-http-transport.h"

std::shared_ptr<agent_mcp_http_transport> make_agent_mcp_android_https_transport(
        JavaVM * java_vm,
        jclass transport_class,
        jmethodID post_method);
