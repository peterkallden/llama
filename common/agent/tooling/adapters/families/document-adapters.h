#pragma once

#include "agent/tooling/adapters/adapter-bindings.h"
#include "agent/tooling/catalog/tool-catalog.h"
#include "agent/tooling/registry/tool-registry.h"

bool common_adapter_document_current_revision(
        const common_native_tool_bindings & bindings,
        const std::string & id,
        std::string & document_uri,
        uint64_t & revision,
        std::string & error);

bool common_try_register_document_tool_adapter(
        const common_tool_definition & definition,
        const common_native_tool_bindings & bindings,
        common_tool_registry & registry,
        bool & installed,
        std::string & error);
