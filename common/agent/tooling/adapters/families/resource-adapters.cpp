#include "agent/tooling/adapters/families/resource-adapters.h"

#include "agent/tooling/adapters/support/adapter-support.h"
#include "agent/tooling/contracts/tool-result-contracts.h"
#include "base64.hpp"

#include <ctime>
#include <memory>

namespace {

using json = common_adapter_json;

agent_resource_read_authority authority(const common_native_tool_bindings & bindings) {
    return make_agent_resource_read_authority(bindings.resource_runtime, std::time(nullptr));
}

bool has_text(const agent_resource_descriptor & descriptor) {
    return common_resource_media_type_is_text_like(descriptor.mime_type);
}

std::vector<std::string> representations(const agent_resource_descriptor & descriptor) {
    std::vector<std::string> result{"bytes"};
    if (has_text(descriptor)) result.push_back("text");
    return result;
}

} // namespace

bool common_try_register_resource_tool_adapter(
        const common_tool_definition & definition,
        const common_native_tool_bindings & bindings,
        common_tool_registry & registry,
        bool & installed,
        std::string & error) {
    installed = false;
    const auto id = definition.executor_id;
    if (id != "builtin.resource_inspect" && id != "builtin.resource_read") return false;
    if (bindings.resource_runtime.store == nullptr) return true;

    if (id == "builtin.resource_inspect") {
        installed = common_adapter_register_definition(definition, registry, [bindings](const std::string & input) {
            std::string error; json args;
            if (!common_adapter_parse_object(input, args, error) || !args.contains("uri") || !args["uri"].is_string()) return common_adapter_validation_failure("tool.resource_inspect.invalid_uri", error.empty() ? "resource_inspect requires a uri" : std::move(error), "Resource inspection requires a valid resource URI.");
            const auto uri = args["uri"].get<std::string>();
            if (uri.empty() || uri.size() > 512) return common_adapter_validation_failure("tool.resource_inspect.invalid_uri", "resource_inspect uri is out of bounds", "Resource inspection requires a bounded resource URI.");
            agent_resource_descriptor descriptor;
            if (!bindings.resource_runtime.store->stat(uri, authority(bindings), descriptor, error)) return common_adapter_not_found_failure("tool.resource_inspect.unavailable", std::move(error), "Resource is unavailable in the current runtime scope.");
            return common_adapter_success_json(common_tool_resource_inspect_result_to_json({descriptor, representations(descriptor)}));
        }, error);
        return installed;
    }

    installed = common_adapter_register_definition(definition, registry, [bindings](const std::string & input) {
        std::string error; json args;
        if (!common_adapter_parse_object(input, args, error) || !args.contains("uri") || !args["uri"].is_string()) return common_adapter_validation_failure("tool.resource_read.invalid_uri", error.empty() ? "resource_read requires a uri" : std::move(error), "Resource read requires a valid resource URI.");
        const auto uri = args["uri"].get<std::string>();
        const auto representation = args.value("representation", std::string("text"));
        const int64_t offset = args.value("offset", int64_t{0});
        const int max_bytes = args.value("max_bytes", 8192);
        if (uri.empty() || uri.size() > 512 || representation.empty() || representation.size() > 64 || offset < 0 || offset > 1073741824LL || max_bytes < 1 || max_bytes > 32768) return common_adapter_validation_failure("tool.resource_read.out_of_bounds", "resource_read arguments are out of bounds", "Resource read arguments are out of bounds.");
        if (representation != "text" && representation != "bytes") return common_adapter_not_found_failure("tool.resource_read.representation_unavailable", "resource representation is not available in the current runtime", "The requested resource representation is not available.");
        const auto read_authority = authority(bindings);
        agent_resource_descriptor descriptor;
        if (!bindings.resource_runtime.store->stat(uri, read_authority, descriptor, error)) return common_adapter_not_found_failure("tool.resource_read.unavailable", std::move(error), "Resource is unavailable in the current runtime scope.");

        if (representation == "text" && !has_text(descriptor) && (bindings.resource_processing_service != nullptr || bindings.resource_processing_provider_factory)) {
            agent_resource_processing_binding_request request;
            request.source_uri = descriptor.uri;
            request.operation_id = "resource-read/" + (bindings.resource_runtime.turn_id.empty() ? std::string("turn") : bindings.resource_runtime.turn_id) + "/" + std::to_string(std::hash<std::string>{}(uri));
            request.authority = read_authority;
            request.media_type.declared_type = descriptor.mime_type;
            request.target_representation = "text";
            std::shared_ptr<agent_resource_processing_provider> owned;
            auto * provider = bindings.resource_processing_service;
            if (bindings.resource_processing_provider_factory) { owned = bindings.resource_processing_provider_factory(request); provider = owned.get(); }
            if (provider == nullptr) return common_adapter_execution_failure("tool.resource_read.processing_failed", "resource processor provider could not be created", "The resource representation could not be materialized.");
            const auto processed = provider->process(request);
            if (!processed.success || processed.resources.empty()) return common_adapter_not_found_failure("tool.resource_read.representation_unavailable", processed.safe_summary, "The requested resource representation is not available.");
            if (!bindings.resource_runtime.store->stat(processed.resources.front().uri, read_authority, descriptor, error) || !has_text(descriptor)) return common_adapter_execution_failure("tool.resource_read.processing_failed", "processor did not return a readable text resource", "The resource representation could not be materialized.");
        } else if (representation == "text" && !has_text(descriptor)) {
            return common_adapter_not_found_failure("tool.resource_read.representation_unavailable", "text representation is not available for resource media type " + descriptor.mime_type, "The requested resource representation is not available.");
        }

        std::string content, encoding;
        if (representation == "bytes") {
            if (!bindings.resource_runtime.store->read_bytes_range(uri, read_authority, static_cast<size_t>(offset), static_cast<size_t>(max_bytes), content, error)) return common_adapter_execution_failure("tool.resource_read.read_failed", std::move(error), "Resource bytes could not be read.");
            content = base64::encode(content); encoding = "base64";
        } else if (!bindings.resource_runtime.store->read_text_range(uri, read_authority, static_cast<size_t>(offset), static_cast<size_t>(max_bytes), content, error)) {
            return common_adapter_execution_failure("tool.resource_read.read_failed", std::move(error), "Resource content could not be read.");
        }
        return common_adapter_success_json(common_tool_resource_read_result_to_json({descriptor, representation, content, encoding}));
    }, error);
    return installed;
}
