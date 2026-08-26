#include "agent/runtime-json-contracts.h"

#include <algorithm>
#include <cctype>
#include <regex>

using json = nlohmann::ordered_json;

namespace {

bool infer_calculator_expression(const std::string & text, std::string & expression) {
    static const std::regex arithmetic(R"((\(?\s*\d+(?:\.\d+)?(?:\s*[-+*/]\s*\d+(?:\.\d+)?)+\s*\)?))");
    std::smatch match;
    if (!std::regex_search(text, match, arithmetic) || match.size() < 2) {
        return false;
    }
    expression = match[1].str();
    return true;
}

bool infer_memory_search_query(const std::string & prompt, std::string & query) {
    const auto first = prompt.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return false;
    }
    const auto last = prompt.find_last_not_of(" \t\r\n");
    query = prompt.substr(first, last - first + 1);
    return query.size() <= 1024;
}

struct model_resource_candidate {
    std::string handle;
    const common_agent_input_resource * resource = nullptr;
};

std::vector<model_resource_candidate> model_resource_candidates(
        const common_agent_request & request,
        const std::string & name = {}) {
    std::vector<model_resource_candidate> candidates;
    const auto add = [&](const std::vector<common_agent_input_resource> & resources,
                         const char prefix) {
        for (size_t index = 0; index < resources.size(); ++index) {
            const auto & resource = resources[index];
            if (resource.resource.uri.empty() ||
                    (!name.empty() && resource.resource.name != name)) {
                continue;
            }
            candidates.push_back({std::string(1, prefix) + std::to_string(index + 1), &resource});
        }
    };
    add(request.input_resources, 'r');
    add(request.available_resources, 's');
    return candidates;
}

std::string render_resource_candidates(
        const std::vector<model_resource_candidate> & candidates) {
    std::string rendered;
    for (const auto & candidate : candidates) {
        rendered += " " + candidate.handle;
        if (!candidate.resource->resource.name.empty()) {
            rendered += " (" + candidate.resource->resource.name + ")";
        }
    }
    return rendered;
}

bool is_model_resource_handle(const std::string & value) {
    return value.size() >= 2 && (value.front() == 'r' || value.front() == 's') &&
        std::all_of(value.begin() + 1, value.end(), [](const char character) {
            return std::isdigit(static_cast<unsigned char>(character));
        });
}

bool resolve_model_resource_value(
        const common_agent_request & request,
        const std::string & value,
        std::string & uri,
        std::string & error) {
    if (is_model_resource_handle(value)) {
        const auto candidates = model_resource_candidates(request);
        const auto candidate = std::find_if(
            candidates.begin(), candidates.end(),
            [&](const model_resource_candidate & item) { return item.handle == value; });
        if (candidate == candidates.end()) {
            error = "unknown resource '" + value + "'. Choose one of:" +
                render_resource_candidates(candidates);
            if (candidates.empty()) error += " none";
            return false;
        }
        uri = candidate->resource->resource.uri;
        return true;
    }
    // Canonical resource and dataset URIs must remain untouched. Only a
    // human-facing name is eligible for exact lookup in host-listed resources.
    if (value.find("://") != std::string::npos) {
        uri = value;
        return true;
    }
    const auto candidates = model_resource_candidates(request, value);
    if (candidates.size() == 1) {
        uri = candidates.front().resource->resource.uri;
        return true;
    }
    if (candidates.size() > 1) {
        error = "ambiguous resource '" + value + "'. Choose one of:" +
            render_resource_candidates(candidates);
        return false;
    }
    error = "unknown resource '" + value + "'. Choose one of:" +
        render_resource_candidates(model_resource_candidates(request));
    if (model_resource_candidates(request).empty()) error += " none";
    return false;
}

bool resolve_model_resource_handle(
        const common_agent_request & request,
        const json & arguments,
        json & normalized_arguments,
        std::string & error) {
    const char * supplied_key = nullptr;
    for (const char * key : {"id", "resource", "resource_id"}) {
        if (arguments.contains(key)) {
            supplied_key = key;
            break;
        }
    }
    if (supplied_key == nullptr) return true;
    if (!arguments.at(supplied_key).is_string()) {
        error = "resource handle must be a string";
        return false;
    }
    const auto handle = arguments.at(supplied_key).get<std::string>();
    std::string uri;
    if (!resolve_model_resource_value(request, handle, uri, error)) return false;
    normalized_arguments.erase("id");
    normalized_arguments.erase("resource");
    normalized_arguments.erase("resource_id");
    normalized_arguments["uri"] = uri;
    return true;
}

} // namespace

nlohmann::ordered_json common_agent_runtime_user_correction_to_json(
        const std::string & source_turn_id,
        const std::string & statement) {
    return {
        {"source_turn_id", source_turn_id},
        {"statement", statement},
    };
}

std::string common_agent_runtime_user_correction_json(
        const std::string & source_turn_id,
        const std::string & statement) {
    return common_agent_runtime_user_correction_to_json(
        source_turn_id,
        statement).dump();
}

nlohmann::ordered_json common_agent_runtime_failure_observation_to_json(
        const common_agent_failure & failure) {
    return {
        {"failure", {
            {"code", failure.code},
            {"class", common_agent_failure_class_name(failure.classification)},
            {"stage", failure.stage},
            {"tool", failure.tool_name},
            {"step_id", failure.step_id},
            {"retryable", failure.retryable},
            {"safe_summary", failure.safe_summary},
            {"evidence_id", failure.evidence_id},
            {"repair_context", failure.repair_context_json.empty()
                ? json(nullptr)
                : json::parse(failure.repair_context_json, nullptr, false)},
        }},
    };
}

std::string common_agent_runtime_failure_observation_json(
        const common_agent_failure & failure) {
    return common_agent_runtime_failure_observation_to_json(failure).dump();
}

nlohmann::ordered_json common_agent_runtime_reflection_learning_hint_to_json(
        const common_reflection_learning_hint & hint) {
    return {
        {"category", hint.category},
        {"statement", hint.statement},
        {"expected_reuse", hint.expected_reuse},
    };
}

std::string common_agent_runtime_reflection_learning_hint_json(
        const common_reflection_learning_hint & hint) {
    return common_agent_runtime_reflection_learning_hint_to_json(hint).dump();
}

nlohmann::ordered_json common_agent_runtime_reasoning_observation_to_json(
        const std::string & reasoning_text) {
    const auto parsed = json::parse(reasoning_text, nullptr, false);
    if (parsed.is_object()) {
        return parsed;
    }
    return {
        {"summary", reasoning_text},
        {"format", "unstructured"},
    };
}

std::string common_agent_runtime_normalize_reasoning_observation_json(
        const std::string & reasoning_text) {
    return common_agent_runtime_reasoning_observation_to_json(reasoning_text).dump();
}

bool common_agent_runtime_apply_safe_tool_defaults_to_json(
        const common_agent_request & request,
        const std::string & tool_name,
        const json & arguments,
        json & normalized_arguments,
        bool & changed,
        std::string & error) {
    error.clear();
    changed = false;
    if (!arguments.is_object()) {
        error = "tool arguments must be a JSON object";
        return false;
    }

    normalized_arguments = arguments;
    const auto set_prompt_query = [&](size_t max_length) {
        if (normalized_arguments.contains("query")) {
            return;
        }
        std::string query;
        if (infer_memory_search_query(request.prompt, query) && query.size() <= max_length) {
            normalized_arguments["query"] = std::move(query);
            changed = true;
        }
    };

    if ((tool_name == "math.calculate" || tool_name == "calculator") && !normalized_arguments.contains("expression")) {
        std::string expression;
        if (infer_calculator_expression(request.prompt, expression)) {
            normalized_arguments["expression"] = std::move(expression);
            changed = true;
        }
    } else if (tool_name == "memory.search" || tool_name == "memory_search") {
        set_prompt_query(1024);
    } else if (tool_name == "repository.search" || tool_name == "workspace.search") {
        set_prompt_query(256);
        if (!normalized_arguments.contains("path")) { normalized_arguments["path"] = ""; changed = true; }
        if (!normalized_arguments.contains("max_results")) { normalized_arguments["max_results"] = 16; changed = true; }
    } else if (tool_name == "web.search" || tool_name == "web_search") {
        set_prompt_query(256);
        if (!normalized_arguments.contains("limit")) { normalized_arguments["limit"] = 5; changed = true; }
    } else if (tool_name == "repository.read" || tool_name == "workspace.read") {
        if (!normalized_arguments.contains("start_line")) { normalized_arguments["start_line"] = 1; changed = true; }
        if (!normalized_arguments.contains("end_line")) { normalized_arguments["end_line"] = 200; changed = true; }
    } else if (tool_name == "resource.read" || tool_name == "resource_read") {
        if (!resolve_model_resource_handle(request, arguments, normalized_arguments, error)) return false;
        if (normalized_arguments.contains("uri") && normalized_arguments["uri"].is_string() &&
                normalized_arguments["uri"].get<std::string>().find("://") == std::string::npos) {
            std::string uri;
            if (!resolve_model_resource_value(request, normalized_arguments["uri"].get<std::string>(), uri, error)) return false;
            normalized_arguments["uri"] = uri;
        }
        if (!normalized_arguments.contains("max_bytes")) { normalized_arguments["max_bytes"] = 8192; changed = true; }
        if (normalized_arguments.contains("uri")) changed = true;
    } else if (tool_name == "resource.inspect" || tool_name == "resource_inspect") {
        if (!resolve_model_resource_handle(request, arguments, normalized_arguments, error)) return false;
        if (normalized_arguments.contains("uri") && normalized_arguments["uri"].is_string() &&
                normalized_arguments["uri"].get<std::string>().find("://") == std::string::npos) {
            std::string uri;
            if (!resolve_model_resource_value(request, normalized_arguments["uri"].get<std::string>(), uri, error)) return false;
            normalized_arguments["uri"] = uri;
        }
        if (normalized_arguments.contains("uri")) changed = true;
    } else if (tool_name == "dataset.inspect" || tool_name == "dataset.schema" || tool_name == "dataset.sample") {
        // Compatibility repair for the old compact-schema example. The
        // placeholder is invalid, but a single current attachment is an
        // unambiguous safe default.
        if (normalized_arguments.contains("dataset") &&
                normalized_arguments["dataset"].is_string() &&
                normalized_arguments["dataset"].get<std::string>() == "$datasets.datasets[]" &&
                request.input_resources.size() == 1 &&
                !request.input_resources.front().resource.uri.empty()) {
            normalized_arguments.erase("dataset");
            normalized_arguments["resource"] = request.input_resources.front().resource.uri;
            changed = true;
        }
        // The model-facing contract calls the attachment a resource, but a
        // small model can copy the dataset_ref type annotation and emit
        // dataset:"r1" instead. Treat the handle as a resource selection;
        // otherwise the literal r1 reaches the dataset registry and becomes
        // the misleading "dataset reference unavailable" failure seen by
        // the web client.
        if (normalized_arguments.contains("dataset") &&
                normalized_arguments["dataset"].is_string()) {
            const auto dataset_value = normalized_arguments["dataset"].get<std::string>();
            if (dataset_value.size() >= 2 &&
                    (dataset_value.front() == 'r' || dataset_value.front() == 's') &&
                    std::all_of(dataset_value.begin() + 1, dataset_value.end(), [](const char character) {
                        return std::isdigit(static_cast<unsigned char>(character));
                    })) {
                json resource_arguments = json::object({{"resource", dataset_value}});
                json resolved_resource = resource_arguments;
                if (!resolve_model_resource_handle(request, resource_arguments, resolved_resource, error)) {
                    return false;
                }
                normalized_arguments.erase("dataset");
                normalized_arguments["resource"] = resolved_resource["uri"];
                changed = true;
            }
        }
        // A filename is a convenience lookup key, not a dataset identity.
        // Resolve it against both current attachments and host-listed scoped
        // resources when exactly one resource has that name. Ambiguity is a
        // repairable host error; an unmatched value remains available for a
        // registered dataset lookup below.
        if (normalized_arguments.contains("dataset") &&
                normalized_arguments["dataset"].is_string()) {
            const auto dataset_value = normalized_arguments["dataset"].get<std::string>();
            if (!is_model_resource_handle(dataset_value) && dataset_value.find("://") == std::string::npos) {
                std::string uri;
                std::string resource_error;
                if (resolve_model_resource_value(request, dataset_value, uri, resource_error)) {
                    normalized_arguments.erase("dataset");
                    normalized_arguments["resource"] = uri;
                    changed = true;
                } else if (resource_error.rfind("ambiguous resource ", 0) == 0) {
                    error = std::move(resource_error);
                    return false;
                }
            }
        }
        // If there is one caller-owned attachment, a non-canonical dataset
        // value is an ambiguous model alias rather than a trustworthy
        // registry reference. Apply the same single-attachment default as an
        // omitted argument, while preserving explicit dataset:// references.
        if (normalized_arguments.contains("dataset") &&
                normalized_arguments["dataset"].is_string() &&
                request.input_resources.size() == 1 &&
                !request.input_resources.front().resource.uri.empty()) {
            const auto dataset_value = normalized_arguments["dataset"].get<std::string>();
            if (dataset_value.rfind("dataset://", 0) != 0) {
                normalized_arguments.erase("dataset");
                normalized_arguments["resource"] = request.input_resources.front().resource.uri;
                changed = true;
            }
        }
        const bool resource_handle = arguments.contains("id") ||
            (arguments.contains("resource") && arguments["resource"].is_string() &&
                (arguments["resource"].get<std::string>().rfind("r", 0) == 0 ||
                 arguments["resource"].get<std::string>().rfind("s", 0) == 0));
        if (resource_handle) {
            if (!resolve_model_resource_handle(request, arguments, normalized_arguments, error)) return false;
            if (normalized_arguments.contains("uri")) {
                normalized_arguments["resource"] = normalized_arguments["uri"];
                normalized_arguments.erase("uri");
                changed = true;
            }
        }
        // A single current-turn attachment is an unambiguous safe source for
        // dataset inspection. Keep the source as a resource reference rather
        // than making the model invent a dataset-list alias. With multiple
        // attachments the model must select one explicitly using resource:rN.
        if (!normalized_arguments.contains("dataset") &&
                !normalized_arguments.contains("resource") &&
                !normalized_arguments.contains("path") &&
                request.input_resources.size() == 1 &&
                !request.input_resources.front().resource.uri.empty()) {
            normalized_arguments["resource"] = request.input_resources.front().resource.uri;
            changed = true;
        }
    } else if (tool_name == "repository.list" || tool_name == "workspace.list") {
        if (!normalized_arguments.contains("path")) { normalized_arguments["path"] = ""; changed = true; }
        if (!normalized_arguments.contains("depth")) { normalized_arguments["depth"] = 1; changed = true; }
    } else if (tool_name == "document.tables" || tool_name == "document.table") {
        // Document tools use the resource_ref-shaped `resource` property in
        // their strict schema, but model-facing calls may contain the same
        // current-turn handle used by resource_read, for example `r1`.
        // Resolve that handle before the document binding attempts a store
        // lookup; otherwise the literal handle is incorrectly treated as a
        // canonical URI and the representation appears to be unavailable.
        const auto is_resource_handle = [](const json & value) {
            if (!value.is_string()) return false;
            const auto handle = value.get<std::string>();
            return handle.size() >= 2 && handle.front() == 'r' &&
                std::all_of(handle.begin() + 1, handle.end(),
                    [](const char character) { return std::isdigit(static_cast<unsigned char>(character)); });
        };
        const bool resource_handle = arguments.contains("id") ||
            (arguments.contains("resource") && is_resource_handle(arguments["resource"])) ||
            arguments.contains("resource_id");
        if (resource_handle) {
            if (!resolve_model_resource_handle(request, arguments, normalized_arguments, error)) return false;
            if (normalized_arguments.contains("uri")) {
                normalized_arguments["resource"] = normalized_arguments["uri"];
                normalized_arguments.erase("uri");
                changed = true;
            }
        }
        if (normalized_arguments.contains("resource") &&
                normalized_arguments["resource"].is_string()) {
            const auto supplied_resource = normalized_arguments["resource"].get<std::string>();
            if (!is_model_resource_handle(supplied_resource) && supplied_resource.find("://") == std::string::npos) {
                std::string uri;
                if (!resolve_model_resource_value(request, supplied_resource, uri, error)) return false;
                normalized_arguments["resource"] = uri;
                changed = true;
            }
        }
        if (normalized_arguments.contains("resource") &&
                normalized_arguments["resource"].is_string()) {
            const std::string supplied_resource = normalized_arguments["resource"].get<std::string>();
            const common_agent_input_resource * matching_resource = nullptr;
            bool ambiguous_resource_name = false;
            for (const auto & input : request.input_resources) {
                if (supplied_resource == input.resource.uri) {
                    matching_resource = nullptr;
                    ambiguous_resource_name = false;
                    break;
                }
                // Resource descriptors are rendered as bounded human-readable
                // lines in the prompt. Small models may copy that complete
                // line instead of returning only the authoritative URI. Accept
                // that exact URI-prefixed projection, but resolve only against
                // caller-owned input resources.
                if (!input.resource.uri.empty() &&
                        supplied_resource.size() > input.resource.uri.size() &&
                        supplied_resource.compare(0, input.resource.uri.size(), input.resource.uri) == 0 &&
                        supplied_resource[input.resource.uri.size()] == ' ') {
                    matching_resource = &input;
                    continue;
                }
                if (!input.resource.name.empty() && supplied_resource == input.resource.name) {
                    if (matching_resource != nullptr) {
                        ambiguous_resource_name = true;
                        break;
                    }
                    matching_resource = &input;
                }
            }
            if (!ambiguous_resource_name && matching_resource != nullptr &&
                    !matching_resource->resource.uri.empty()) {
                normalized_arguments["resource"] = matching_resource->resource.uri;
                changed = true;
            }
        }
        // A single caller-owned input resource is an unambiguous safe default.
        // With multiple resources the model must select one explicitly.
        if (!normalized_arguments.contains("resource") && request.input_resources.size() == 1 &&
                !request.input_resources.front().resource.uri.empty()) {
            normalized_arguments["resource"] = request.input_resources.front().resource.uri;
            changed = true;
        }
    }

    return true;
}

bool common_agent_runtime_apply_safe_tool_defaults(
        const common_agent_request & request,
        common_agent_tool_call & call) {
    const auto arguments = json::parse(call.arguments_json, nullptr, false);
    json normalized_arguments;
    bool changed = false;
    std::string error;
    if (!common_agent_runtime_apply_safe_tool_defaults_to_json(
            request,
            call.name,
            arguments,
            normalized_arguments,
            changed,
            error)) {
        return false;
    }

    if (changed) {
        call.arguments_json = normalized_arguments.dump();
    }
    return changed;
}
