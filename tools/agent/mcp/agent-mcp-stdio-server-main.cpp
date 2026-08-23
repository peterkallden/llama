#include "agent-mcp-stdio-server.h"
#include "agent-mcp-http-server.h"

#include "tools/agent/cli/agent-cli-memory-tools.h"

#include "../cli/agent-cli-host-adapter.h"
#include "../cli/agent-cli-selection.h"
#include "../host/agent-host-config.h"
#include "../resource/agent-resource-store.h"
#include "../tooling/agent-tool-provider.h"
#include "tools/agent/cli/agent-cli-options.h"
#include "tools/agent/cli/agent-cli-scope.h"
#include "memory/memory-in-memory.h"
#ifdef LLAMA_MEMORY_USE_COZO
#include "memory/cozo/memory-cozo.h"
#endif
#include "plan/plan-in-memory.h"
#ifdef LLAMA_PLAN_USE_COZO
#include "plan/cozo/plan-cozo.h"
#endif
#ifdef LLAMA_MEMORY_USE_SQLITE
#include "memory/sqlite/memory-sqlite.h"
#endif
#ifdef LLAMA_PLAN_USE_SQLITE
#include "plan/sqlite/plan-sqlite.h"
#endif

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <map>
#include <memory>
#include <string>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

namespace {

bool configured_profile_allows_policy_gated_writes(const args & options) {
    common_tool_profile_snapshot snapshot;
    std::string error;
    if (!resolve_common_tool_profile_snapshot(
            options.tool_profile, options.tool_capabilities, options.tool_profiles,
            snapshot, error)) {
        return false;
    }
    return snapshot.allow_policy_gated_writes.value_or(false);
}

struct server_http_options {
    bool enabled = false;
    std::string listen_address = "127.0.0.1";
    int port = 0;
    std::string path = "/mcp";
    std::string allowed_origin;
    std::string token_env;
    std::string bearer_token;
    agent_mcp_caller_policy default_policy;
    size_t max_body_bytes = 1024 * 1024;
    size_t max_result_bytes = 1024 * 1024;
};

std::string resolve_memory_backend(
        const std::string & backend,
        const std::string & memory_db,
        std::string & error) {
    std::string resolved = backend;
    if (resolved == "auto") {
        resolved = memory_db.empty() ? "in-memory" : "cozo";
    }
    if (resolved == "in-memory" && !memory_db.empty()) {
        error = "--memory-db requires --backend cozo or the default auto backend";
        return {};
    }
    error.clear();
    return resolved;
}

std::string resolve_plan_backend(
        const std::string & backend,
        const std::string & plan_db,
        std::string & error) {
    std::string resolved = backend;
    if (resolved == "auto") {
        resolved = plan_db.empty() ? "in-memory" : "cozo";
    }
    if (resolved == "in-memory" && !plan_db.empty()) {
        error = "--plan-db requires --plan-backend cozo or the default auto backend";
        return {};
    }
    error.clear();
    return resolved;
}

const char * plan_scope_name(common_plan_scope scope) {
    switch (scope) {
        case common_plan_scope::turn: return "turn";
        case common_plan_scope::session: return "session";
        case common_plan_scope::project: return "project";
        case common_plan_scope::global: return "global";
    }
    return "turn";
}

const char * find_server_config_path(int argc, char ** argv) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--config") == 0) {
            return i + 1 < argc ? argv[i + 1] : nullptr;
        }
    }
    return nullptr;
}

class server_agent_embedding_provider final : public agent_embedding_provider {
public:
    server_agent_embedding_provider(std::string model_path, int n_gpu_layers)
        : model_path_(std::move(model_path)),
          n_gpu_layers_(n_gpu_layers) {}

    bool embed(
            const std::string & purpose,
            const std::string & text,
            std::vector<float> & embedding,
            std::string & error) override {
        return ensure_memory_cli_embedding_from_model(
            model_path_,
            n_gpu_layers_,
            text,
            embedding,
            purpose.c_str(),
            error);
    }

private:
    std::string model_path_;
    int n_gpu_layers_ = 0;
};

std::string summarize_tool_result(
        const agent_tool_result & result,
        const agent_mcp_json & structured_content) {
    if (result.ok && !result.content_summary.empty()) {
        return result.content_summary;
    }
    if (!result.ok && !result.safe_summary.empty()) {
        return result.safe_summary;
    }
    if (structured_content.is_object()) {
        const std::string summary = structured_content.value("summary", std::string());
        if (!summary.empty()) {
            return summary;
        }
        const std::string result_text = structured_content.value("result_text", std::string());
        if (!result_text.empty()) {
            return result_text;
        }
    }
    return {};
}

void append_resource_links(
        const std::vector<common_runtime_resource_ref> & resources,
        agent_mcp_server_tool_result & result) {
    if (!result.content.is_array()) {
        result.content = agent_mcp_json::array();
    }

    for (const auto & resource : resources) {
        agent_mcp_json entry = {
            {"type", "resource_link"},
            {"uri", resource.uri},
        };
        if (!resource.name.empty()) {
            entry["name"] = resource.name;
        }
        if (!resource.description.empty()) {
            entry["description"] = resource.description;
        }
        if (!resource.mime_type.empty()) {
            entry["mimeType"] = resource.mime_type;
        }
        if (resource.size_bytes > 0) {
            entry["sizeBytes"] = resource.size_bytes;
        }
        if (!resource.lineage.parent_uri.empty()) {
            entry["lineage"] = {
                {"parent_uri", resource.lineage.parent_uri},
                {"chunk_index", resource.lineage.chunk_index},
                {"chunk_count", resource.lineage.chunk_count},
                {"byte_offset", resource.lineage.byte_offset},
                {"byte_length", resource.lineage.byte_length},
                {"overlap_bytes", resource.lineage.overlap_bytes},
                {"derivation", resource.lineage.derivation},
            };
        }

        agent_mcp_json metadata = agent_mcp_json::object();
        if (!resource.metadata.purpose.empty()) {
            metadata["purpose"] = resource.metadata.purpose;
        }
        if (!resource.metadata.content_summary.empty()) {
            metadata["content_summary"] = resource.metadata.content_summary;
        }
        if (!resource.metadata.usage_hint.empty()) {
            metadata["usage_hint"] = resource.metadata.usage_hint;
        }
        if (!resource.metadata.limitations.empty()) {
            metadata["limitations"] = resource.metadata.limitations;
        }
        if (!resource.metadata.keywords.empty()) {
            metadata["keywords"] = resource.metadata.keywords;
        }
        if (!resource.metadata.entities.empty()) {
            metadata["entities"] = resource.metadata.entities;
        }
        if (!metadata.empty()) {
            entry["metadata"] = std::move(metadata);
        }

        result.content.push_back(std::move(entry));
    }
}

agent_mcp_json render_mcp_resource(
        const agent_resource_descriptor & descriptor) {
    agent_mcp_json entry = {
        {"uri", descriptor.uri},
        {"name", descriptor.name},
        {"description", descriptor.description},
        {"mimeType", descriptor.mime_type},
        {"sizeBytes", descriptor.size_bytes},
    };

    agent_mcp_json metadata = agent_mcp_json::object();
    if (!descriptor.lineage.parent_uri.empty()) {
        entry["lineage"] = {
            {"parent_uri", descriptor.lineage.parent_uri},
            {"chunk_index", descriptor.lineage.chunk_index},
            {"chunk_count", descriptor.lineage.chunk_count},
            {"byte_offset", descriptor.lineage.byte_offset},
            {"byte_length", descriptor.lineage.byte_length},
            {"overlap_bytes", descriptor.lineage.overlap_bytes},
            {"derivation", descriptor.lineage.derivation},
        };
    }
    if (!descriptor.metadata.purpose.empty()) {
        metadata["purpose"] = descriptor.metadata.purpose;
    }
    if (!descriptor.metadata.content_summary.empty()) {
        metadata["content_summary"] = descriptor.metadata.content_summary;
    }
    if (!descriptor.metadata.usage_hint.empty()) {
        metadata["usage_hint"] = descriptor.metadata.usage_hint;
    }
    if (!descriptor.metadata.limitations.empty()) {
        metadata["limitations"] = descriptor.metadata.limitations;
    }
    if (!descriptor.metadata.keywords.empty()) {
        metadata["keywords"] = descriptor.metadata.keywords;
    }
    if (!descriptor.metadata.entities.empty()) {
        metadata["entities"] = descriptor.metadata.entities;
    }
    if (!metadata.empty()) {
        entry["metadata"] = std::move(metadata);
    }

    return entry;
}

agent_mcp_server_tool_result render_mcp_result(
        const agent_tool_result & tool_result) {
    agent_mcp_server_tool_result result;
    result.ok = tool_result.ok;
    result.failure_code = tool_result.failure_code;
    result.failure_class = common_tool_failure_class_name(tool_result.failure_class);
    result.retryable = tool_result.retryable;
    result.safe_summary = tool_result.safe_summary;
    result.raw_diagnostic = tool_result.raw_diagnostic;

    const auto structured = agent_mcp_json::parse(tool_result.content_json, nullptr, false);
    if (!structured.is_discarded()) {
        result.structured_content = structured;
    }

    const std::string text = summarize_tool_result(tool_result, result.structured_content);
    if (!text.empty()) {
        result.content = agent_mcp_json::array({
            {
                {"type", "text"},
                {"text", text},
            },
        });
    } else {
        result.content = agent_mcp_json::array();
    }

    append_resource_links(tool_result.resource_refs, result);
    return result;
}

bool parse_server_args(
        int argc,
        char ** argv,
        args & options,
        std::vector<agent_host_mcp_provider_config> & configured_mcp_providers,
        server_http_options & http,
        std::string & error) {
    options = {};
    configured_mcp_providers.clear();
    http = {};
    options.command = "agent-mcp-stdio-server";
    options.memory_scope = "session";
    options.memory_namespace = "local";
    options.memory_session = "default";
    options.tool_profile = "minimal";
    options.max_tool_rounds = 16;
    options.memory_token_budget = 768;

    const char * explicit_config_path = find_server_config_path(argc, argv);
    std::string resolved_config_path;
    std::string config_resolution_error;
    if (resolve_agent_host_config_path(
            explicit_config_path != nullptr ? explicit_config_path : "",
            resolved_config_path,
            config_resolution_error)) {
        agent_host_config config;
        if (!load_agent_host_config(resolved_config_path, config, error)) {
            return false;
        }
        apply_agent_host_config_to_args(config, options);
        configured_mcp_providers = config.mcp_providers;
    } else if (!config_resolution_error.empty()) {
        error = config_resolution_error;
        return false;
    }

    auto need_value = [&](const char * flag, int & index) -> const char * {
        if (index + 1 >= argc) {
            error = std::string(flag) + " requires a value";
            return nullptr;
        }
        return argv[++index];
    };

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--config") == 0) {
            const char * value = need_value(argv[i], i); if (!value) return false; (void) value;
        } else if (std::strcmp(argv[i], "--tool-profile") == 0) {
            const char * value = need_value(argv[i], i); if (!value) return false; options.tool_profile = value;
        } else if (std::strcmp(argv[i], "--repository-root") == 0) {
            const char * value = need_value(argv[i], i); if (!value) return false; options.repository_root = value;
        } else if (std::strcmp(argv[i], "--backend") == 0) {
            const char * value = need_value(argv[i], i); if (!value) return false; options.backend = value;
        } else if (std::strcmp(argv[i], "--memory-db") == 0) {
            const char * value = need_value(argv[i], i); if (!value) return false; options.memory_db = value;
        } else if (std::strcmp(argv[i], "--plan-backend") == 0) {
            const char * value = need_value(argv[i], i); if (!value) return false; options.plan_backend = value;
        } else if (std::strcmp(argv[i], "--plan-db") == 0) {
            const char * value = need_value(argv[i], i); if (!value) return false; options.plan_db = value;
        } else if (std::strcmp(argv[i], "--plan-id") == 0) {
            const char * value = need_value(argv[i], i); if (!value) return false; options.plan_id = value;
        } else if (std::strcmp(argv[i], "--memory-scope") == 0) {
            const char * value = need_value(argv[i], i); if (!value) return false; options.memory_scope = value;
        } else if (std::strcmp(argv[i], "--memory-namespace") == 0) {
            const char * value = need_value(argv[i], i); if (!value) return false; options.memory_namespace = value;
        } else if (std::strcmp(argv[i], "--memory-session") == 0) {
            const char * value = need_value(argv[i], i); if (!value) return false; options.memory_session = value;
        } else if (std::strcmp(argv[i], "--memory-project") == 0) {
            const char * value = need_value(argv[i], i); if (!value) return false; options.memory_project = value;
        } else if (std::strcmp(argv[i], "--memory-turn") == 0) {
            const char * value = need_value(argv[i], i); if (!value) return false; options.memory_turn = value;
        } else if (std::strcmp(argv[i], "--memory-global-opt-in") == 0) {
            options.memory_global_opt_in = true;
        } else if (std::strcmp(argv[i], "--embedding-model") == 0) {
            const char * value = need_value(argv[i], i); if (!value) return false; options.embedding_model = value;
        } else if (std::strcmp(argv[i], "--model") == 0) {
            const char * value = need_value(argv[i], i); if (!value) return false; options.model = value;
        } else if (std::strcmp(argv[i], "--n-gpu-layers") == 0 || std::strcmp(argv[i], "-ngl") == 0) {
            const char * value = need_value(argv[i], i); if (!value) return false; options.n_gpu_layers = std::atoi(value);
        } else if (std::strcmp(argv[i], "--resource-blob-backend") == 0) {
            const char * value = need_value(argv[i], i); if (!value) return false; options.resource_blob_backend = value;
        } else if (std::strcmp(argv[i], "--resource-blob-root") == 0) {
            const char * value = need_value(argv[i], i); if (!value) return false; options.resource_blob_root = value;
        } else if (std::strcmp(argv[i], "--resource-metadata-backend") == 0) {
            const char * value = need_value(argv[i], i); if (!value) return false; options.resource_metadata_backend = value;
        } else if (std::strcmp(argv[i], "--resource-metadata-db") == 0) {
            const char * value = need_value(argv[i], i); if (!value) return false; options.resource_metadata_db = value;
        } else if (std::strcmp(argv[i], "--max-tool-rounds") == 0) {
            const char * value = need_value(argv[i], i); if (!value) return false; options.max_tool_rounds = static_cast<size_t>(std::max(1, std::atoi(value)));
        } else if (std::strcmp(argv[i], "--http-listen") == 0) {
            const char * value = need_value(argv[i], i); if (!value) return false; http.enabled = true; http.listen_address = value;
        } else if (std::strcmp(argv[i], "--http-port") == 0) {
            const char * value = need_value(argv[i], i); if (!value) return false; http.enabled = true; http.port = std::atoi(value);
        } else if (std::strcmp(argv[i], "--http-path") == 0) {
            const char * value = need_value(argv[i], i); if (!value) return false; http.enabled = true; http.path = value;
        } else if (std::strcmp(argv[i], "--http-allowed-origin") == 0) {
            const char * value = need_value(argv[i], i); if (!value) return false; http.enabled = true; http.allowed_origin = value;
        } else if (std::strcmp(argv[i], "--http-token-env") == 0) {
            const char * value = need_value(argv[i], i); if (!value) return false; http.enabled = true; http.token_env = value;
        } else if (std::strcmp(argv[i], "--http-max-body-bytes") == 0) {
            const char * value = need_value(argv[i], i); if (!value) return false; http.max_body_bytes = static_cast<size_t>(std::max(1, std::atoi(value)));
        } else if (std::strcmp(argv[i], "--http-max-result-bytes") == 0) {
            const char * value = need_value(argv[i], i); if (!value) return false; http.max_result_bytes = static_cast<size_t>(std::max(1, std::atoi(value)));
        } else {
            error = "unknown argument: " + std::string(argv[i]);
            return false;
        }
    }

    common_memory_scope memory_scope;
    if (!common_memory_scope_parse(options.memory_scope, memory_scope)) {
        error = "unsupported --memory-scope: " + options.memory_scope;
        return false;
    }

    common_plan_scope plan_scope = common_cli_matching_plan_scope(memory_scope);
    options.plan_scope = plan_scope_name(plan_scope);

    if (memory_scope == common_memory_scope::turn && options.memory_turn.empty()) {
        options.memory_turn = "mcp-turn";
    }
    if (memory_scope == common_memory_scope::project && options.memory_project.empty()) {
        error = "--memory-project is required for project-scoped MCP tool export";
        return false;
    }
    if (memory_scope == common_memory_scope::global && !options.memory_global_opt_in) {
        error = "--memory-global-opt-in is required for global-scoped MCP tool export";
        return false;
    }

    if (options.tool_profile.empty()) {
        error = "--tool-profile must not be empty";
        return false;
    }

    if (http.enabled) {
        if (http.listen_address.empty() || http.path.empty() || http.token_env.empty()) {
            error = "HTTP MCP mode requires --http-listen, --http-path and --http-token-env";
            return false;
        }
        const char * token = std::getenv(http.token_env.c_str());
        if (token == nullptr || *token == '\0') {
            error = "HTTP MCP bearer token environment variable is empty: " + http.token_env;
            return false;
        }
        http.bearer_token = token;
    }

    error.clear();
    return true;
}

bool open_server_stores(
        const args & options,
        std::unique_ptr<common_memory_store> & memory_store,
        std::unique_ptr<common_plan_store> & plan_store,
        std::unique_ptr<agent_resource_store> & resource_store,
        std::string & error) {
    const std::string memory_backend = resolve_memory_backend(options.backend, options.memory_db, error);
    if (!error.empty()) {
        return false;
    }
    if (memory_backend == "cozo") {
#ifdef LLAMA_MEMORY_USE_COZO
        if (options.memory_db.empty()) {
            error = "--backend cozo requires --memory-db PATH";
            return false;
        }
        memory_store = std::make_unique<common_memory_cozo_store>();
#else
        error = "this binary was built without LLAMA_MEMORY_COZO";
        return false;
#endif
    } else if (memory_backend == "sqlite") {
#ifdef LLAMA_MEMORY_USE_SQLITE
        if (options.memory_db.empty()) { error = "--backend sqlite requires --memory-db PATH"; return false; }
        memory_store = std::make_unique<common_memory_sqlite_store>();
#else
        error = "this binary was built without LLAMA_AGENT_STORAGE_SQLITE";
        return false;
#endif
    } else if (memory_backend == "in-memory") {
        memory_store = std::make_unique<common_memory_in_memory_store>();
    } else {
        error = "unknown memory backend: " + memory_backend;
        return false;
    }
    if (!memory_store->open(options.memory_db, error)) {
        return false;
    }

    const std::string plan_backend = resolve_plan_backend(options.plan_backend, options.plan_db, error);
    if (!error.empty()) {
        return false;
    }
    if (plan_backend == "cozo") {
#ifdef LLAMA_PLAN_USE_COZO
        if (options.plan_db.empty()) {
            error = "--plan-backend cozo requires --plan-db PATH";
            return false;
        }
        plan_store = std::make_unique<common_plan_cozo_store>();
#else
        error = "this binary was built without LLAMA_PLAN_COZO";
        return false;
#endif
    } else if (plan_backend == "sqlite") {
#ifdef LLAMA_PLAN_USE_SQLITE
        if (options.plan_db.empty()) { error = "--plan-backend sqlite requires --plan-db PATH"; return false; }
        plan_store = std::make_unique<common_plan_sqlite_store>();
#else
        error = "this binary was built without LLAMA_AGENT_STORAGE_SQLITE";
        return false;
#endif
    } else if (plan_backend == "in-memory") {
        plan_store = std::make_unique<common_plan_in_memory_store>();
    } else {
        error = "unknown plan backend: " + plan_backend;
        return false;
    }
    if (!plan_store->open(options.plan_db, error)) {
        return false;
    }

    resource_store = make_agent_resource_store({
        options.resource_blob_backend,
        options.resource_blob_root,
        options.resource_metadata_backend,
        options.resource_metadata_db,
    }, error);
    if (!resource_store) {
        return false;
    }

    error.clear();
    return true;
}

common_memory_query make_memory_query(const args & options) {
    common_memory_query query;
    query.limit = options.limit;
    query.token_budget = options.memory_token_budget;
    apply_memory_scope(options, query);
    return query;
}

agent_tool_context make_tool_context(
        const args & options,
        const std::string & repository_root,
        const std::vector<agent_host_mcp_provider_config> & configured_mcp_providers) {
    agent_tool_context context;
    context.request_id = "mcp-stdio-server";
    context.turn_id = options.memory_turn.empty() ? "mcp-turn" : options.memory_turn;
    context.profile_id = options.tool_profile;
    context.repository_root = repository_root;
    context.memory_scope = common_cli_memory_scope(options);
    parse_plan_scope(options.plan_scope, context.plan_scope);
    context.scope.namespace_id = options.memory_namespace;
    context.scope.session_id = options.memory_session;
    context.scope.project_id = options.memory_project;
    context.scope.turn_id = context.turn_id;
    context.scope.memory_scope = context.memory_scope;
    context.scope.plan_scope = context.plan_scope;
    context.scope.memory_global_opt_in = options.memory_global_opt_in;
    context.allow_network = has_enabled_stdio_mcp_provider(configured_mcp_providers);
    context.allow_policy_gated_writes = false;
    context.allow_memory_proposals = context.allow_policy_gated_writes;
    context.allow_plan_proposals = context.allow_policy_gated_writes;
    context.max_calls = std::max<size_t>(options.max_tool_rounds, 1);
    return context;
}

agent_host_tool_selection_request make_server_tool_selection_request(
        const args & options,
        const std::string & repository_root,
        const std::vector<agent_host_mcp_provider_config> & configured_mcp_providers) {
    agent_host_tool_selection_request request;
    request.tool_context = make_tool_context(options, repository_root, configured_mcp_providers);
    request.repository_root = repository_root;
    request.resource_store_config = {
        options.resource_blob_backend,
        options.resource_blob_root,
        options.resource_metadata_backend,
        options.resource_metadata_db,
    };
    request.tool_capabilities = options.tool_capabilities;
    request.tool_profiles = options.tool_profiles;
    request.sandbox = options.sandbox;
    request.resource_processor_policies = options.resource_processor_policies;
    append_configured_stdio_mcp_providers(configured_mcp_providers, request.mcp_providers);
    return request;
}

bool resolve_server_tool_selection(
        common_memory_store & memory_store,
        common_plan_store & plan_store,
        agent_resource_store & resource_store,
        std::string * current_plan_id,
        const std::string & tool_profile,
        const agent_host_tool_selection_request & request,
        const common_memory_query & query,
        agent_embedding_provider * embedding_provider,
        common_agent_cli_tool_selection & selection,
        std::string & error) {
    return resolve_agent_host_tool_selection(
        memory_store,
        &plan_store,
        &resource_store,
        current_plan_id,
        tool_profile,
        request,
        query,
        embedding_provider,
        selection,
        error);
}

bool register_resolved_profile_tools(
        const common_tool_catalog & catalog,
        const std::string & profile_id,
        const args & options,
        const common_agent_cli_tool_selection & initial_selection,
        const std::function<bool(common_agent_cli_tool_selection &, std::string &)> & resolve_selection,
        agent_mcp_server_tool_registry & registry,
        std::string & error) {
    if (initial_selection.tool_view == nullptr) {
        error = "failed to resolve native tool view";
        return false;
    }

    const std::vector<common_tool_definition> definitions = catalog.load_profile(profile_id, error);
    if (!error.empty()) {
        return false;
    }

    std::map<std::string, common_tool_definition> definitions_by_name;
    for (const auto & definition : definitions) {
        definitions_by_name.emplace(definition.name, definition);
    }

    for (const auto & tool : initial_selection.tooling.tools) {
        if (options.plan_db.empty() && tool.name.rfind("plan_", 0) == 0) {
            continue;
        }

        const auto definition_it = definitions_by_name.find(tool.name);
        const bool has_definition = definition_it != definitions_by_name.end();
        const common_tool_definition definition = has_definition ? definition_it->second : common_tool_definition{};
        if (!registry.register_tool({
                tool.name,
                tool.description.empty()
                    ? (has_definition ? definition.description : tool.name)
                    : tool.description,
                tool.parameters.empty()
                    ? (has_definition ? definition.input_schema_json : R"({"type":"object"})")
                    : tool.parameters,
                initial_selection.tool_view->is_read_only(tool.name),
                has_definition
                    ? definition.requires_confirmation
                    : initial_selection.tool_view->is_policy_gated(tool.name),
                has_definition && definition.risk_class == common_tool_risk_class::network_read,
                has_definition && definition.risk_class == common_tool_risk_class::memory_proposal,
                has_definition && definition.risk_class == common_tool_risk_class::plan_proposal,
                [resolve_selection, tool_name = tool.name](
                        const agent_mcp_json & arguments,
                        agent_mcp_server_tool_result & result,
                        std::string & call_error) {
                    std::string error;
                    common_agent_cli_tool_selection selection;
                    if (!resolve_selection(selection, error) || selection.tool_view == nullptr) {
                        call_error = "failed to resolve MCP stdio server tool view: " + error;
                        return false;
                    }

                    const agent_tool_result tool_result = selection.tool_view->call({
                        "mcp-stdio-server:" + tool_name,
                        tool_name,
                        arguments.dump(),
                    }, error);
                    result = render_mcp_result(tool_result);
                    call_error = tool_result.ok ? std::string() : error;
                    return true;
                },
            }, error)) {
            return false;
        }
    }

    error.clear();
    return true;
}

} // namespace

int main(int argc, char ** argv) {
#ifdef _WIN32
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
#endif

    args options;
    std::vector<agent_host_mcp_provider_config> configured_mcp_providers;
    server_http_options http_options;
    std::string error;
    if (!parse_server_args(argc, argv, options, configured_mcp_providers, http_options, error)) {
        std::fprintf(stderr, "failed to parse MCP stdio server args: %s\n", error.c_str());
        return 1;
    }

    std::string repository_root;
    if (!options.repository_root.empty()) {
        repository_root = std::filesystem::weakly_canonical(options.repository_root).string();
    }

    std::unique_ptr<common_memory_store> memory_store;
    std::unique_ptr<common_plan_store> plan_store;
    std::unique_ptr<agent_resource_store> resource_store;
    if (!open_server_stores(options, memory_store, plan_store, resource_store, error)) {
        std::fprintf(stderr, "failed to open MCP stdio server stores: %s\n", error.c_str());
        return 1;
    }

    std::unique_ptr<agent_embedding_provider> embedding_provider;
    const std::string embedding_model = options.embedding_model;
    if (!embedding_model.empty()) {
        embedding_provider = std::make_unique<server_agent_embedding_provider>(
            embedding_model,
            options.n_gpu_layers);
    }

    common_tool_catalog catalog;
    common_tool_bootstrap_result bootstrap;
    if (!catalog.bootstrap(
            options.tool_profile,
            bootstrap,
            error,
            options.tool_capabilities,
            options.tool_profiles)) {
        std::fprintf(stderr, "failed to bootstrap MCP stdio server tool profile: %s\n", error.c_str());
        return 1;
    }

    std::string current_plan_id = options.plan_id;
    auto tool_request = make_server_tool_selection_request(
        options,
        repository_root,
        configured_mcp_providers);
    const auto memory_query = make_memory_query(options);
    const auto resolve_selection = [&](
            common_agent_cli_tool_selection & selection,
            std::string & selection_error) {
        return resolve_server_tool_selection(
            *memory_store,
            *plan_store,
            *resource_store,
            &current_plan_id,
            options.tool_profile,
            tool_request,
            memory_query,
            embedding_provider.get(),
            selection,
            selection_error);
    };
    common_agent_cli_tool_selection initial_selection;
    if (!resolve_selection(initial_selection, error)) {
        std::fprintf(stderr, "failed to resolve MCP stdio server tool view: %s\n", error.c_str());
        return 1;
    }

    agent_resource_runtime resource_runtime;
    resource_runtime.store = resource_store.get();
    resource_runtime.namespace_id = options.memory_namespace;
    resource_runtime.session_id = options.memory_session;
    resource_runtime.project_id = options.memory_project;
    resource_runtime.turn_id = options.memory_turn.empty() ? "mcp-turn" : options.memory_turn;
    const auto resource_authority = make_agent_resource_read_authority(resource_runtime);

    agent_mcp_server_tool_registry registry;
    if (!register_resolved_profile_tools(
            catalog,
            options.tool_profile,
            options,
            initial_selection,
            resolve_selection,
            registry,
            error)) {
        std::fprintf(stderr, "failed to register MCP stdio server tools: %s\n", error.c_str());
        return 1;
    }

    const auto list_resources = [resource_store = resource_store.get(), resource_authority](agent_mcp_json & result, std::string & callback_error) {
                std::vector<agent_resource_descriptor> descriptors;
                if (!resource_store->list(resource_authority, descriptors, callback_error)) {
                    return false;
                }
                result = {
                    {"resources", agent_mcp_json::array()},
                };
                for (const auto & descriptor : descriptors) {
                    result["resources"].push_back(render_mcp_resource(descriptor));
                }
                callback_error.clear();
                return true;
            };
    const auto read_resource = [resource_store = resource_store.get(), resource_authority](const agent_mcp_json & params, agent_mcp_json & result, std::string & callback_error) {
                const std::string uri = params.value("uri", "");
                if (uri.empty()) {
                    callback_error = "resources/read requires a uri";
                    return false;
                }

                agent_resource_descriptor descriptor;
                if (!resource_store->stat(uri, resource_authority, descriptor, callback_error)) {
                    return false;
                }

                std::string text;
                if (!resource_store->read_text(uri, resource_authority, 32768, text, callback_error)) {
                    return false;
                }

                agent_mcp_json content = render_mcp_resource(descriptor);
                content["text"] = text;
                result = {
                    {"contents", agent_mcp_json::array({std::move(content)})},
                };
                callback_error.clear();
                return true;
            };

    if (http_options.enabled) {
        http_options.default_policy = {
            "local-http",
            "llama-agent",
            options.memory_namespace,
            options.memory_project,
            options.tool_profile,
            {},
            configured_profile_allows_policy_gated_writes(options),
        };
        agent_mcp_http_server http_server(
            std::move(registry),
            {
                http_options.listen_address,
                http_options.port,
                http_options.path,
                http_options.allowed_origin,
                http_options.bearer_token,
                nullptr,
                {},
                http_options.max_body_bytes,
                http_options.max_result_bytes,
                "llama-agent-mcp-http-server",
                "0.2",
                "2024-11-05",
                list_resources,
                read_resource,
                {},
            });
        if (!http_server.listen(error)) {
            std::fprintf(stderr, "failed to run MCP HTTP server: %s\n", error.c_str());
            return 1;
        }
        return 0;
    }

    agent_mcp_stdio_server server(
        std::move(registry),
        {
            "llama-agent-mcp-stdio-server",
            "0.2",
            "2024-11-05",
            false,
            false,
            false,
            list_resources,
            read_resource,
        });
    return server.run(stdin, stdout, stderr);
}
