#include "agent-cli-host-adapter.h"

#include "tools/agent/cli/agent-cli-memory-tools.h"
#include "tools/agent/resource/processors/agent-pdf-page-image-processor.h"
#include "tools/agent/resource/processors/agent-pdf-text-processor.h"
#include "tools/agent/resource/processors/agent-docx-text-processor.h"
#include "tools/agent/resource/processors/agent-tesseract-ocr-processor.h"
#include "tools/agent/resource/processors/agent-xlsx-workbook-json-processor.h"
#include "tools/agent/resource/dispatch/agent-resource-processor-dispatch.h"

#include "../runtime/agent-plan-orchestration.h"
#include "../runtime/agent-runtime-assembly.h"
#include "../runtime/agent-runtime-execution.h"
#include "agent/sandbox-docker-runtime.h"
#include "agent/sandbox-kubernetes-runtime.h"
#include "agent/sandbox-local-runtime.h"
#include "agent/sandbox-runtime.h"
#include "../diagnostics/agent-clangd-provider.h"
#include "../data/agent-data-store-factory.h"
#include "../data/agent-dataset-importer.h"
#include "../tooling/agent-sandbox-helper.h"
#include "tools/agent/cli/agent-cli-scope.h"

#include <algorithm>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <memory>
#include <cstdlib>
#include <nlohmann/json.hpp>

using json = nlohmann::ordered_json;

namespace {

class cli_operation_resource_processing_provider final
    : public agent_resource_processing_provider {
public:
    cli_operation_resource_processing_provider(
            std::shared_ptr<agent_resource_processing_host> host,
            std::vector<std::shared_ptr<agent_resource_processor>> processors,
            std::shared_ptr<agent_resource_processor_registry> registry,
            std::shared_ptr<agent_resource_processing_service> service)
        : host_(std::move(host)),
          processors_(std::move(processors)),
          registry_(std::move(registry)),
          service_(std::move(service)) {}

    agent_resource_processing_result process(
            const agent_resource_processing_binding_request & request) const override {
        return service_->process(request);
    }

private:
    // These members deliberately keep the registry's non-owning processor
    // pointers valid for the complete operation-scoped provider lifetime.
    std::shared_ptr<agent_resource_processing_host> host_;
    std::vector<std::shared_ptr<agent_resource_processor>> processors_;
    std::shared_ptr<agent_resource_processor_registry> registry_;
    std::shared_ptr<agent_resource_processing_service> service_;
};

class cli_agent_embedding_provider final : public agent_embedding_provider {
public:
    explicit cli_agent_embedding_provider(const args & options)
        : options(options) {}

    bool embed(
            const std::string & purpose,
            const std::string & text,
            std::vector<float> & embedding,
            std::string & error) override {
        return ensure_memory_cli_embedding(options, text, embedding, purpose.c_str(), error);
    }

private:
    const args & options;
};

agent_tool_context make_agent_cli_tool_context(
        const args & options,
        const common_memory_query & query,
        const std::string & repository_root) {
    agent_tool_context tool_context;
    tool_context.request_id = "cli-run";
    tool_context.turn_id = options.memory_turn;
    tool_context.scope = common_cli_make_agent_scope_with_matching_plan_scope(options);
    tool_context.memory_scope = query.scope;
    tool_context.plan_scope = tool_context.scope.plan_scope;
    tool_context.profile_id = options.tool_profile;
    tool_context.repository_root = repository_root;
    tool_context.allow_network = !options.mcp_tool_command.empty();
    tool_context.allow_policy_gated_writes = false;
    tool_context.allow_memory_proposals = tool_context.allow_policy_gated_writes;
    tool_context.allow_plan_proposals = tool_context.allow_policy_gated_writes;
    tool_context.max_calls = options.max_tool_rounds > 0 ? options.max_tool_rounds : 1;
    tool_context.default_timeout_ms = options.tool_timeout_ms > 0 ? options.tool_timeout_ms : tool_context.default_timeout_ms;
    tool_context.execution_control = make_common_agent_runtime_execution_control({
        options.turn_timeout_ms,
        options.inference_step_timeout_ms,
        options.tool_timeout_ms,
        options.mcp_connect_timeout_ms,
        options.mcp_request_timeout_ms,
        options.mcp_shutdown_timeout_ms,
    });
    return tool_context;
}

agent_host_tool_selection_request make_agent_cli_tool_selection_request(
        const args & options,
        const common_memory_query & query,
        const std::string & repository_root) {
    agent_host_tool_selection_request request;
    request.tool_context = make_agent_cli_tool_context(options, query, repository_root);
    request.repository_root = repository_root;
    request.resource_store_config = {
        options.resource_blob_backend,
        options.resource_blob_root,
        options.resource_metadata_backend,
        options.resource_metadata_db,
    };
    request.data_store_config = {options.data_backend, options.data_db};
    request.tool_capabilities = options.tool_capabilities;
    request.tool_profiles = options.tool_profiles;
    request.sandbox = options.sandbox;
    request.resource_processor_policies = options.resource_processor_policies;
    append_legacy_stdio_mcp_provider(
        options.mcp_tool_command,
        options.mcp_tool_args,
        options.mcp_tool_server_name,
        options.mcp_tool_prefix,
        request.mcp_providers);
    return request;
}

} // namespace

bool has_enabled_stdio_mcp_provider(
        const std::vector<agent_host_mcp_provider_config> & providers) {
    for (const auto & provider : providers) {
        if (provider.enabled &&
                provider.type == "mcp" &&
                provider.transport == "stdio" &&
                !provider.command.empty()) {
            return true;
        }
    }
    return false;
}

bool has_enabled_mcp_provider(
        const std::vector<agent_host_mcp_provider_config> & providers) {
    for (const auto & provider : providers) {
        if (provider.enabled && provider.type == "mcp") {
            return true;
        }
    }
    return false;
}

void append_configured_stdio_mcp_providers(
        const std::vector<agent_host_mcp_provider_config> & configured_providers,
        std::vector<agent_host_stdio_mcp_provider_request> & request_providers) {
    for (const auto & provider : configured_providers) {
        if (!provider.enabled || provider.type != "mcp" ||
                (provider.transport == "stdio" && provider.command.empty()) ||
                (provider.transport != "stdio" && provider.url.empty())) {
            continue;
        }
        agent_host_stdio_mcp_provider_request request_provider;
        request_provider.server_name = provider.server_name.empty() ? provider.id : provider.server_name;
        request_provider.transport = provider.transport;
        request_provider.exposed_name_prefix = provider.prefix;
        request_provider.command_line = provider.command;
        request_provider.url = provider.url;
        request_provider.allowed_tools = provider.allowed_tools;
        request_provider.connect_timeout_ms = provider.connect_timeout_ms;
        request_provider.request_timeout_ms = provider.request_timeout_ms;
        request_provider.shutdown_timeout_ms = provider.shutdown_timeout_ms;
        request_provider.max_result_bytes = provider.max_result_bytes;
        if (!provider.token_env.empty()) {
            const char * token = std::getenv(provider.token_env.c_str());
            if (token != nullptr) request_provider.bearer_token = token;
        }
        request_providers.push_back(std::move(request_provider));
    }
}

void append_legacy_stdio_mcp_provider(
        const std::string & command,
        const std::vector<std::string> & args,
        const std::string & server_name,
        const std::string & prefix,
        std::vector<agent_host_stdio_mcp_provider_request> & request_providers) {
    if (command.empty()) {
        return;
    }
    agent_host_stdio_mcp_provider_request provider;
    provider.server_name = server_name;
    provider.transport = "stdio";
    provider.exposed_name_prefix = prefix;
    provider.command_line.push_back(command);
    provider.command_line.insert(provider.command_line.end(), args.begin(), args.end());
    request_providers.push_back(std::move(provider));
}

common_agent_runtime_host_post_run make_agent_cli_runtime_post_run(
        common_memory_store & store,
        const args & options,
        bool memory_enabled) {
    return [&store, &options, memory_enabled](const common_agent_result &, std::string & hook_error) {
        if (!options.record_episode) {
            hook_error.clear();
            return true;
        }
        if (!memory_enabled) {
            std::fprintf(stderr, "warning: skipping episode recording because no query embedding could be generated\n");
            hook_error.clear();
            return true;
        }

        common_memory_record episode;
        episode.id = "episode-" + std::to_string(std::time(nullptr));
        episode.kind = common_memory_kind::episode;
        episode.content = options.prompt;
        episode.created_at = std::time(nullptr);
        episode.accessed_at = episode.created_at;
        episode.importance = 0.5f;
        episode.confidence = 0.5f;
        apply_memory_scope(options, episode);
        if (!store.put(episode, hook_error)) {
            std::fprintf(stderr, "failed to record memory episode: %s\n", hook_error.c_str());
            hook_error.clear();
        }
        return true;
    };
}

bool resolve_agent_host_tool_selection(
        common_memory_store & store,
        common_plan_store * plan_store,
        agent_resource_store * resource_store,
        std::string * current_plan_id,
        const std::string & tool_profile,
        const agent_host_tool_selection_request & request,
        const common_memory_query & query,
        agent_embedding_provider * embedding_provider,
        common_agent_cli_tool_selection & selection,
        std::string & error) {
    selection = {};
    selection.tooling.resource_runtime.store = resource_store;
    selection.tooling.resource_runtime.namespace_id = request.tool_context.scope.namespace_id;
    selection.tooling.resource_runtime.session_id = request.tool_context.scope.session_id;
    selection.tooling.resource_runtime.project_id = request.tool_context.scope.project_id;
    selection.tooling.resource_runtime.turn_id = request.tool_context.scope.turn_id;

    if (request.data_store == nullptr) {
        selection.owned_data_store = make_agent_data_store(request.data_store_config, error);
        if (!error.empty()) {
            error = "data store setup failed: " + error;
            return false;
        }
    }

    common_tool_catalog tool_catalog;
    std::unique_ptr<native_agent_tool_provider> native_provider;
    agent_tool_context resolved_tool_context = request.tool_context;
    if (!tool_profile.empty()) {
        common_tool_bootstrap_result bootstrap;
        if (!tool_catalog.bootstrap(
                tool_profile,
                bootstrap,
                error,
                request.tool_capabilities,
                request.tool_profiles)) {
            error = "tool bootstrap failed: " + error;
            return false;
        }
        auto profile_snapshot = std::make_shared<common_tool_profile_snapshot>();
        if (!tool_catalog.resolve_profile(tool_profile, *profile_snapshot, error)) {
            error = "tool profile snapshot resolution failed: " + error;
            return false;
        }
        // Sandbox-backed tools must not be model-visible when the host has no
        // execution backend. Keep the profile itself intact for diagnostics,
        // but resolve an effective snapshot that cannot advertise tools which
        // would only fail later with sandbox.backend_unavailable.
        if (request.sandbox.backend != "docker" && request.sandbox.backend != "kubernetes") {
            std::vector<common_tool_definition> available_tools;
            for (auto & definition : profile_snapshot->tools) {
                if (definition.risk_class != common_tool_risk_class::sandbox_execution) {
                    available_tools.push_back(std::move(definition));
                }
            }
            profile_snapshot->tools = std::move(available_tools);
        }
        if (request.data_store == nullptr && selection.owned_data_store == nullptr) {
            std::vector<common_tool_definition> available_tools;
            for (auto & definition : profile_snapshot->tools) {
                if (definition.executor_id.rfind("builtin.data.", 0) != 0 &&
                        definition.executor_id != "builtin.statistics.describe" &&
                        definition.executor_id != "builtin.statistics.outliers" &&
                        definition.executor_id != "builtin.statistics.value_counts") {
                    available_tools.push_back(std::move(definition));
                }
            }
            profile_snapshot->tools = std::move(available_tools);
        }
        for (const auto & definition : profile_snapshot->tools) {
            selection.tooling.capabilities.insert(selection.tooling.capabilities.end(),
                definition.capabilities.begin(), definition.capabilities.end());
        }
        std::sort(selection.tooling.capabilities.begin(), selection.tooling.capabilities.end());
        selection.tooling.capabilities.erase(std::unique(selection.tooling.capabilities.begin(), selection.tooling.capabilities.end()), selection.tooling.capabilities.end());
        const auto * resolved_profile = tool_catalog.find_profile(tool_profile);
        resolved_tool_context.profile_snapshot = profile_snapshot;
        if (resolved_profile != nullptr) {
            if (resolved_profile->allow_network.has_value()) {
                resolved_tool_context.allow_network = *resolved_profile->allow_network;
            }
            if (resolved_profile->allow_policy_gated_writes.has_value()) {
                resolved_tool_context.allow_policy_gated_writes = *resolved_profile->allow_policy_gated_writes;
                resolved_tool_context.allow_memory_proposals = *resolved_profile->allow_policy_gated_writes;
                resolved_tool_context.allow_plan_proposals = *resolved_profile->allow_policy_gated_writes;
            }
        }

        common_native_tool_bindings bindings;
        if (!request.repository_root.empty()) {
            bindings.repository_root = request.repository_root;
        }
        bindings.plan_store = plan_store;
        bindings.plan_id = current_plan_id;
        if (resource_store == nullptr) {
            selection.owned_resource_store = make_agent_resource_store(
                request.resource_store_config,
                error);
            if (!selection.owned_resource_store) {
                error = "resource store setup failed: " + error;
                return false;
            }
            resource_store = selection.owned_resource_store.get();
        }
        bindings.resource_runtime = selection.tooling.resource_runtime;
        bindings.resource_runtime.store = resource_store;
        selection.tooling.resource_runtime.store = resource_store;
        selection.resource_processor_registry = std::make_shared<agent_resource_processor_registry>();
        auto pdf_text_processor = std::make_shared<agent_pdf_text_processor>();
        if (!selection.resource_processor_registry->add(*pdf_text_processor, error)) {
            error = "resource processor setup failed: " + error;
            return false;
        }
        selection.resource_processors.push_back(std::move(pdf_text_processor));
        selection.resource_processing_service = std::make_shared<agent_resource_processing_service>(
            *resource_store,
            *selection.resource_processor_registry);
        bindings.resource_processing_service = selection.resource_processing_service.get();
        bindings.data_store = request.data_store != nullptr
            ? request.data_store
            : selection.owned_data_store.get();
        if (bindings.data_store != nullptr && resource_store != nullptr) {
            const auto runtime = bindings.resource_runtime;
            bindings.dataset_from_resource = [resource_store, runtime, data_store = bindings.data_store](
                    const std::string & uri, const std::string & operation) {
                std::string error;
                agent_resource_descriptor resource;
                const auto authority = make_agent_resource_read_authority(runtime, std::time(nullptr));
                if (!resource_store->stat(uri, authority, resource, error)) {
                    return common_tool_execution_result::failure(
                        "tool.dataset.resource_unavailable", common_tool_failure_class::not_found, false,
                        "The resource is unavailable as a dataset.", std::move(error));
                }
                if (common_normalize_resource_media_type(resource.mime_type) != "text/csv") {
                    return common_tool_execution_result::failure(
                        "tool.dataset.resource_unsupported", common_tool_failure_class::validation, false,
                        "Only CSV resources are currently materialized as datasets.",
                        "resource MIME type is not text/csv");
                }
                std::string csv;
                if (!resource_store->read_bytes(uri, authority, 128 * 1024 * 1024, csv, error)) {
                    return common_tool_execution_result::failure(
                        "tool.dataset.resource_read_failed", common_tool_failure_class::execution, true,
                        "The CSV resource could not be read.", std::move(error));
                }
                std::string worksheet_json;
                if (!normalize_agent_csv_text(csv, resource.name, worksheet_json, error)) {
                    return common_tool_execution_result::failure(
                        "tool.dataset.resource_invalid", common_tool_failure_class::validation, false,
                        "The CSV resource could not be normalized.", std::move(error));
                }
                agent_dataset_import_request import;
                import.source_resource_uri = resource.uri;
                import.source_workbook_name = resource.name;
                import.source_representation = "csv:dataset";
                import.source_representation_uri = resource.uri;
                import.import_processor_id = "csv-resource-import-v1";
                import.import_processor_version = "1";
                import.worksheet_json = std::move(worksheet_json);
                std::vector<common_agent_dataset_descriptor> imported;
                if (!import_agent_worksheet_envelope(*data_store, import, imported, error) || imported.size() != 1) {
                    return common_tool_execution_result::failure(
                        "tool.dataset.materialization_failed", common_tool_failure_class::execution, false,
                        "The CSV resource could not be materialized as a dataset.", std::move(error));
                }
                const auto & dataset = imported.front();
                if (operation == "schema") {
                    json columns = json::array();
                    for (const auto & column : dataset.columns) columns.push_back({
                        {"name", column.name}, {"type", common_agent_dataset_column_type_name(column.type)},
                        {"nullable", column.nullable}});
                    return common_tool_execution_result::success(json({
                        {"dataset", dataset.ref.uri}, {"source", dataset.ref.source_resource_uri},
                        {"columns", columns}}).dump());
                }
                if (operation == "sample") {
                    json query = {{"dataset", dataset.ref.uri}, {"limit", 20},
                        {"max_scan_rows", 10000}, {"max_result_rows", 100}};
                    std::string sample_json;
                    if (!data_store->execute("data.query", query.dump(), sample_json, error)) {
                        return common_tool_execution_result::failure(
                            "tool.dataset.sample_failed", common_tool_failure_class::execution, false,
                            "The CSV dataset sample could not be read.", std::move(error));
                    }
                    return common_tool_execution_result::success(std::move(sample_json));
                }
                return common_tool_execution_result::success(json({
                    {"dataset", dataset.ref.uri}, {"name", dataset.ref.name},
                    {"rows", dataset.ref.row_count}, {"columns", dataset.ref.column_count},
                    {"source", dataset.ref.source_resource_uri}}).dump());
            };
            auto read_document_json = [resource_store, runtime](
                    const std::string & uri, std::string & document_json,
                    agent_resource_descriptor & descriptor, std::string & read_error) {
                const auto authority = make_agent_resource_read_authority(runtime, std::time(nullptr));
                if (!resource_store->stat(uri, authority, descriptor, read_error)) {
                    std::vector<agent_resource_descriptor> visible_resources;
                    std::string list_error;
                    if (resource_store->list(authority, visible_resources, list_error)) {
                        read_error += " (visible_resources=" + std::to_string(visible_resources.size()) + ")";
                    }
                    return false;
                }
                if (descriptor.mime_type != "application/json") {
                    read_error = "document table tools require an application/json document representation";
                    return false;
                }
                return resource_store->read_bytes(uri, authority, 4 * 1024 * 1024, document_json, read_error);
            };
            bindings.document_tables = [read_document_json](const std::string & input) mutable {
                const auto arguments = json::parse(input, nullptr, false);
                if (!arguments.is_object()) return common_tool_execution_result::failure(
                    "tool.document.tables.invalid_arguments", common_tool_failure_class::validation, false,
                    "Document table arguments are invalid.", "invalid document table arguments");
                agent_resource_descriptor descriptor;
                std::string document_json, error;
                if (!read_document_json(arguments["resource"].get<std::string>(), document_json, descriptor, error))
                    return common_tool_execution_result::failure("tool.document.tables.unavailable", common_tool_failure_class::not_found, false,
                        "The document representation is unavailable.", std::move(error));
                std::string worksheet_json;
                if (!normalize_agent_pandoc_document_json(document_json, worksheet_json, error))
                    return common_tool_execution_result::failure("tool.document.tables.invalid_document", common_tool_failure_class::validation, false,
                        "The document representation could not be normalized.", std::move(error));
                common_agent_document_table_catalog catalog;
                if (!make_agent_document_table_catalog(worksheet_json, catalog, error))
                    return common_tool_execution_result::failure("tool.document.tables.invalid_document", common_tool_failure_class::validation, false,
                        "The document table catalog could not be created.", std::move(error));
                const size_t limit = std::min<size_t>(arguments.value("max_results", 32), catalog.tables.size());
                json tables = json::array();
                for (size_t index = 0; index < limit; ++index) {
                    const auto & table = catalog.tables[index];
                    tables.push_back({{"index", table.table_index}, {"name", table.name},
                        {"caption", table.caption}, {"node_id", table.node_id}});
                }
                return common_tool_execution_result::success(json({
                    {"resource", arguments["resource"]}, {"tables", tables},
                    {"truncated", catalog.tables.size() > limit}}).dump());
            };
            bindings.document_table = [read_document_json, data_store = bindings.data_store](const std::string & input) mutable {
                const auto arguments = json::parse(input, nullptr, false);
                if (!arguments.is_object()) return common_tool_execution_result::failure(
                    "tool.document.table.invalid_arguments", common_tool_failure_class::validation, false,
                    "Document table arguments are invalid.", "invalid document table arguments");
                agent_resource_descriptor descriptor;
                std::string document_json, error;
                if (!read_document_json(arguments["resource"].get<std::string>(), document_json, descriptor, error))
                    return common_tool_execution_result::failure("tool.document.table.unavailable", common_tool_failure_class::not_found, false,
                        "The document representation is unavailable.", std::move(error));
                std::string worksheet_json;
                if (!normalize_agent_pandoc_document_json(document_json, worksheet_json, error))
                    return common_tool_execution_result::failure("tool.document.table.invalid_document", common_tool_failure_class::validation, false,
                        "The document representation could not be normalized.", std::move(error));
                common_agent_document_table_catalog catalog;
                if (!make_agent_document_table_catalog(worksheet_json, catalog, error))
                    return common_tool_execution_result::failure("tool.document.table.invalid_document", common_tool_failure_class::validation, false,
                        "The document table catalog could not be created.", std::move(error));
                common_agent_document_table_locator locator;
                if (arguments.contains("table")) locator.name = arguments["table"].get<std::string>();
                else if (arguments.contains("table_index")) locator.table_index = arguments["table_index"].get<size_t>();
                else locator.node_id = arguments["node_id"].get<std::string>();
                common_agent_document_table_entry selected;
                if (!resolve_common_agent_document_table(catalog, locator, selected, error))
                    return common_tool_execution_result::failure("tool.document.table.not_found", common_tool_failure_class::not_found, false,
                        "The requested document table was not found.", std::move(error));
                agent_dataset_import_request import;
                // The catalog descriptor is intentionally bounded and does
                // not carry the full derived-resource lineage. The imported
                // dataset keeps the semantic representation URI; the
                // resource store remains authoritative for its parent chain.
                import.source_resource_uri = descriptor.uri;
                import.source_workbook_name = descriptor.name;
                import.source_representation = "document:table";
                import.source_representation_uri = descriptor.uri;
                import.import_processor_id = "document-json-table-import-v1";
                import.import_processor_version = "1";
                import.worksheet_json = worksheet_json;
                import.sheet_index = selected.table_index;
                std::vector<common_agent_dataset_descriptor> imported;
                if (!import_agent_worksheet_envelope(*data_store, import, imported, error) || imported.size() != 1)
                    return common_tool_execution_result::failure("tool.document.table.materialization_failed", common_tool_failure_class::execution, false,
                        "The selected document table could not be materialized.", std::move(error));
                return common_tool_execution_result::success(json({
                    {"table_index", selected.table_index}, {"name", selected.name},
                    {"node_id", selected.node_id}, {"dataset", imported.front().ref.uri},
                    {"source_resource", imported.front().ref.source_resource_uri}}).dump());
            };
        }
        bindings.memory_store = &store;
        bindings.memory_query = query;
#ifdef LLAMA_AGENT_TOOLS_USE_CLANG
        std::shared_ptr<agent_clangd_diagnostics_provider> clangd_provider;
        if (request.diagnostics.semantic_backend == "clangd" && !request.repository_root.empty()) {
            clangd_provider = std::make_shared<agent_clangd_diagnostics_provider>(agent_clangd_session_config{
                request.diagnostics.clangd_executable,
                request.repository_root,
                request.diagnostics.compile_commands,
                request.tool_context.default_timeout_ms,
                request.tool_context.default_timeout_ms,
            });
            bindings.diagnostics_symbol = [clangd_provider](const std::string & input) { return clangd_provider->symbol(input); };
            bindings.diagnostics_references = [clangd_provider](const std::string & input) { return clangd_provider->references(input); };
            bindings.diagnostics_call_hierarchy = [clangd_provider](const std::string & input) { return clangd_provider->call_hierarchy(input); };
        }
#endif
        if (embedding_provider != nullptr) {
            bindings.embed_memory_query = [provider = embedding_provider](const std::string & text, std::vector<float> & embedding, std::string & embedding_error) {
                return provider != nullptr &&
                    provider->embed("tool query", text, embedding, embedding_error);
            };
        }

        std::shared_ptr<common_agent_sandbox_docker_runtime> docker_runtime;
        std::shared_ptr<common_agent_sandbox_kubernetes_runtime> kubernetes_runtime;
        std::shared_ptr<common_agent_workspace_manager> workspace_manager;
        if (request.sandbox.backend == "docker" || request.sandbox.backend == "kubernetes" ||
                request.resource_processor_policies.find("pdf.page_image") != request.resource_processor_policies.end() ||
                request.resource_processor_policies.find("ocr.tesseract") != request.resource_processor_policies.end() ||
                request.resource_processor_policies.find("docx.text") != request.resource_processor_policies.end() ||
                request.resource_processor_policies.find("odt.text") != request.resource_processor_policies.end() ||
                request.resource_processor_policies.find("html.text") != request.resource_processor_policies.end() ||
                request.resource_processor_policies.find("xlsx.workbook") != request.resource_processor_policies.end()) {
            const auto backend = request.sandbox.backend;
            docker_runtime = std::make_shared<common_agent_sandbox_docker_runtime>(
                common_agent_docker_sandbox_config{
                    request.sandbox.docker_executable,
                    request.sandbox.docker_default_image,
                });
            workspace_manager = std::make_shared<common_agent_workspace_manager>(
                request.sandbox.workspace);
            const auto policies = request.sandbox.classes;
            const auto defaults = request.sandbox.defaults;
            const auto tool_context = request.tool_context;
            auto * bound_resource_store = resource_store;
            kubernetes_runtime = std::make_shared<common_agent_sandbox_kubernetes_runtime>(
                common_agent_kubernetes_sandbox_config{
                    request.sandbox.kubernetes_executable,
                    request.sandbox.kubernetes_kubeconfig,
                    request.sandbox.kubernetes_context,
                    request.sandbox.kubernetes_insecure_skip_tls_verify,
                    request.sandbox.kubernetes_namespace,
                    request.sandbox.kubernetes_service_account,
                    request.sandbox.kubernetes_runtime_class,
                    request.sandbox.kubernetes_storage_class,
                    request.sandbox.kubernetes_workspace_storage_size,
                    request.sandbox.kubernetes_artifact_storage_size,
                    request.sandbox.kubernetes_staging_image,
                    request.sandbox.kubernetes_pvc_retention,
                    request.sandbox.kubernetes_staging_timeout_ms,
                    request.sandbox.kubernetes_cleanup,
                });
            bindings.sandbox_execute = [backend, docker_runtime, kubernetes_runtime, workspace_manager, policies, defaults, tool_context, bound_resource_store](
                    common_agent_sandbox_request sandbox_request) {
                common_agent_sandbox_policy policy = defaults;
                const auto it = policies.find(sandbox_request.execution_class);
                if (it != policies.end()) policy = it->second;
                policy.execution_class = sandbox_request.execution_class;
                common_agent_sandbox_runtime & runtime = backend == "docker"
                    ? static_cast<common_agent_sandbox_runtime &>(*docker_runtime)
                    : static_cast<common_agent_sandbox_runtime &>(*kubernetes_runtime);
                common_agent_sandbox_tool_helper helper(runtime, policy);
                helper.set_workspace_manager(workspace_manager.get());
                helper.set_resource_store(bound_resource_store, {
                    tool_context.scope.namespace_id,
                    tool_context.scope.session_id,
                    tool_context.scope.project_id,
                    tool_context.scope.turn_id,
                });
                common_agent_workspace_context workspace_context;
                workspace_context.workspace_id = tool_context.scope.project_id.empty()
                    ? "session:" + tool_context.scope.session_id
                    : "project:" + tool_context.scope.project_id;
                workspace_context.project_id = tool_context.scope.project_id;
                workspace_context.namespace_id = tool_context.scope.namespace_id;
                workspace_context.session_id = tool_context.scope.session_id;
                workspace_context.turn_id = tool_context.scope.turn_id;
                workspace_context.input_resources = sandbox_request.workspace.input_resources;
                return helper.run_for_workspace(
                    workspace_context,
                    sandbox_request.operation_id,
                    std::move(sandbox_request));
            };
        } else {
            auto unavailable_runtime = std::make_shared<common_agent_sandbox_unavailable_runtime>();
            const auto policies = request.sandbox.classes;
            const auto defaults = request.sandbox.defaults;
            bindings.sandbox_execute = [unavailable_runtime, policies, defaults](
                    common_agent_sandbox_request sandbox_request) {
                common_agent_sandbox_policy policy = defaults;
                const auto it = policies.find(sandbox_request.execution_class);
                if (it != policies.end()) policy = it->second;
                policy.execution_class = sandbox_request.execution_class;
                common_agent_sandbox_tool_helper helper(*unavailable_runtime, policy);
                return helper.run(sandbox_request);
            };
        }

        if (request.sandbox.backend == "docker" || request.sandbox.backend == "kubernetes" ||
                request.resource_processor_policies.find("pdf.page_image") != request.resource_processor_policies.end() ||
                request.resource_processor_policies.find("ocr.tesseract") != request.resource_processor_policies.end() ||
                request.resource_processor_policies.find("docx.text") != request.resource_processor_policies.end() ||
                request.resource_processor_policies.find("odt.text") != request.resource_processor_policies.end() ||
                request.resource_processor_policies.find("html.text") != request.resource_processor_policies.end() ||
                request.resource_processor_policies.find("xlsx.workbook") != request.resource_processor_policies.end()) {
            const auto processor_policies = request.resource_processor_policies;
            const auto sandbox_classes = request.sandbox.classes;
            const auto sandbox_defaults = request.sandbox.defaults;
            auto docker_runtime_for_processors = docker_runtime;
            auto kubernetes_runtime_for_processors = kubernetes_runtime;
            auto local_runtime_for_processors = std::make_shared<common_agent_sandbox_local_runtime>();
            auto workspace_manager_for_processors = workspace_manager;
            auto * bound_resource_store_for_processors = resource_store;
            const auto sandbox_backend = request.sandbox.backend;
            bindings.resource_processing_provider_factory = [
                    processor_policies,
                    sandbox_classes,
                    sandbox_defaults,
                    docker_runtime_for_processors,
                    kubernetes_runtime_for_processors,
                    local_runtime_for_processors,
                    workspace_manager_for_processors,
                    bound_resource_store_for_processors,
                    sandbox_backend](const agent_resource_processing_binding_request & binding)
                    -> std::shared_ptr<agent_resource_processing_provider> {
                const auto source_type = common_normalize_resource_media_type(
                    !binding.media_type.resolved_type.empty()
                        ? binding.media_type.resolved_type
                        : binding.media_type.declared_type);
                const auto dispatch = resolve_agent_resource_processor_dispatch({
                    processor_policies, source_type, sandbox_backend});
                const bool has_page_policy = dispatch.has_page_policy;
                const bool has_ocr_policy = dispatch.has_ocr_policy;
                const bool wants_page_local = dispatch.wants_page_local;
                const bool wants_page_sandbox = dispatch.wants_page_sandbox;
                const bool wants_ocr_local = dispatch.wants_ocr_local;
                const bool wants_ocr_sandbox = dispatch.wants_ocr_sandbox;
                const bool wants_pandoc_local = dispatch.wants_pandoc_local;
                const bool wants_pandoc_sandbox = dispatch.wants_pandoc_sandbox;
                const bool wants_xlsx_local = dispatch.wants_xlsx_local;
                const bool wants_xlsx_sandbox = dispatch.wants_xlsx_sandbox;
                if (!dispatch.selected) {
                    return std::shared_ptr<agent_resource_processing_provider>();
                }

                auto make_policy = [&sandbox_classes, &sandbox_defaults](
                        const std::string & execution_class,
                        const std::string & image) {
                    auto policy = sandbox_defaults;
                    const auto it = sandbox_classes.find(execution_class);
                    if (it != sandbox_classes.end()) policy = it->second;
                    policy.execution_class = execution_class;
                    if (!image.empty()) policy.image = image;
                    return policy;
                };

                const std::string operation_id = binding.operation_id.empty()
                    ? "resource-read/turn"
                    : binding.operation_id;
                common_agent_workspace_context workspace;
                workspace.namespace_id = binding.authority.namespace_id;
                workspace.session_id = binding.authority.session_id;
                workspace.project_id = binding.authority.project_id;
                workspace.turn_id = binding.authority.turn_id;
                workspace.workspace_id = workspace.project_id.empty()
                    ? "session:" + workspace.session_id
                    : "project:" + workspace.project_id;
                agent_resource_processing_execution_context execution;
                execution.workspace = workspace;
                execution.operation_id = operation_id;

                std::shared_ptr<agent_resource_processing_host> host;
                const auto & selected_policy = dispatch.policy;
                const auto & selected_execution_class = dispatch.execution_class;
                const auto & execution_backend = dispatch.execution_backend;
                if (execution_backend == "docker") {
                    auto value = std::make_shared<agent_sandbox_resource_processing_host>(
                        *docker_runtime_for_processors,
                        make_policy(selected_execution_class, selected_policy.image));
                    value->set_workspace_manager(workspace_manager_for_processors.get());
                    value->set_resource_store(bound_resource_store_for_processors, binding.authority);
                    host = std::move(value);
                } else if (execution_backend == "kubernetes") {
                    auto value = std::make_shared<agent_sandbox_resource_processing_host>(
                        *kubernetes_runtime_for_processors,
                        make_policy(selected_execution_class, selected_policy.image));
                    value->set_workspace_manager(workspace_manager_for_processors.get());
                    value->set_resource_store(bound_resource_store_for_processors, binding.authority);
                    host = std::move(value);
                } else if (execution_backend == "local" &&
                        (wants_page_local || wants_ocr_local || wants_pandoc_local || wants_xlsx_local)) {
                    auto value = std::make_shared<agent_sandbox_resource_processing_host>(
                        *local_runtime_for_processors,
                        make_policy(selected_execution_class, selected_policy.image));
                    value->set_workspace_manager(workspace_manager_for_processors.get());
                    value->set_resource_store(bound_resource_store_for_processors, binding.authority);
                    host = std::move(value);
                } else {
                    return std::shared_ptr<agent_resource_processing_provider>();
                }

                auto registry = std::make_shared<agent_resource_processor_registry>();
                std::vector<std::shared_ptr<agent_resource_processor>> processors;
                std::string registration_error;
                if (has_page_policy) {
                    const auto & policy = *dispatch.page_policy;
                    auto processor = std::make_shared<agent_pdf_page_image_processor>(
                        *host,
                        execution,
                        agent_resource_backend_kind::local_mupdf,
                        policy.executable.empty() ? "mutool" : policy.executable);
                    if (!registry->add(*processor, registration_error)) return std::shared_ptr<agent_resource_processing_provider>();
                    processors.push_back(std::move(processor));
                }
                if (has_ocr_policy) {
                    const auto & policy = *dispatch.ocr_policy;
                    auto processor = std::make_shared<agent_tesseract_ocr_processor>(
                        *host,
                        execution,
                        agent_resource_backend_kind::local_tesseract,
                        policy.executable.empty() ? "tesseract" : policy.executable);
                    if (!registry->add(*processor, registration_error)) return std::shared_ptr<agent_resource_processing_provider>();
                    processors.push_back(std::move(processor));
                }
                if (wants_pandoc_local) {
                    const auto & policy = *dispatch.selected_pandoc_policy;
                    agent_pandoc_options options;
                    if (source_type == "application/vnd.oasis.opendocument.text") {
                        options.input_format = "odt";
                        options.output_format = "markdown";
                        options.output_extension = "md";
                    } else if (source_type == "text/html") {
                        options.input_format = "html";
                        options.output_format = "markdown";
                        options.output_extension = "md";
                    }
                    auto processor = std::make_shared<agent_pandoc_processor>(
                        *host,
                        execution,
                        agent_resource_backend_kind::local_pandoc,
                        policy.executable.empty() ? "pandoc" : policy.executable,
                        options);
                    if (!registry->add(*processor, registration_error)) return std::shared_ptr<agent_resource_processing_provider>();
                    processors.push_back(std::move(processor));
                }
                if (wants_pandoc_sandbox) {
                    const auto & policy = *dispatch.selected_pandoc_policy;
                    agent_pandoc_options options;
                    if (source_type == "application/vnd.oasis.opendocument.text") {
                        options.input_format = "odt";
                        options.output_format = "markdown";
                        options.output_extension = "md";
                    } else if (source_type == "text/html") {
                        options.input_format = "html";
                        options.output_format = "markdown";
                        options.output_extension = "md";
                    }
                    const auto backend = policy.backend == "kubernetes"
                        ? agent_resource_backend_kind::kubernetes
                        : agent_resource_backend_kind::docker;
                    auto processor = std::make_shared<agent_pandoc_processor>(
                        *host,
                        execution,
                        backend,
                        policy.executable.empty() ? "pandoc" : policy.executable,
                        options);
                    if (!registry->add(*processor, registration_error)) return std::shared_ptr<agent_resource_processing_provider>();
                    processors.push_back(std::move(processor));
                }
                if (wants_xlsx_local || wants_xlsx_sandbox) {
                    const auto & policy = *dispatch.selected_xlsx_policy;
                    const auto backend = wants_xlsx_local ? agent_resource_backend_kind::local_process
                        : (policy.backend == "kubernetes" ? agent_resource_backend_kind::kubernetes
                            : agent_resource_backend_kind::docker);
                    auto processor = std::make_shared<agent_xlsx_workbook_json_processor>(
                        *host, execution, backend,
                        policy.executable.empty() ? "python" : policy.executable,
                        policy.script.empty() ? "scripts/agent-xlsx-to-json.py" : policy.script);
                    if (!registry->add(*processor, registration_error)) return std::shared_ptr<agent_resource_processing_provider>();
                    processors.push_back(std::move(processor));
                }
                auto service = std::make_shared<agent_resource_processing_service>(
                    *bound_resource_store_for_processors, *registry);
                return std::make_shared<cli_operation_resource_processing_provider>(
                    std::move(host), std::move(processors), std::move(registry), std::move(service));
            };
        }

        selection.tooling.resource_runtime.processing_provider_factory =
            bindings.resource_processing_provider_factory;

        native_provider = std::make_unique<native_agent_tool_provider>(
            tool_catalog,
            [bindings](const agent_tool_context & context, common_native_tool_bindings & resolved, std::string & binding_error) mutable {
                resolved = bindings;
                resolved.resource_runtime.namespace_id = context.scope.namespace_id;
                resolved.resource_runtime.session_id = context.scope.session_id;
                resolved.resource_runtime.project_id = context.scope.project_id;
                resolved.resource_runtime.turn_id = context.scope.turn_id;
                binding_error.clear();
                return true;
            });
    }

    std::vector<std::unique_ptr<mcp_agent_tool_provider>> mcp_providers;
    for (const auto & provider_request : request.mcp_providers) {
        std::unique_ptr<agent_mcp_tool_client> client;
        if (provider_request.transport == "stdio") {
            if (provider_request.command_line.empty()) {
                error = "MCP provider command line must not be empty";
                return false;
            }
            client = std::make_unique<agent_mcp_stdio_client>(agent_mcp_stdio_client_config{
                provider_request.server_name,
                provider_request.command_line,
                {},
                provider_request.request_timeout_ms,
                provider_request.shutdown_timeout_ms,
            });
        } else {
            if (provider_request.url.empty()) {
                error = "HTTP MCP provider url must not be empty";
                return false;
            }
            client = std::make_unique<agent_mcp_http_client>(agent_mcp_http_client_config{
                provider_request.server_name,
                provider_request.url,
                provider_request.bearer_token,
                provider_request.allowed_tools,
                provider_request.connect_timeout_ms,
                provider_request.request_timeout_ms,
                provider_request.shutdown_timeout_ms,
                provider_request.max_result_bytes,
            });
        }
        auto provider = std::make_unique<mcp_agent_tool_provider>(
            provider_request.server_name,
            *client,
            provider_request.exposed_name_prefix);
        selection.mcp_clients.push_back(std::move(client));
        mcp_providers.push_back(std::move(provider));
    }

    if (native_provider || !mcp_providers.empty()) {
        composite_agent_tool_provider provider;
        if (native_provider) {
            provider.add_provider(*native_provider);
        }
        for (const auto & mcp_provider : mcp_providers) {
            provider.add_provider(*mcp_provider);
        }

        if (!(selection.tool_view = provider.resolve_tools(resolved_tool_context, error))) {
            error = "tool provider resolution failed: " + error;
            return false;
        }
        selection.tooling.tools = selection.tool_view->chat_tools();
        selection.tooling.profile_tools_active = true;
        selection.tooling.tool_view = selection.tool_view.get();
        error.clear();
        return true;
    }

    error.clear();
    return true;
}

bool resolve_agent_cli_tool_selection(
        common_memory_store & store,
        common_plan_store * plan_store,
        agent_resource_store * resource_store,
        std::string * current_plan_id,
        const args & options,
        const common_memory_query & query,
        bool memory_enabled,
        common_agent_cli_tool_selection & selection,
        std::string & error) {
    selection = {};
    std::unique_ptr<agent_embedding_provider> embedding_provider;
    if (memory_enabled) {
        embedding_provider = std::make_unique<cli_agent_embedding_provider>(options);
    }

    const auto repository_root = !options.repository_root.empty()
        ? std::filesystem::weakly_canonical(options.repository_root).string()
        : std::string();
    const auto request = make_agent_cli_tool_selection_request(options, query, repository_root);
    const bool ok = resolve_agent_host_tool_selection(
        store,
        plan_store,
        resource_store,
        current_plan_id,
        options.tool_profile,
        request,
        query,
        embedding_provider.get(),
        selection,
        error);
    if (ok) {
        selection.embedding_provider = std::move(embedding_provider);
    }
    return ok;
}

common_agent_runtime_turn_request make_agent_cli_runtime_turn_request(
        const args & options,
        const common_agent_scope & scope,
        const common_agent_orchestration_config & orchestration_config,
        common_memory_scope memory_scope,
        bool memory_enabled,
        const std::string & fallback_reason,
        agent_embedding_provider * embedding_provider,
        common_agent_request request,
        common_agent_generation_options generation_options,
        std::vector<common_agent_input_resource> input_resources) {
    common_agent_runtime_turn_request turn_request;
    turn_request.request = std::move(request);
    turn_request.request.input_resources = std::move(input_resources);
    turn_request.scope = scope;
    turn_request.inference_options = make_agent_inference_options({
        options.model,
        options.n_predict,
        options.n_gpu_layers,
        true,
    });
    turn_request.inference_options.n_threads = options.n_threads;
    turn_request.inference_options.context_size_tokens = static_cast<size_t>(std::max(0, options.context_size));
    turn_request.policy = make_agent_runtime_policy({
        options.agent_inference_backend,
        options.tool_profile,
        options.memory_learn,
        options.memory_learn_show_candidate,
        options.plan_show_summary,
        options.agent_trace,
        static_cast<size_t>(options.max_tool_rounds),
        options.tool_capabilities,
        options.tool_profiles,
    });
    std::string deliberation_error;
    common_agent_deliberation_policy deliberation_policy;
    if (resolve_common_agent_deliberation_policy(
            options.thinking_mode,
            options.max_reflection_rounds,
            options.max_plan_revisions,
            options.max_research_iterations,
            deliberation_policy,
            deliberation_error)) {
        deliberation_policy.max_tool_rounds = options.max_tool_rounds > 0
            ? static_cast<int>(options.max_tool_rounds)
            : deliberation_policy.max_tool_rounds;
        turn_request.policy.deliberation_policy = deliberation_policy;
    }
    turn_request.runtime_config = make_agent_runtime_config({
        {options.n_predict, options.n_threads, options.generation_trace, static_cast<size_t>(std::max(0, options.context_size)), options.context_budgets},
        options.context_budgets,
        options.memory_learn == "post-turn",
        {options.memory_learn_min_confidence, options.memory_learn_min_reuse},
        [embedding_provider](const std::string & text, std::vector<float> & embedding, std::string & error) {
            return embedding_provider != nullptr &&
                embedding_provider->embed("memory candidate", text, embedding, error);
        },
        options.max_continuations,
    });
    turn_request.orchestration_config = orchestration_config;
    turn_request.generation_options = generation_options;
    turn_request.generation_options.generation_trace = options.generation_trace;
    if (turn_request.generation_options.n_threads < 1) {
        turn_request.generation_options.n_threads = options.n_threads;
    }
    if (!turn_request.generation_options.t_max_predict_ms && options.inference_step_timeout_ms > 0) {
        turn_request.generation_options.t_max_predict_ms = options.inference_step_timeout_ms;
    }
    if (!turn_request.generation_options.t_max_prompt_ms && options.inference_step_timeout_ms > 0) {
        turn_request.generation_options.t_max_prompt_ms = options.inference_step_timeout_ms;
    }
    turn_request.memory_scope = memory_scope;
    turn_request.memory_enabled = memory_enabled;
    turn_request.fallback_reason = fallback_reason;
    return turn_request;
}

common_agent_runtime_host_inputs make_agent_cli_runtime_host_chat_inputs(
        common_memory_store & store,
        args & options,
        const std::vector<common_chat_msg> & messages,
        common_memory_scope memory_scope,
        const std::vector<common_memory_hit> & memories,
        bool memory_enabled,
        const std::string & fallback_reason,
        const common_agent_runtime_tooling & tooling,
        agent_embedding_provider * embedding_provider,
        common_agent_runtime_host_post_run post_run,
        std::vector<common_agent_input_resource> input_resources) {
    common_agent_scope runtime_scope = common_cli_make_agent_scope_with_matching_plan_scope(options);
    common_agent_request request;
    request.messages = messages;
    common_agent_generation_options generation_options;
    generation_options.n_threads = options.n_threads;
    auto turn_request = make_agent_cli_runtime_turn_request(
        options,
        runtime_scope,
        make_agent_orchestration_config({
            options.prompt,
            options.agent_plan,
            options.agent_blueprint,
            options.agent_bootstrap,
            options.agent_import,
            options.agent_export,
        }),
        memory_scope,
        memory_enabled,
        fallback_reason,
        embedding_provider,
        std::move(request),
        generation_options,
        std::move(input_resources));
    common_agent_runtime_host_build_context build_context{
        store,
        nullptr,
        std::move(turn_request),
        nullptr,
        nullptr,
        memories,
        tooling,
    };
    auto inputs = make_agent_runtime_host_chat_inputs(build_context);
    inputs.reset_session_on_completion = true;
    inputs.post_run = std::move(post_run);
    return inputs;
}

common_agent_runtime_host_inputs make_agent_cli_runtime_host_agent_inputs(
        common_memory_store & store,
        common_plan_store & plan_store,
        args & options,
        common_agent_scope & scope,
        std::string & current_plan_id,
        const std::vector<common_blueprint_candidate> & installed_blueprint_candidates,
        const common_agent_orchestration_config & orchestration_config,
        common_memory_scope memory_scope,
        const std::vector<common_memory_hit> & memories,
        bool memory_enabled,
        const std::string & fallback_reason,
        const common_agent_runtime_tooling & tooling,
        agent_embedding_provider * embedding_provider,
        common_agent_runtime_host_post_run post_run,
        std::vector<common_agent_input_resource> input_resources) {
    auto turn_request = make_agent_cli_runtime_turn_request(
        options,
        scope,
        orchestration_config,
        memory_scope,
        memory_enabled,
        fallback_reason,
        embedding_provider);
    turn_request.request.input_resources = std::move(input_resources);
    common_agent_runtime_host_build_context build_context{
        store,
        &plan_store,
        std::move(turn_request),
        &current_plan_id,
        &installed_blueprint_candidates,
        memories,
        tooling,
    };
    auto inputs = make_agent_runtime_host_agent_inputs(build_context, orchestration_config);
    inputs.reset_session_on_completion = true;
    inputs.post_run = std::move(post_run);
    return inputs;
}

int finish_agent_cli_runtime_result(const common_agent_result & result) {
    std::printf("%s\n", result.response.c_str());
    std::fprintf(stderr, "decoded %d tokens\n", result.total_decoded_tokens);
    return 0;
}
