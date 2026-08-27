#include "agent-cli-host-adapter.h"
#include "../openapi/agent-openapi-http.h"

#include "tools/agent/cli/agent-cli-memory-tools.h"
#include "tools/agent/resource/assembly/agent-resource-processor-factory.h"
#include "tools/agent/host/agent-sandbox-assembly.h"
#include "tools/agent/mcp/agent-mcp-client-factory.h"
#include "tools/agent/resource/processors/agent-pdf-text-processor.h"

#include "../runtime/agent-plan-orchestration.h"
#include "../runtime/agent-runtime-assembly.h"
#include "../runtime/agent-runtime-execution.h"
#include "agent/sandbox/sandbox-docker-runtime.h"
#include "agent/sandbox/sandbox-kubernetes-runtime.h"
#include "agent/sandbox/sandbox-lxc-runtime.h"
#include "agent/sandbox/sandbox-local-runtime.h"
#include "agent/sandbox/sandbox-runtime.h"
#include "../diagnostics/agent-clangd-provider.h"
#include "../diagnostics/agent-native-crash.h"
#include "../data/agent-data-store-factory.h"
#include "../data/agent-dataset-importer.h"
#include "../tooling/agent-sandbox-helper.h"
#include "tools/agent/cli/agent-cli-scope.h"

#include <algorithm>
#include <fstream>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <memory>
#include <cstdlib>
#include <nlohmann/json.hpp>

using json = nlohmann::ordered_json;

namespace {

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
        if (request.sandbox.backend != "docker" && request.sandbox.backend != "kubernetes" &&
                request.sandbox.backend != "lxc") {
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
        if (request.diagnostics.native_crash_backend != "none") {
            const auto native_crash_limits = agent_native_crash_limits{
                request.diagnostics.native_crash_timeout_ms,
                request.diagnostics.native_crash_max_frames,
                request.diagnostics.native_crash_max_threads,
                request.diagnostics.native_crash_max_output_bytes,
            };
            bindings.diagnostics_native_crash = [
                    backend = request.diagnostics.native_crash_backend,
                    gdb = request.diagnostics.gdb_executable,
                    cdb = request.diagnostics.cdb_executable,
                    repository_root = request.repository_root,
                    native_crash_limits](const std::string & input) {
                return agent_execute_native_crash(input, backend, gdb, cdb, repository_root, native_crash_limits);
            };
        }
        if (embedding_provider != nullptr) {
            bindings.embed_memory_query = [provider = embedding_provider](const std::string & text, std::vector<float> & embedding, std::string & embedding_error) {
                return provider != nullptr &&
                    provider->embed("tool query", text, embedding, embedding_error);
            };
        }

        const auto needs_sandbox_assembly = [&request]() {
            return request.sandbox.backend == "docker" || request.sandbox.backend == "kubernetes" ||
                request.sandbox.backend == "lxc" ||
                request.resource_processor_policies.find("pdf.page_image") != request.resource_processor_policies.end() ||
                request.resource_processor_policies.find("ocr.tesseract") != request.resource_processor_policies.end() ||
                request.resource_processor_policies.find("docx.text") != request.resource_processor_policies.end() ||
                request.resource_processor_policies.find("odt.text") != request.resource_processor_policies.end() ||
                request.resource_processor_policies.find("html.text") != request.resource_processor_policies.end() ||
                request.resource_processor_policies.find("xlsx.workbook") != request.resource_processor_policies.end();
        };
        std::shared_ptr<common_agent_sandbox_docker_runtime> docker_runtime;
        std::shared_ptr<common_agent_sandbox_kubernetes_runtime> kubernetes_runtime;
        std::shared_ptr<common_agent_sandbox_lxc_runtime> lxc_runtime;
        std::shared_ptr<common_agent_workspace_manager> workspace_manager;
        if (needs_sandbox_assembly()) {
            auto sandbox_assembly = make_agent_host_sandbox_assembly({
                request.sandbox,
                request.tool_context.scope,
                resource_store,
            });
            docker_runtime = std::move(sandbox_assembly.docker_runtime);
            kubernetes_runtime = std::move(sandbox_assembly.kubernetes_runtime);
            lxc_runtime = std::move(sandbox_assembly.lxc_runtime);
            workspace_manager = std::move(sandbox_assembly.workspace_manager);
            bindings.sandbox_execute = std::move(sandbox_assembly.execute);
        } else {
            bindings.sandbox_execute = make_agent_host_sandbox_assembly({
                request.sandbox,
                request.tool_context.scope,
                resource_store,
            }).execute;
        }

        if (needs_sandbox_assembly()) {
            agent_resource_processing_assembly_request assembly;
            assembly.policies = request.resource_processor_policies;
            assembly.sandbox_classes = request.sandbox.classes;
            assembly.sandbox_defaults = request.sandbox.defaults;
            assembly.docker_runtime = docker_runtime;
            assembly.kubernetes_runtime = kubernetes_runtime;
            assembly.lxc_runtime = lxc_runtime;
            assembly.local_runtime = std::make_shared<common_agent_sandbox_local_runtime>();
            assembly.workspace_manager = workspace_manager;
            assembly.resource_store = resource_store;
            assembly.sandbox_backend = request.sandbox.backend;
            bindings.resource_processing_provider_factory =
                make_agent_resource_processing_provider_factory(std::move(assembly));
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
        const auto client_request = agent_mcp_client_factory_request{
            provider_request.server_name,
            provider_request.transport,
            provider_request.command_line,
            provider_request.url,
            provider_request.bearer_token,
            provider_request.allowed_tools,
            provider_request.connect_timeout_ms,
            provider_request.request_timeout_ms,
            provider_request.shutdown_timeout_ms,
            provider_request.max_result_bytes,
        };
        auto client = make_agent_mcp_client(client_request, error);
        if (!client) return false;
        auto provider = std::make_unique<mcp_agent_tool_provider>(
            provider_request.server_name,
            *client,
            provider_request.exposed_name_prefix);
        selection.mcp_clients.push_back(std::move(client));
        mcp_providers.push_back(std::move(provider));
    }

    for (const auto & provider_config : request.openapi_providers) {
        if (!provider_config.enabled) continue;
        const std::filesystem::path configured_spec_path(provider_config.spec_path);
        const std::filesystem::path spec_path = configured_spec_path.is_absolute() ||
                provider_config.source_directory.empty()
            ? configured_spec_path
            : std::filesystem::path(provider_config.source_directory) / configured_spec_path;
        std::ifstream spec_file(spec_path);
        if (!spec_file) {
            if (provider_config.required) {
                error = "required OpenAPI spec could not be opened: " + spec_path.string();
                return false;
            }
            continue;
        }
        nlohmann::json spec;
        std::string provider_error;
        try { spec_file >> spec; }
        catch (const std::exception & exception) {
            provider_error = "OpenAPI spec could not be parsed: " + std::string(exception.what());
        }
        if (!provider_error.empty()) {
            if (provider_config.required) { error = provider_error; return false; }
            continue;
        }
        agent_openapi_catalog catalog;
        if (!build_agent_openapi_catalog(spec, provider_config, catalog, provider_error)) {
            if (provider_config.required) {
                error = "OpenAPI provider '" + provider_config.id + "' failed: " + provider_error;
                return false;
            }
            continue;
        }
        selection.openapi_providers.push_back(std::make_unique<agent_openapi_tool_provider>(
            std::move(catalog), make_agent_openapi_http_executor(provider_config)));
    }

    if (native_provider || !mcp_providers.empty() || !selection.openapi_providers.empty()) {
        composite_agent_tool_provider provider;
        if (native_provider) {
            provider.add_provider(*native_provider);
        }
        for (const auto & mcp_provider : mcp_providers) {
            provider.add_provider(*mcp_provider);
        }
        for (const auto & openapi_provider : selection.openapi_providers) {
            provider.add_provider(*openapi_provider);
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
    turn_request.request.require_tool_execution = options.require_tool_execution;
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
    turn_request.runtime_config.generation_config.enable_tool_family_routing =
        options.agent_plan == "auto";
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
