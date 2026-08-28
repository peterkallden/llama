#include "tools/agent/daemon/agent-daemon-adapter.h"
#include "tools/agent/daemon/agent-daemon-service.h"
#include "tools/agent/host/agent-host-config.h"
#include "tools/agent/mcp/agent-mcp-auth.h"

#include <nlohmann/json.hpp>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

using json = nlohmann::ordered_json;

namespace {

bool has_tool(const std::vector<common_chat_tool> & tools, const std::string & name) {
    for (const auto & tool : tools) {
        if (tool.name == name) {
            return true;
        }
    }
    return false;
}

std::filesystem::path get_fake_server_path(const char * argv0) {
    std::filesystem::path argv_path = argv0 != nullptr ? std::filesystem::path(argv0) : std::filesystem::path();
    if (argv_path.has_parent_path()) {
        argv_path = std::filesystem::absolute(argv_path);
    } else {
        argv_path = std::filesystem::current_path() / argv_path;
    }
#ifdef _WIN32
    return argv_path.parent_path() / "llama-agent-mcp-stdio-fake-server.exe";
#else
    return argv_path.parent_path() / "llama-agent-mcp-stdio-fake-server";
#endif
}

} // namespace

int main(int argc, char ** argv) {
    const auto server_path = get_fake_server_path(argc > 0 ? argv[0] : nullptr);
    if (!std::filesystem::exists(server_path)) {
        std::fprintf(stderr, "fake MCP stdio server not found: %s\n", server_path.string().c_str());
        return 1;
    }

    const auto config_root = std::filesystem::temp_directory_path() / "llama-agent-daemon-mcp-config-smoke";
    std::filesystem::create_directories(config_root);
    const auto config_path = config_root / "daemon-config.json";
    {
        std::ofstream out(config_path);
        out << json{
            {"model", {
                {"backend", "server-context"},
                {"path", "fake.gguf"},
            }},
            {"runtime", {
                {"thinking_mode", "deliberate"},
                {"context_size", 4096},
                {"n_threads", 4},
                {"max_reflection_rounds", 3},
                {"max_plan_revisions", 2},
                {"context_budgets", {
                    {"plan_chars", 4096},
                    {"tool_observation_chars", 8192},
                    {"resource_chunk_max_bytes", 3072},
                    {"resource_chunk_overlap_bytes", 192},
                    {"working_state", {
                        {"max_total_chars", 6000},
                        {"max_value_chars", 768},
                        {"max_completed_steps", 12},
                        {"max_remaining_steps", 10},
                        {"max_constraints", 8},
                        {"max_open_questions", 6},
                        {"max_resource_refs", 10},
                        {"max_chunk_status", 12},
                        {"max_tool_results", 8},
                    }},
                }},
            }},
            {"tools", {
                {"profile", "research"},
                {"providers", json::array({
                    json{
                        {"type", "mcp"},
                        {"id", "github"},
                        {"enabled", true},
                        {"required", true},
                        {"transport", "stdio"},
                        {"command", json::array({server_path.string()})},
                        {"prefix", "github"},
                        {"server_name", "github"},
                    },
                    json{
                        {"type", "mcp"},
                        {"id", "github_alt"},
                        {"enabled", true},
                        {"transport", "stdio"},
                        {"command", json::array({server_path.string()})},
                        {"prefix", "github_alt"},
                        {"server_name", "github-alt"},
                    },
                    json{
                        {"type", "openapi"},
                        {"id", "sales-api"},
                        {"spec_path", "sales.openapi.json"},
                        {"base_url", "https://api.example.test"},
                        {"policy", {
                            {"access", "read_only"},
                            {"exposure", "auto"},
                            {"operations", {
                                {"searchSales", {{"access", "read"}}},
                            }},
                        }},
                        {"auth", {
                            {"type", "bearer"},
                            {"scheme", "bearerAuth"},
                            {"token_env", "SALES_API_TOKEN"},
                        }},
                        {"limits", {
                            {"connect_timeout_ms", 4000},
                            {"request_timeout_ms", 15000},
                            {"max_result_bytes", 1048576},
                        }},
                    },
                })},
            }},
            {"limits", {
                {"queue_capacity", 5},
                {"worker_count", 2},
                {"inference_max_active", 1},
                {"max_tool_rounds", 2},
                {"max_continuations", 3},
            }},
        }.dump(2);
    }

    daemon_options options;
    char program[] = "llama-agent-daemon";
    char config_flag[] = "--config";
    std::string config_path_string = config_path.string();
    std::string error;
    std::vector<char> config_path_buffer(config_path_string.begin(), config_path_string.end());
    config_path_buffer.push_back('\0');
    char * parse_argv[] = {program, config_flag, config_path_buffer.data()};
    if (!parse_agent_daemon_args(3, parse_argv, options)) {
        std::fprintf(stderr, "daemon config parse failed\n");
        return 1;
    }

    daemon_options cli_options;
    char cli_program[] = "llama-agent-daemon";
    char model_flag[] = "--model";
    char model_value[] = "fake.gguf";
    char thinking_flag[] = "--thinking-mode";
    char thinking_value[] = "deliberate";
    char reflection_limit_flag[] = "--max-reflection-rounds";
    char reflection_limit_value[] = "3";
    char revision_limit_flag[] = "--max-plan-revisions";
    char revision_limit_value[] = "2";
    char research_limit_flag[] = "--max-research-iterations";
    char research_limit_value[] = "1";
    char threads_flag[] = "--threads";
    char threads_value[] = "1";
    char * cli_parse_argv[] = {
        cli_program, model_flag, model_value, thinking_flag, thinking_value,
        reflection_limit_flag, reflection_limit_value,
        revision_limit_flag, revision_limit_value,
        research_limit_flag, research_limit_value,
        threads_flag, threads_value,
    };
    if (!parse_agent_daemon_args(
            static_cast<int>(sizeof(cli_parse_argv) / sizeof(cli_parse_argv[0])),
            cli_parse_argv, cli_options) ||
            cli_options.thinking_mode != "deliberate" ||
            cli_options.max_reflection_rounds != 3 ||
            cli_options.max_plan_revisions != 2 ||
            cli_options.max_research_iterations != 1 ||
            cli_options.n_threads != 1) {
        std::fprintf(stderr, "daemon thinking policy CLI parse failed\n");
        return 1;
    }

    agent_host_config loaded_config;
    if (!load_agent_host_config(config_path.string(), loaded_config, error)) {
        std::fprintf(stderr, "explicit host config load failed: %s\n", error.c_str());
        return 1;
    }
    if (loaded_config.schema_version != 1) {
        std::fprintf(stderr, "host config schema_version mismatch\n");
        return 1;
    }
    if (loaded_config.worker_count != 2) {
        std::fprintf(stderr, "host config worker_count mismatch\n");
        return 1;
    }
    if (loaded_config.mcp_providers.empty() ||
            !loaded_config.mcp_providers.front().required) {
        std::fprintf(stderr, "host config MCP required policy mismatch\n");
        return 1;
    }
    if (loaded_config.openapi_providers.size() != 1 ||
            loaded_config.openapi_providers.front().id != "sales-api" ||
            loaded_config.openapi_providers.front().access != "read_only" ||
            loaded_config.openapi_providers.front().exposure != "auto" ||
            loaded_config.openapi_providers.front().operations.size() != 1 ||
            loaded_config.openapi_providers.front().auth.scheme != "bearerAuth" ||
            loaded_config.openapi_providers.front().auth.token_env != "SALES_API_TOKEN") {
        std::fprintf(stderr, "host config OpenAPI provider contract failed\n");
        return 1;
    }
    if (loaded_config.runtime_context_size != 4096 || loaded_config.n_threads != 4) {
        std::fprintf(stderr, "host config n_threads mismatch\n");
        return 1;
    }
    daemon_options applied_daemon_options;
    apply_agent_host_config_to_daemon_options(loaded_config, applied_daemon_options);
    if (applied_daemon_options.context_size != 4096 || applied_daemon_options.n_threads != 4) {
        std::fprintf(stderr, "host config n_threads was not applied to daemon options\n");
        return 1;
    }
    if (loaded_config.context_budgets.plan_chars != 4096 ||
            loaded_config.context_budgets.tool_observation_chars != 8192 ||
            loaded_config.context_budgets.resource_chunk_max_bytes != 3072 ||
            loaded_config.context_budgets.resource_chunk_overlap_bytes != 192 ||
            loaded_config.context_budgets.working_state.max_total_chars != 6000 ||
            loaded_config.context_budgets.working_state.max_value_chars != 768 ||
            loaded_config.context_budgets.working_state.max_completed_steps != 12 ||
            loaded_config.context_budgets.working_state.max_remaining_steps != 10 ||
            loaded_config.context_budgets.working_state.max_constraints != 8 ||
            loaded_config.context_budgets.working_state.max_open_questions != 6 ||
            loaded_config.context_budgets.working_state.max_resource_refs != 10 ||
            loaded_config.context_budgets.working_state.max_chunk_status != 12 ||
            loaded_config.context_budgets.working_state.max_tool_results != 8 ||
            loaded_config.max_continuations != 3) {
        std::fprintf(stderr, "host context budgets mismatch\n");
        return 1;
    }
    if (loaded_config.inference_max_active != 1) {
        std::fprintf(stderr, "host config inference_max_active mismatch\n");
        return 1;
    }
    if (loaded_config.thinking_mode != "deliberate" ||
            loaded_config.max_reflection_rounds != 3 ||
            loaded_config.max_plan_revisions != 2) {
        std::fprintf(stderr, "deliberation config values were not loaded\n");
        return 1;
    }
    if (!validate_agent_host_config(loaded_config, error)) {
        std::fprintf(stderr, "host config validation failed: %s\n", error.c_str());
        return 1;
    }
    args stdio_options;
    apply_agent_host_config_to_args(loaded_config, stdio_options);
    if (stdio_options.thinking_mode != "deliberate" ||
            stdio_options.n_threads != 4 ||
            stdio_options.max_reflection_rounds != 3 ||
            stdio_options.max_plan_revisions != 2 ||
            stdio_options.max_continuations != 3) {
        std::fprintf(stderr, "deliberation config was not applied to stdio args\n");
        return 1;
    }
    agent_host_config invalid_thinking_config = loaded_config;
    invalid_thinking_config.thinking_mode = "unknown";
    if (validate_agent_host_config(invalid_thinking_config, error) ||
            error.find("runtime.thinking_mode") == std::string::npos) {
        std::fprintf(stderr, "invalid thinking mode was accepted\n");
        return 1;
    }
    agent_host_config invalid_limit_config = loaded_config;
    invalid_limit_config.max_reflection_rounds = -1;
    if (validate_agent_host_config(invalid_limit_config, error) ||
            error.find("runtime deliberation limits") == std::string::npos) {
        std::fprintf(stderr, "negative deliberation limit was accepted\n");
        return 1;
    }
    agent_host_config invalid_continuation_config = loaded_config;
    invalid_continuation_config.max_continuations = 17;
    if (validate_agent_host_config(invalid_continuation_config, error) ||
            error.find("limits.max_continuations") == std::string::npos) {
        std::fprintf(stderr, "unbounded continuation limit was accepted\n");
        return 1;
    }
    agent_host_config invalid_sqlite_path_config = loaded_config;
    invalid_sqlite_path_config.memory_backend = "sqlite";
    invalid_sqlite_path_config.memory_db.clear();
    if (validate_agent_host_config(invalid_sqlite_path_config, error) ||
            error.find("stores.memory.path") == std::string::npos) {
        std::fprintf(stderr, "SQLite memory backend without a path was accepted\n");
        return 1;
    }
    invalid_sqlite_path_config = loaded_config;
    invalid_sqlite_path_config.plan_backend = "sqlite";
    invalid_sqlite_path_config.plan_db.clear();
    if (validate_agent_host_config(invalid_sqlite_path_config, error) ||
            error.find("stores.plan.path") == std::string::npos) {
        std::fprintf(stderr, "SQLite plan backend without a path was accepted\n");
        return 1;
    }
    agent_host_config invalid_openapi_config = loaded_config;
    invalid_openapi_config.openapi_providers.front().auth.type = "bearer";
    invalid_openapi_config.openapi_providers.front().auth.token_env.clear();
    if (validate_agent_host_config(invalid_openapi_config, error) ||
            error.find("OpenAPI bearer auth requires token_env") == std::string::npos) {
        std::fprintf(stderr, "OpenAPI bearer provider without token_env was accepted\n");
        return 1;
    }
    const json roundtrip = agent_host_config_to_json(loaded_config);
    if (!roundtrip.is_object() ||
            roundtrip.value("schema_version", 0) != 1 ||
            !roundtrip.contains("tools") ||
            !roundtrip["tools"].is_object()) {
        std::fprintf(stderr, "host config roundtrip serialization mismatch\n");
        return 1;
    }
    if (roundtrip["runtime"]["context_budgets"]["plan_chars"] != 4096 ||
            roundtrip["runtime"]["context_budgets"]["tool_observation_chars"] != 8192 ||
            roundtrip["runtime"]["context_budgets"]["resource_chunk_max_bytes"] != 3072 ||
            roundtrip["runtime"]["context_budgets"]["resource_chunk_overlap_bytes"] != 192 ||
            roundtrip["runtime"]["context_budgets"]["working_state"]["max_total_chars"] != 6000 ||
            roundtrip["runtime"]["context_budgets"]["working_state"]["max_tool_results"] != 8 ||
            roundtrip["limits"]["max_continuations"] != 3) {
        std::fprintf(stderr, "host context budgets roundtrip mismatch\n");
        return 1;
    }
    if (!roundtrip["tools"]["providers"].is_array() ||
            roundtrip["tools"]["providers"].size() != 3 ||
            roundtrip["tools"]["providers"][2]["type"] != "openapi" ||
            roundtrip["tools"]["providers"][2]["policy"]["access"] != "read_only" ||
            roundtrip["tools"]["providers"][2]["auth"]["scheme"] != "bearerAuth" ||
            roundtrip["tools"]["providers"][2]["auth"]["token_env"] != "SALES_API_TOKEN") {
        std::fprintf(stderr, "host config OpenAPI roundtrip mismatch\n");
        return 1;
    }

    // The provider directory is an additive alternative to the inline array.
    // Verify that both provider families are loaded, sorted deterministically,
    // and not duplicated when the effective config is serialized.
    const auto fragment_directory = config_root / "providers.d";
    std::filesystem::create_directories(fragment_directory);
    {
        std::ofstream out(fragment_directory / "20-openapi.json");
        out << json{
            {"type", "openapi"},
            {"id", "fragment-api"},
            {"spec_path", "fragment.openapi.json"},
            {"base_url", "https://api.example.test"},
        }.dump(2);
    }
    {
        std::ofstream out(fragment_directory / "10-mcp.json");
        out << json{
            {"type", "mcp"},
            {"id", "fragment-mcp"},
            {"transport", "stdio"},
            {"command", json::array({server_path.string()})},
        }.dump(2);
    }
    const auto fragmented_config_path = config_root / "fragmented-config.json";
    {
        std::ofstream out(fragmented_config_path);
        out << json{
            {"schema_version", 1},
            {"model", {{"path", "fake.gguf"}}},
            {"tools", {{"include_dir", "providers.d"}}},
        }.dump(2);
    }
    agent_host_config fragmented_config;
    if (!load_agent_host_config(fragmented_config_path.string(), fragmented_config, error) ||
            fragmented_config.tools_include_dir != "providers.d" ||
            fragmented_config.mcp_providers.size() != 1 ||
            fragmented_config.openapi_providers.size() != 1 ||
            fragmented_config.mcp_providers.front().id != "fragment-mcp" ||
            fragmented_config.openapi_providers.front().id != "fragment-api" ||
            fragmented_config.openapi_providers.front().source_directory != config_root.string()) {
        std::fprintf(stderr, "provider directory configuration failed: %s\n", error.c_str());
        return 1;
    }
    const json fragmented_roundtrip = agent_host_config_to_json(fragmented_config);
    if (fragmented_roundtrip["tools"]["include_dir"] != "providers.d" ||
            !fragmented_roundtrip["tools"]["providers"].is_array() ||
            !fragmented_roundtrip["tools"]["providers"].empty()) {
        std::fprintf(stderr, "provider directory roundtrip duplicated effective providers\n");
        return 1;
    }
    agent_host_config fragmented_reloaded;
    if (!load_agent_host_config(fragmented_config_path.string(), fragmented_reloaded, error) ||
            fragmented_reloaded.mcp_providers.size() != 1 ||
            fragmented_reloaded.openapi_providers.size() != 1) {
        std::fprintf(stderr, "provider directory reload failed: %s\n", error.c_str());
        return 1;
    }

    agent_host_config legacy_config;
    const json legacy_config_json = {
        {"schema_version", 1},
        {"runtime", {{"context_size", 2048}}},
    };
    if (!parse_agent_host_config_json(legacy_config_json, legacy_config, error) ||
            legacy_config.max_continuations != 2 ||
            legacy_config.context_budgets.working_state.max_total_chars != 8192 ||
            legacy_config.context_budgets.working_state.max_tool_results != 32) {
        std::fprintf(stderr, "legacy host config defaults were not preserved\n");
        return 1;
    }

    agent_host_config capability_config;
    const json capability_config_json = {
        {"schema_version", 1},
        {"stores", {
            {"data", {
                {"backend", "cozo"},
                {"path", "C:/agent-data/structured.cozo"},
            }},
        }},
        {"tools", {
            {"profile", "local-developer"},
            {"capabilities", {
                {"workspace.read", json::array({"repository.list", "repository.read"})},
                {"repository.inspect", json::array({"repository.search", "repository.diff"})},
            }},
            {"profiles", {
                {"local-developer", {
                    {"include_capabilities", json::array({"workspace.read", "repository.inspect"})},
                    {"exclude_capabilities", json::array()},
                    {"allow_network", false},
                    {"allow_policy_gated_writes", true},
                }},
            }},
        }},
        {"sandbox", {
            {"backend", "docker"},
            {"docker", {
                {"executable", "docker"},
                {"default_image", "llama-agent-dev:latest"},
            }},
            {"kubernetes", {
                {"executable", "kubectl"},
                {"namespace", "llama-agent-jobs"},
                {"service_account", "llama-agent-runner"},
                {"runtime_class", "standard"},
                {"cleanup", true},
            }},
            {"workspace", {
                {"root", "C:/agent-workspaces"},
                {"artifact_root", "C:/agent-artifacts"},
                {"operation_mode", "ephemeral"},
                {"project_mode", "persistent"},
            }},
            {"defaults", {
                {"image", "llama-agent-default:latest"},
                {"timeout_ms", 300000},
                {"cpu_count", 2},
                {"max_output_bytes", 131072},
                {"network", "none"},
                {"filesystem", "readonly"},
                {"allow_artifacts", true},
            }},
            {"classes", {
                {"developer-build", {
                    {"image", "llama-agent-dev:latest"},
                    {"timeout_ms", 120000},
                    {"memory_bytes", 8589934592ull},
                    {"cpu_count", 4},
                    {"network", "none"},
                    {"filesystem", "workspace_write"},
                    {"allow_artifacts", true},
                }},
            }},
        }},
    };
    if (!parse_agent_host_config_json(capability_config_json, capability_config, error) ||
            capability_config.tool_profile != "local-developer" ||
            capability_config.data_backend != "cozo" ||
            capability_config.data_db != "C:/agent-data/structured.cozo" ||
            capability_config.tool_capabilities.size() != 2 ||
            capability_config.tool_profiles.size() != 1 ||
            capability_config.tool_profiles.at("local-developer").include_capabilities.size() != 2 ||
            capability_config.sandbox.backend != "docker" ||
            capability_config.sandbox.docker_executable != "docker" ||
            capability_config.sandbox.docker_default_image != "llama-agent-dev:latest" ||
            capability_config.sandbox.kubernetes_namespace != "llama-agent-jobs" ||
            !capability_config.sandbox.kubernetes_storage_class.empty() ||
            capability_config.sandbox.kubernetes_insecure_skip_tls_verify ||
            capability_config.sandbox.kubernetes_workspace_storage_size != "4Gi" ||
            capability_config.sandbox.kubernetes_artifact_storage_size != "1Gi" ||
            capability_config.sandbox.kubernetes_staging_image != "alpine:3.20" ||
            capability_config.sandbox.kubernetes_pvc_retention != "project" ||
            capability_config.sandbox.kubernetes_staging_timeout_ms != 120000 ||
            capability_config.sandbox.kubernetes_service_account != "llama-agent-runner" ||
            capability_config.sandbox.classes.at("developer-build").image != "llama-agent-dev:latest" ||
            capability_config.sandbox.classes.at("developer-build").limits.max_output_bytes != 131072 ||
            capability_config.sandbox.workspace.workspace_root != "C:/agent-workspaces" ||
            capability_config.sandbox.workspace.operation_mode != "ephemeral" ||
            capability_config.sandbox.workspace.project_mode != "persistent" ||
            capability_config.sandbox.classes.at("developer-build").filesystem != common_agent_sandbox_filesystem_scope::workspace_write) {
        std::fprintf(stderr, "capability/profile host config was not parsed\n");
        return 1;
    }
    if (!capability_config.tool_profiles.at("local-developer").allow_network.has_value() ||
            capability_config.tool_profiles.at("local-developer").allow_network.value() ||
            !capability_config.tool_profiles.at("local-developer").allow_policy_gated_writes.value_or(false)) {
        std::fprintf(stderr, "profile policy fields were not parsed\n");
        return 1;
    }
    const json capability_roundtrip = agent_host_config_to_json(capability_config);
    if (capability_roundtrip["tools"]["capabilities"]["workspace.read"].size() != 2 ||
            capability_roundtrip["tools"]["profiles"]["local-developer"]["include_capabilities"].size() != 2 ||
            capability_roundtrip["tools"]["profiles"]["local-developer"]["allow_network"].get<bool>() ||
            capability_roundtrip["stores"]["data"]["backend"] != "cozo" ||
            capability_roundtrip["stores"]["data"]["path"] != "C:/agent-data/structured.cozo" ||
            capability_roundtrip["sandbox"]["classes"]["developer-build"]["filesystem"] != "workspace_write" ||
            capability_roundtrip["sandbox"]["classes"]["developer-build"]["image"] != "llama-agent-dev:latest" ||
            capability_roundtrip["sandbox"]["defaults"]["image"] != "llama-agent-default:latest" ||
            capability_roundtrip["sandbox"]["kubernetes"]["namespace"] != "llama-agent-jobs" ||
            capability_roundtrip["sandbox"]["docker"]["default_image"] != "llama-agent-dev:latest" ||
            capability_roundtrip["sandbox"]["workspace"]["operation_mode"] != "ephemeral") {
        std::fprintf(stderr, "capability/profile host config roundtrip failed\n");
        return 1;
    }
    if (!validate_agent_host_config(capability_config, error)) {
        std::fprintf(stderr, "sandbox host config was rejected: %s\n", error.c_str());
        return 1;
    }
    agent_host_config invalid_capability_config = capability_config;
    invalid_capability_config.tool_profiles.at("local-developer").include_capabilities.push_back("missing.capability");
    if (validate_agent_host_config(invalid_capability_config, error) ||
            error.find("unknown included capability") == std::string::npos) {
        std::fprintf(stderr, "unknown profile capability was accepted\n");
        return 1;
    }
    common_tool_catalog configured_catalog;
    common_tool_bootstrap_result configured_bootstrap;
    if (!configured_catalog.bootstrap(
            capability_config.tool_profile,
            configured_bootstrap,
            error,
            capability_config.tool_capabilities,
            capability_config.tool_profiles)) {
        std::fprintf(stderr, "configured capability profile bootstrap failed: %s\n", error.c_str());
        return 1;
    }
    const auto configured_tools = configured_catalog.load_profile(
        capability_config.tool_profile, error);
    if (configured_tools.size() != 4 ||
            error.size() != 0 ||
            configured_tools[0].name != "repository.list" ||
            configured_tools[3].name != "repository.diff") {
        std::fprintf(stderr, "configured capability profile did not resolve its tools\n");
        return 1;
    }
    common_tool_profile_snapshot configured_snapshot;
    if (!configured_catalog.resolve_profile(
            capability_config.tool_profile, configured_snapshot, error) ||
            configured_snapshot.id != "local-developer" ||
            configured_snapshot.tools.size() != 4 ||
            configured_snapshot.allow_network.value_or(true) ||
            !configured_snapshot.allow_policy_gated_writes.value_or(false)) {
        std::fprintf(stderr, "configured immutable profile snapshot was not resolved\n");
        return 1;
    }

    agent_host_config remote_config;
    const json remote_config_json = {
        {"schema_version", 1},
        {"tools", { {"providers", json::array({ json{
            {"type", "mcp"},
            {"id", "remote-github"},
            {"enabled", true},
            {"transport", "streamable_http"},
            {"url", "https://mcp.example.test/mcp"},
            {"auth", {
                {"type", "bearer"},
                {"token_env", "REMOTE_GITHUB_MCP_TOKEN"},
            }},
            {"allowed_tools", json::array({"search_issues"})},
            {"connect_timeout_ms", 4000},
            {"request_timeout_ms", 15000},
            {"shutdown_timeout_ms", 2000},
            {"max_result_bytes", 1048576},
        }})}}},
    };
    if (!parse_agent_host_config_json(remote_config_json, remote_config, error) ||
            remote_config.mcp_providers.size() != 1 ||
            remote_config.mcp_providers[0].url != "https://mcp.example.test/mcp" ||
            remote_config.mcp_providers[0].auth.token_env != "REMOTE_GITHUB_MCP_TOKEN" ||
            remote_config.mcp_providers[0].allowed_tools.size() != 1 ||
            remote_config.mcp_providers[0].max_result_bytes != 1048576) {
        std::fprintf(stderr, "remote MCP provider config contract failed: %s\n", error.c_str());
        return 1;
    }

    agent_host_config inbound_config;
    const json inbound_config_json = {
        {"mcp", {{"inbound", {
            {"enabled", true},
            {"listen", "127.0.0.1"},
            {"port", 8081},
            {"path", "/mcp"},
            {"tokens", json::array({
                json{
                    {"id", "readonly"},
                    {"token_env", "LLAMA_AGENT_READ_TOKEN"},
                    {"audience", "llama-agent"},
                    {"namespace", "namespace-a"},
                    {"project", "project-a"},
                    {"tool_profile", "minimal"},
                    {"allowed_tools", json::array({"math.calculate"})},
                    {"allow_writes", false},
                },
                json{
                    {"id", "admin"},
                    {"token_env", "LLAMA_AGENT_ADMIN_TOKEN"},
                    {"audience", "llama-agent"},
                    {"namespace", "namespace-a"},
                    {"project", "project-a"},
                    {"tool_profile", "research"},
                    {"allow_writes", true},
                    {"allow_admin", true},
                },
            })},
        }}}},
    };
    if (!parse_agent_host_config_json(inbound_config_json, inbound_config, error) ||
            !inbound_config.inbound_mcp_enabled ||
            inbound_config.inbound_mcp_tokens.size() != 2 ||
            inbound_config.inbound_mcp_tokens[0].allowed_tools.size() != 1 ||
            inbound_config.inbound_mcp_tokens[1].allow_writes != true ||
            inbound_config.inbound_mcp_tokens[1].allow_admin != true) {
        std::fprintf(stderr, "inbound MCP token config contract failed: %s\n", error.c_str());
        return 1;
    }
    daemon_options inbound_options;
    apply_agent_host_config_to_daemon_options(inbound_config, inbound_options);
    if (!inbound_options.http_enabled ||
            inbound_options.http_token_profiles.size() != 2 ||
            inbound_options.http_token_profiles[0].token_env != "LLAMA_AGENT_READ_TOKEN") {
        std::fprintf(stderr, "inbound MCP token config was not projected to daemon options\n");
        return 1;
    }
    agent_host_config jwt_config;
    const json jwt_config_json = {
        {"mcp", {{"inbound", {
            {"enabled", true},
            {"authorization", {
                {"mode", "jwt"},
                {"issuer", "https://issuer.example.test/"},
                {"audience", "https://agent.example.test/mcp"},
                {"jwks_uri", "https://issuer.example.test/.well-known/jwks.json"},
                {"allowed_algorithms", json::array({"RS256"})},
                {"required_scopes", json::array({"agent:mcp"})},
                {"tool_profile", "minimal"},
                {"allowed_tools", json::array({"math.calculate"})},
                {"allow_writes", false},
            }},
        }}}},
    };
    if (!parse_agent_host_config_json(jwt_config_json, jwt_config, error) ||
            jwt_config.inbound_mcp_authorization_mode != "jwt" ||
            !jwt_config.inbound_mcp_tokens.empty() ||
            jwt_config.inbound_mcp_jwt_required_scopes.size() != 1) {
        std::fprintf(stderr, "JWT inbound MCP config contract failed: %s\n", error.c_str());
        return 1;
    }
    daemon_options jwt_options;
    apply_agent_host_config_to_daemon_options(jwt_config, jwt_options);
    if (jwt_options.http_authorization_mode != "jwt" ||
            jwt_options.http_jwt_jwks_uri.empty()) {
        std::fprintf(stderr, "JWT inbound MCP config projection failed\n");
        return 1;
    }
    agent_mcp_jwt_authenticator jwt_authenticator({
        "https://issuer.example.test/",
        "https://agent.example.test/mcp",
        "https://issuer.example.test/.well-known/jwks.json",
        {"RS256"},
        {"agent:mcp"},
        {"jwt-caller", "https://agent.example.test/mcp", "local", "project-a", "minimal", {"math.calculate"}, false},
    });
    agent_mcp_caller_policy unused_policy;
    if (jwt_authenticator.authenticate({"Bearer malformed-token"}, unused_policy, error) ||
            error != "JWT must contain three segments") {
        std::fprintf(stderr, "JWT malformed-token rejection contract failed: %s\n", error.c_str());
        return 1;
    }

    common_agent_daemon_runtime runtime;
    if (!initialize_agent_daemon_environment(options, runtime, error)) {
        std::fprintf(stderr, "daemon MCP environment init failed: %s\n", error.c_str());
        return 1;
    }
    if (!runtime.host) {
        std::fprintf(stderr, "daemon MCP environment did not create a session manager\n");
        return 1;
    }

    common_agent_runtime_tooling tooling;
    if (!resolve_agent_daemon_tooling(
            options,
            nullptr,
            {
                common_agent_runtime_host_mode::chat,
                "find tooling",
                "session-a",
                "namespace-a",
                "",
                "turn-a",
                common_memory_scope::session,
                common_plan_scope::turn,
                0,
            },
            *runtime.memory_store,
            *runtime.plan_store,
            runtime.resource_store.get(),
            tooling,
            error)) {
        std::fprintf(stderr, "daemon tooling resolve failed: %s\n", error.c_str());
        return 1;
    }
    if (!tooling.tool_view) {
        std::fprintf(stderr, "daemon tooling resolve did not return a tool view\n");
        return 1;
    }
    if (!has_tool(tooling.tools, "math.calculate") ||
            !has_tool(tooling.tools, "github_search_issues") ||
            !has_tool(tooling.tools, "github_alt_search_issues")) {
        std::fprintf(stderr, "daemon tooling resolve did not expose expected native+MCP tools\n");
        return 1;
    }

    daemon_options custom_options = options;
    custom_options.tool_profile = "custom-research";
    custom_options.tool_capabilities = {
        {"custom.utility", {"math.calculate"}},
        {"custom.network-read", {"web.search"}},
        {"custom.proposals", {"memory.remember"}},
    };
    common_tool_profile custom_profile;
    custom_profile.id = "custom-research";
    custom_profile.include_capabilities = {
        "custom.utility",
        "custom.network-read",
        "custom.proposals",
    };
    custom_profile.allow_network = true;
    custom_profile.allow_policy_gated_writes = false;
    custom_options.tool_profiles = {{custom_profile.id, custom_profile}};
    common_agent_runtime_tooling custom_tooling;
    if (!resolve_agent_daemon_tooling(
            custom_options,
            nullptr,
            {
                common_agent_runtime_host_mode::chat,
                "custom profile tooling",
                "session-a",
                "namespace-a",
                "",
                "turn-custom",
                common_memory_scope::session,
                common_plan_scope::turn,
                0,
            },
            *runtime.memory_store,
            *runtime.plan_store,
            runtime.resource_store.get(),
            custom_tooling,
            error) ||
            !custom_tooling.tool_view ||
            !has_tool(custom_tooling.tools, "math.calculate") ||
            !has_tool(custom_tooling.tools, "web.search") ||
            has_tool(custom_tooling.tools, "memory.remember")) {
        std::fprintf(stderr, "custom profile policy was not applied through daemon tooling\n");
        return 1;
    }

    common_agent_runtime_session_host_turn_request restricted_request;
    restricted_request.mode = common_agent_runtime_host_mode::chat;
    restricted_request.prompt = "restricted policy";
    restricted_request.session_id = "session-a";
    restricted_request.namespace_id = "namespace-a";
    restricted_request.project_id = "project-a";
    restricted_request.turn_id = "turn-policy";
    restricted_request.allow_policy_gated_writes = false;
    restricted_request.allowed_exposed_tool_names = {"math.calculate"};
    common_agent_runtime_tooling restricted_tooling;
    if (!resolve_agent_daemon_tooling(
            options,
            nullptr,
            restricted_request,
            *runtime.memory_store,
            *runtime.plan_store,
            runtime.resource_store.get(),
            restricted_tooling,
            error) ||
            restricted_tooling.tool_view == nullptr ||
            !restricted_tooling.tool_view->exposes_tool("math.calculate") ||
            restricted_tooling.tool_view->exposes_tool("github_search_issues") ||
            restricted_tooling.tool_view->exposes_tool("github_create_issue")) {
        std::fprintf(stderr, "restricted daemon tool policy projection failed: %s\n", error.c_str());
        return 1;
    }
    restricted_tooling.tool_view = nullptr;
    restricted_tooling.owned_resources.clear();

    auto calculator_result = tooling.tool_view->call({
        "call-1",
        "math.calculate",
        R"({"expression":"6 * 7"})",
    }, error);
    if (!calculator_result.ok || calculator_result.content_json.find("42") == std::string::npos) {
        std::fprintf(stderr, "daemon native tool call failed: %s\n", calculator_result.content_json.c_str());
        return 1;
    }

    auto github_result = tooling.tool_view->call({
        "call-2",
        "github_search_issues",
        R"({"query":"runtime host"})",
    }, error);
    if (!github_result.ok || github_result.content_json.find("stub issue") == std::string::npos) {
        std::fprintf(stderr, "daemon MCP tool call failed: %s\n", github_result.content_json.c_str());
        return 1;
    }
    auto github_alt_result = tooling.tool_view->call({
        "call-3",
        "github_alt_search_issues",
        R"({"query":"runtime host"})",
    }, error);
    if (!github_alt_result.ok || github_alt_result.content_json.find("stub issue") == std::string::npos) {
        std::fprintf(stderr, "daemon secondary MCP tool call failed: %s\n", github_alt_result.content_json.c_str());
        return 1;
    }
    const auto resolved_tool_count = tooling.tools.size();
    tooling.tool_view = nullptr;
    tooling.owned_resources.clear();
    tooling.tools.clear();
    tooling.profile_tools_active = false;

    // The service takes ownership of the runtime below. Keep non-owning
    // aliases for the store-backed diagnostic check that runs while the
    // service is still alive; the moved-from runtime must not be dereferenced.
    auto * daemon_memory_store = runtime.memory_store.get();
    auto * daemon_plan_store = runtime.plan_store.get();
    auto * daemon_resource_store = runtime.resource_store.get();
    if (daemon_memory_store == nullptr || daemon_plan_store == nullptr || daemon_resource_store == nullptr) {
        std::fprintf(stderr, "daemon MCP environment stores were unexpectedly unavailable before service move\n");
        return 1;
    }
    common_agent_daemon_service service(std::move(runtime));
    common_agent_daemon_command_result status;
    if (!service.populate_status(status, error) || !status.status.ready || !status.status.live) {
        std::fprintf(stderr, "daemon MCP status was not ready after init: %s\n", error.c_str());
        return 1;
    }

    common_agent_daemon_command shutdown_command;
    shutdown_command.request_id = "shutdown-1";
    shutdown_command.type = common_agent_daemon_command_type::shutdown;
    common_agent_daemon_command_result shutdown_result;
    error.clear();
    if (!service.execute(shutdown_command, shutdown_result, error) ||
            !shutdown_result.ok ||
            shutdown_result.event != "shutdown") {
        std::fprintf(stderr, "daemon shutdown lifecycle contract failed: %s\n", error.c_str());
        return 1;
    }

    common_agent_daemon_command rejected_turn;
    rejected_turn.request_id = "turn-after-shutdown";
    rejected_turn.type = common_agent_daemon_command_type::run_turn;
    rejected_turn.turn = common_agent_daemon_turn_payload{};
    rejected_turn.turn->request.request_id = rejected_turn.request_id;
    rejected_turn.turn->request.turn.mode = common_agent_runtime_host_mode::chat;
    rejected_turn.turn->request.turn.prompt = "hello";
    rejected_turn.turn->request.turn.session_id = "session-a";
    rejected_turn.turn->request.turn.namespace_id = "namespace-a";
    rejected_turn.turn->request.turn.project_id = "project-a";
    rejected_turn.turn->request.turn.turn_id = "turn-a";
    rejected_turn.turn->request.turn.memory_scope = common_memory_scope::session;
    rejected_turn.turn->request.turn.plan_scope = common_plan_scope::turn;
    common_agent_daemon_command_result rejected_turn_result;
    error.clear();
    if (service.execute(rejected_turn, rejected_turn_result, error) ||
            rejected_turn_result.events.empty() ||
            rejected_turn_result.events.back().type != "turn.rejected" ||
            error.find("not accepting new turns") == std::string::npos) {
        std::fprintf(stderr, "daemon post-shutdown turn rejection contract failed: %s\n", error.c_str());
        return 1;
    }

    daemon_options bad_options = options;
    if (bad_options.mcp_providers.size() < 2) {
        std::fprintf(stderr, "daemon MCP bad tooling test requires at least two MCP providers\n");
        return 1;
    }
    bad_options.mcp_providers[1].prefix = bad_options.mcp_providers[0].prefix;
    common_agent_runtime_tooling bad_tooling;
    error.clear();
    if (resolve_agent_daemon_tooling(
            bad_options,
            nullptr,
            {
                common_agent_runtime_host_mode::chat,
                "find tooling",
                "session-a",
                "namespace-a",
                "",
                "turn-a",
                common_memory_scope::session,
                common_plan_scope::turn,
                0,
            },
            *daemon_memory_store,
            *daemon_plan_store,
            daemon_resource_store,
            bad_tooling,
            error)) {
        std::fprintf(stderr, "daemon MCP bad tooling unexpectedly succeeded\n");
        return 1;
    }
    if (error.find("duplicate tool exposed in composite provider") == std::string::npos) {
        std::fprintf(stderr, "daemon MCP bad tooling did not preserve expected diagnostics: %s\n", error.c_str());
        return 1;
    }

    std::printf("daemon_mcp_ready=%s\n", status.status.ready ? "true" : "false");
    std::printf("daemon_mcp_status=%s\n", common_agent_daemon_state_name(status.status.state));
    std::printf("daemon_tooling_tools=%zu\n", resolved_tool_count);
    return 0;
}
