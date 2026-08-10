#include "tools/agent/cli/agent-cli-host-adapter.h"

#include "tools/agent/cli/agent-cli-options.h"

#include "memory/memory-in-memory.h"

#include <cstdio>
#include <filesystem>
#include <string>

namespace {

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

bool has_tool(const std::vector<common_chat_tool> & tools, const std::string & name) {
    for (const auto & tool : tools) {
        if (tool.name == name) {
            return true;
        }
    }
    return false;
}

} // namespace

int main(int argc, char ** argv) {
    const auto server_path = get_fake_server_path(argc > 0 ? argv[0] : nullptr);
    if (!std::filesystem::exists(server_path)) {
        std::fprintf(stderr, "fake MCP stdio server not found: %s\n", server_path.string().c_str());
        return 1;
    }

    common_memory_in_memory_store store;
    args options;
    options.command = "run";
    options.tool_profile = "cli-research-read";
    options.tool_profile_explicit = true;
    common_tool_profile profile;
    profile.id = options.tool_profile;
    profile.members = {
        {"calculator", 1, true, "{}"},
        {"time_now", 1, true, "{}"},
        {"resource_read", 1, true, "{}"},
    };
    profile.allow_network = true;
    profile.allow_policy_gated_writes = false;
    options.tool_profiles = {{profile.id, profile}};
    options.mcp_tool_command = server_path.string();
    options.mcp_tool_server_name = "github";
    options.mcp_tool_prefix = "github";
    options.max_tool_rounds = 2;
    options.memory_namespace = "local";
    options.memory_session = "smoke";

    common_memory_query query;
    query.text = "resident inference";
    query.scope = common_memory_scope::session;

    common_agent_cli_tool_selection selection;
    std::string error;
    if (!resolve_agent_cli_tool_selection(
            store,
            nullptr,
            nullptr,
            nullptr,
            options,
            query,
            false,
            selection,
            error)) {
        std::fprintf(stderr, "CLI MCP tool selection failed: %s\n", error.c_str());
        return 1;
    }

    if (selection.mcp_clients.empty()) {
        std::fprintf(stderr, "CLI MCP selection did not retain the MCP client lifetime\n");
        return 1;
    }
    if (!selection.tool_view) {
        std::fprintf(stderr, "CLI MCP selection did not return a tool view\n");
        return 1;
    }
    if (!selection.resource_processing_service || selection.resource_processors.size() != 1) {
        std::fprintf(stderr, "CLI MCP selection did not retain the host-owned resource processing service\n");
        return 1;
    }
    if (!has_tool(selection.tooling.tools, "github_search_issues")) {
        std::fprintf(stderr, "CLI MCP selection did not expose github_search_issues\n");
        return 1;
    }
    if (!has_tool(selection.tooling.tools, "calculator")) {
        std::fprintf(stderr, "CLI MCP selection did not expose native calculator from the minimal tool profile\n");
        return 1;
    }
    if (!has_tool(selection.tooling.tools, "resource_read")) {
        std::fprintf(stderr, "CLI MCP selection did not expose resource_read\n");
        return 1;
    }
    if (has_tool(selection.tooling.tools, "github_create_issue")) {
        std::fprintf(stderr, "CLI MCP selection exposed github_create_issue despite writes being disabled\n");
        return 1;
    }

    const auto native_result = selection.tool_view->call({
        "call-native",
        "calculator",
        R"({"expression":"2 + 3"})",
    }, error);
    if (!native_result.ok || native_result.content_json.find("5") == std::string::npos) {
        std::fprintf(stderr, "CLI MCP selection native tool call did not return the expected result: %s\n", native_result.content_json.c_str());
        return 1;
    }

    agent_resource_put_request pdf_request;
    pdf_request.name = "cli-selection-report.pdf";
    pdf_request.description = "PDF source for CLI resource processing coverage.";
    pdf_request.mime_type = "application/pdf";
    pdf_request.scope = common_runtime_resource_scope::session;
    pdf_request.namespace_id = options.memory_namespace;
    pdf_request.session_id = options.memory_session;
    pdf_request.bytes = "%PDF-1.7\n1 0 obj\n<< /Type /Page >>\nstream\nBT (CLI PDF text) Tj ET\nendstream\nendobj\n";
    agent_resource_descriptor pdf_descriptor;
    if (!selection.owned_resource_store ||
            !selection.owned_resource_store->put_bytes(pdf_request, pdf_descriptor, error)) {
        std::fprintf(stderr, "CLI PDF resource setup failed: %s\n", error.c_str());
        return 1;
    }
    const auto pdf_read_result = selection.tool_view->call({
        "call-resource-read-pdf",
        "resource_read",
        std::string(R"({"uri":")") + pdf_descriptor.uri + R"(","representation":"text","max_bytes":1024})",
    }, error);
    if (!pdf_read_result.ok ||
            pdf_read_result.content_json.find("CLI PDF text") == std::string::npos ||
            pdf_read_result.content_json.find("resource.process:pdf-text-local-v1") == std::string::npos) {
        std::fprintf(stderr, "CLI resource_read did not materialize PDF text: %s\n", pdf_read_result.content_json.c_str());
        return 1;
    }

    const auto result = selection.tool_view->call({
        "call-1",
        "github_search_issues",
        R"({"query":"resident inference"})",
    }, error);
    if (!result.ok || result.content_json.find("stub issue") == std::string::npos) {
        std::fprintf(stderr, "CLI MCP selection tool call did not return the expected result: %s\n", result.content_json.c_str());
        return 1;
    }

    args unavailable_options = options;
    unavailable_options.tool_profile = "cli-sandbox-profile";
    common_tool_profile sandbox_profile;
    sandbox_profile.id = unavailable_options.tool_profile;
    sandbox_profile.members = {
        {"calculator", 1, true, "{}"},
        {"development.build", 1, true, "{}"},
        {"development.test", 1, true, "{}"},
    };
    unavailable_options.tool_profiles[sandbox_profile.id] = sandbox_profile;
    common_agent_cli_tool_selection unavailable_selection;
    if (!resolve_agent_cli_tool_selection(
            store,
            nullptr,
            nullptr,
            nullptr,
            unavailable_options,
            query,
            false,
            unavailable_selection,
            error) ||
            !unavailable_selection.tool_view ||
            !has_tool(unavailable_selection.tooling.tools, "calculator") ||
            has_tool(unavailable_selection.tooling.tools, "development.build") ||
            has_tool(unavailable_selection.tooling.tools, "development.test")) {
        std::fprintf(stderr, "CLI selection did not disable sandbox tools without a backend\n");
        return 1;
    }

    std::printf("cli_mcp_tools=%zu\n", selection.tooling.tools.size());
    std::printf("cli_native_result=%s\n", native_result.content_json.c_str());
    std::printf("cli_mcp_search_result=%s\n", result.content_json.c_str());
    return 0;
}
