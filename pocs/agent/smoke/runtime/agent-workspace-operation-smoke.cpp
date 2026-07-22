#include "agent/research/research-workspace-context.h"
#include "agent/research/research-workspace-factory.h"
#include "agent/workspace-manager.h"

#include <cassert>
#include <filesystem>

int main() {
    common_agent_request request;
    request.prompt = "Inspect the project";
    request.session_id = "session-1";
    request.turn_id = "turn-1";
    request.namespace_id = "local";
    request.project_id = "project-1";

    common_agent_research_workspace research;
    std::string error;
    assert(common_agent_research_create_workspace(request, research, error));
    const auto context = common_agent_workspace_context_from_research(research);
    assert(context.workspace_id == research.workspace_id);
    assert(context.session_id == research.session_id);
    assert(context.project_id == "project-1");

    const auto root = std::filesystem::temp_directory_path() / "llama-agent-workspace-operation-smoke";
    common_agent_workspace_manager manager({(root / "workspaces").string(), (root / "artifacts").string()});
    common_agent_workspace_operation operation;
    assert(manager.create_operation(context, "build/op-1", operation, error));
    assert(std::filesystem::is_directory(operation.source_path));
    assert(std::filesystem::is_directory(operation.writable_path));
    assert(std::filesystem::is_directory(operation.artifact_path));
    assert(operation.operation_id == "build/op-1");
    return 0;
}
