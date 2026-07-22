#include "agent/research/research-workspace-context.h"
#include "agent/research/research-workspace-factory.h"
#include "agent/workspace-manager.h"

#include <cassert>
#include <filesystem>

class smoke_resource_store final : public agent_resource_store {
public:
    bool put_text(const agent_resource_put_request &, agent_resource_descriptor &, std::string &) override { return false; }
    bool read_text(const std::string & uri, const agent_resource_read_authority &, size_t, std::string & out, std::string & error) const override {
        if (uri != "resource://input.txt") { error = "resource not found"; return false; }
        out = "input from research\n";
        error.clear();
        return true;
    }
    bool stat(const std::string &, const agent_resource_read_authority &, agent_resource_descriptor &, std::string &) const override { return false; }
    bool list(const agent_resource_read_authority &, std::vector<agent_resource_descriptor> &, std::string &) const override { return false; }
};

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

    smoke_resource_store resources;
    common_runtime_resource_ref input;
    input.uri = "resource://input.txt";
    std::string materialized_path;
    assert(manager.materialize_text_resource(
        operation, input, resources, {}, "input.txt", 4096, materialized_path, error));
    std::ifstream materialized(materialized_path);
    std::string line;
    std::getline(materialized, line);
    assert(line == "input from research");
    return 0;
}
