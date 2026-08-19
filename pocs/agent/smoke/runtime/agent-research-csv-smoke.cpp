#include "agent/thinking/research/research-runner.h"
#include "agent/thinking/research/research-workspace.h"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <sstream>

namespace {

struct csv_source {
    std::string uri;
    std::string path;
    std::string title;
    std::string content;
};

std::string read_bounded_csv(
        const std::string & path,
        const std::string & expected_header,
        std::string & error) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        error = "could not open CSV fixture: " + path;
        return {};
    }
    std::ostringstream content;
    std::string line;
    size_t lines = 0;
    while (lines++ < 8 && std::getline(input, line)) content << line << '\n';
    const auto result = content.str();
    if (result.empty() || result.find(expected_header) == std::string::npos) {
        error = "CSV fixture did not contain the expected bounded header: " + path;
        return {};
    }
    error.clear();
    return result;
}

class csv_resource_tools final : public common_agent_tool_runtime {
public:
    explicit csv_resource_tools(const csv_source & first, const csv_source & second)
        : sources{first, second} {}

    bool is_read_only(const std::string &) const override { return true; }
    bool is_policy_gated(const std::string &) const override { return false; }

    bool validate(const common_agent_tool_call & call, std::string & error) const override {
        if (call.name != "resource_read") {
            error = "CSV research smoke exposes only resource_read";
            return false;
        }
        error.clear();
        return true;
    }

    common_tool_execution_result execute(const common_agent_tool_call & call) const override {
        ++calls;
        call_names.push_back(call.name);
        const auto & source = sources[std::min<size_t>(calls - 1, sources.size() - 1)];
        common_runtime_resource_ref resource;
        resource.uri = source.uri;
        resource.name = source.title;
        resource.description = "bounded UTF-8 CSV research fixture";
        resource.mime_type = "text/csv";
        resource.size_bytes = source.content.size();
        resource.scope = common_runtime_resource_scope::turn;
        last_requested_uri = call.arguments_json;
        return common_tool_execution_result::success(
            source.content,
            "Read bounded CSV evidence from " + source.title,
            {resource});
    }

    mutable size_t calls = 0;
    mutable std::vector<std::string> call_names;
    mutable std::string last_requested_uri;

private:
    std::vector<csv_source> sources;
};

} // namespace

int main() {
    const std::string fixture_root = "pocs/agent/smoke/data/fixtures/";
    std::string error;
    csv_source sales{
        "agent-resource://fixtures/TAB6695_sv.csv",
        fixture_root + "document-table/TAB6695_sv.csv",
        "TAB6695_sv.csv", {}};
    csv_source income{
        "agent-resource://fixtures/TAB6623_sv_sample.csv",
        fixture_root + "research/TAB6623_sv_sample.csv",
        "TAB6623_sv_sample.csv", {}};
    sales.content = read_bounded_csv(sales.path, "\"varugrupp\"", error);
    if (!error.empty()) return std::fprintf(stderr, "%s\n", error.c_str()), 1;
    income.content = read_bounded_csv(income.path, "\"inkomstslag\"", error);
    if (!error.empty()) return std::fprintf(stderr, "%s\n", error.c_str()), 1;
    if (sales.content.find("fisk och skaldjur") == std::string::npos ||
            income.content.find("medianvärde") == std::string::npos) {
        std::fprintf(stderr, "CSV fixtures did not contain expected sample observations\n");
        return 1;
    }

    common_agent_research_workspace workspace;
    workspace.workspace_id = "research-csv-smoke";
    workspace.request_id = "request-csv-1";
    workspace.turn_id = "turn-csv-1";
    workspace.session_id = "session-csv-1";
    workspace.scope.namespace_id = "research-csv-smoke";
    workspace.scope.session_id = workspace.session_id;
    workspace.objective.objective_id = "csv-objective";
    workspace.objective.question = "Compare the bounded statistics CSV evidence.";
    workspace.objective.success_criteria = {"both CSV sources are read", "their provenance is retained"};
    workspace.budget.max_iterations = 2;
    workspace.budget.max_tasks = 2;
    workspace.budget.max_tool_calls = 2;
    workspace.budget.max_sources = 2;
    workspace.budget.minimum_sources = 2;
    workspace.budget.minimum_coverage = 1.0;

    if (!common_agent_research_add_gap(workspace, {
            "csv-gap", "Compare the two bounded CSV sources",
            "The research answer requires two source representations",
            "Both CSV resources support the comparison", 1}, error)) {
        return std::fprintf(stderr, "%s\n", error.c_str()), 1;
    }
    for (const auto & source : {sales, income}) {
        common_runtime_resource_ref resource;
        resource.uri = source.uri;
        resource.name = source.title;
        resource.mime_type = "text/csv";
        resource.size_bytes = source.content.size();
        resource.scope = common_runtime_resource_scope::turn;
        if (!common_agent_research_add_source(workspace, {
                source.title, source.title, source.path, "checked-in CSV fixture", "now",
                source.uri, common_agent_research_source_kind::user_supplied, resource,
                0.9, true, {}, "reference", true}, error)) {
            return std::fprintf(stderr, "%s\n", error.c_str()), 1;
        }
    }

    csv_resource_tools tools(sales, income);
    common_agent_research_runtime_adapter adapter(tools);
    common_agent_research_runner runner;
    const auto result = runner.run(workspace, adapter, error);
    if (!error.empty() || !result.complete ||
            result.stop_reason != common_agent_research_stop_reason::sufficient_coverage ||
            tools.calls != 2 || workspace.sources.size() != 2 || workspace.evidence.size() != 2 ||
            workspace.comparisons.size() != 1 || result.coverage.answered_gaps != 1 ||
            result.coverage.source_diversity != 1.0) {
        std::fprintf(stderr, "CSV research completion failed: %s\n", error.c_str());
        return 1;
    }
    if (tools.call_names != std::vector<std::string>{"resource_read", "resource_read"}) {
        std::fprintf(stderr, "CSV research selected a non-local acquisition tool\n");
        return 1;
    }
    if (result.synthesis_context.find("TAB6695_sv.csv") == std::string::npos ||
            result.synthesis_context.find("TAB6623_sv_sample.csv") == std::string::npos) {
        std::fprintf(stderr, "CSV research synthesis lost source provenance\n");
        return 1;
    }

    const auto checkpoint = make_common_agent_research_workspace_checkpoint(workspace, 1, error);
    if (!error.empty() || !common_agent_research_workspace_checkpoint_valid(checkpoint, error)) {
        std::fprintf(stderr, "CSV research checkpoint failed: %s\n", error.c_str());
        return 1;
    }
    const auto resumed = checkpoint.workspace;
    if (!common_agent_research_workspace_validate(resumed, error) ||
            resumed.evidence.size() != 2 || resumed.sources.size() != 2 ||
            resumed.comparisons.size() != 1) {
        std::fprintf(stderr, "CSV research checkpoint reload failed: %s\n", error.c_str());
        return 1;
    }

    common_agent_research_workspace incomplete = workspace;
    incomplete.gaps.front().status = common_agent_research_gap_status::open;
    incomplete.gaps.front().evidence_ids.clear();
    incomplete.evidence.clear();
    incomplete.comparisons.clear();
    incomplete.sources.pop_back();
    incomplete.budget.minimum_sources = 1;
    incomplete.budget.max_iterations = 0;
    incomplete.iterations_completed = 0;
    incomplete.tool_calls = 0;
    incomplete.tasks.clear();
    const auto incomplete_result = runner.run(incomplete, adapter, error);
    if (!error.empty() || incomplete_result.complete ||
            incomplete_result.stop_reason != common_agent_research_stop_reason::budget_exhausted) {
        std::fprintf(stderr, "CSV incomplete coverage was not deferred: error=%s complete=%d stop=%d\n",
            error.c_str(), incomplete_result.complete ? 1 : 0,
            static_cast<int>(incomplete_result.stop_reason));
        return 1;
    }
    std::printf("research_csv=ok sources=%zu evidence=%zu comparisons=%zu checkpoint=ok incomplete=deferred\n",
        result.sources.size(), result.evidence.size(), result.comparisons.size());
    return 0;
}
