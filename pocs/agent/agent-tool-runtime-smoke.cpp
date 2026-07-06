#include "agent-tool-provider.h"
#include "agent-resource-store.h"
#include "agent-tool-runtime-adapter.h"

#include "agent/agent-runtime.h"
#include "agent/tool-catalog.h"
#include "plan/plan-in-memory.h"

#include <cstdio>
#include <memory>
#include <string>
#include <vector>

namespace {

agent_catalogued_resource_store g_runtime_resource_store(
    std::make_shared<agent_in_memory_blob_store>(),
    std::make_unique<agent_in_memory_resource_catalog>());

class smoke_planner final : public common_planner {
public:
    common_plan_proposal create_plan(const common_agent_request & request, std::string & error) override {
        common_plan_proposal proposal;
        proposal.plan.id = "runtime-smoke-plan";
        proposal.plan.namespace_id = request.namespace_id;
        proposal.plan.session_id = request.session_id;
        proposal.plan.project_id = request.project_id;
        proposal.plan.turn_id = request.turn_id;
        proposal.plan.scope = request.plan_scope;
        proposal.plan.status = common_plan_status::active;
        proposal.plan.purpose = request.prompt;
        proposal.plan.goal = "Run web search and answer using the recorded resource evidence.";
        proposal.plan.success_criteria = "Return the web search result summary.";

        common_plan_step tool_step;
        tool_step.id = "step-tool";
        tool_step.title = "Run web search";
        tool_step.objective = "Collect a bounded search result and host-owned resource reference.";
        tool_step.intended_contribution = tool_step.objective;
        tool_step.status = common_plan_step_status::active;
        tool_step.mode = common_plan_step_mode::tool;
        tool_step.selected_tool = "web_search";
        tool_step.tool_call = common_plan_tool_call{"web_search", R"({"query":"resident runtime resource evidence","limit":5})"};

        common_plan_step final_step;
        final_step.id = "step-final";
        final_step.title = "Return answer";
        final_step.objective = "Respond with the calculator result.";
        final_step.intended_contribution = final_step.objective;
        final_step.status = common_plan_step_status::pending;
        final_step.mode = common_plan_step_mode::final_response;
        final_step.depends_on.push_back(tool_step.id);

        proposal.plan.active_step_id = tool_step.id;
        proposal.plan.steps.push_back(std::move(tool_step));
        proposal.plan.steps.push_back(std::move(final_step));
        error.clear();
        return proposal;
    }
};

class smoke_executor final : public common_action_executor {
public:
    std::string generate_draft(
            const common_agent_request &,
            const common_plan_state & plan,
            const std::vector<std::string> &,
            std::string & error) override {
        for (const auto & observation : plan.observations) {
            if (observation.source == "web_search" &&
                    !observation.resource_refs.empty() &&
                    observation.resource_refs[0].metadata.content_summary.find("Stubbed") != std::string::npos) {
                error.clear();
                return "The search evidence was recorded with a resource reference.";
            }
        }
        error = "web_search observation missing expected resource evidence";
        return {};
    }
};

class smoke_reflector final : public common_reflection_engine {
public:
    common_reflection_result evaluate(
            const common_agent_request &,
            const common_plan_state &,
            const std::string &,
            std::string & error) override {
        error.clear();
        common_reflection_result result;
        result.decision = common_reflection_decision::accept;
        result.ready_to_answer = true;
        result.confidence = 1.0f;
        return result;
    }
};

} // namespace

int main() {
    std::string error;
    common_tool_catalog catalog;
    common_tool_bootstrap_result bootstrap;
    if (!catalog.bootstrap("research", bootstrap, error)) {
        std::fprintf(stderr, "tool bootstrap failed: %s\n", error.c_str());
        return 1;
    }

    native_agent_tool_provider provider(
        catalog,
        [](const agent_tool_context & context, common_native_tool_bindings & bindings, std::string &) {
            bindings.resource_runtime.store = &g_runtime_resource_store;
            bindings.resource_runtime.namespace_id = context.scope.namespace_id;
            bindings.resource_runtime.session_id = context.scope.session_id;
            bindings.resource_runtime.project_id = context.scope.project_id;
            bindings.resource_runtime.turn_id = context.scope.turn_id;
            const agent_resource_runtime runtime = bindings.resource_runtime;
            bindings.web_search = [runtime](const std::string &) {
                agent_resource_put_request request;
                request.name = "runtime-search-results.json";
                request.description = "Runtime smoke full search payload";
                request.mime_type = "application/json";
                request.text = R"({"results":[{"title":"stub runtime result","url":"https://example.com/runtime"}],"provider":"stub"})";
                request.scope = common_runtime_resource_scope::turn;
                request.source_provider = "native";
                request.source_tool = "web_search";
                request.metadata = {
                    "Preserve the full search payload for later runtime evidence.",
                    "Stubbed runtime search result payload.",
                    "Use the resource URI when a later step needs the full search payload.",
                    "Runtime smoke uses stubbed search data.",
                    {"runtime", "search"},
                    {},
                };
                apply_agent_resource_runtime(runtime, request);

                agent_resource_descriptor descriptor;
                std::string error;
                if (!runtime.store->put_text(request, descriptor, error)) {
                    return common_tool_execution_result::failure(
                        "tool.web_search.resource_store_failed",
                        common_tool_failure_class::execution,
                        false,
                        "Runtime smoke failed to write a resource.",
                        error);
                }
                common_runtime_resource_ref resource = descriptor;
                return common_tool_execution_result::success(
                    R"({"results":[{"title":"stub runtime result"}],"provider":"stub"})",
                    "Web search returned one stub candidate; the full result set was stored as a turn resource.",
                    {resource});
            };
            return true;
        });

    agent_tool_context tool_context;
    tool_context.request_id = "runtime-smoke";
    tool_context.turn_id = "turn-1";
    tool_context.profile_id = "research";
    tool_context.max_calls = 2;
    tool_context.allow_network = true;
    tool_context.scope.namespace_id = "runtime-smoke";
    tool_context.scope.session_id = "runtime-smoke-session";
    tool_context.scope.turn_id = "runtime-smoke-turn";

    std::unique_ptr<agent_tool_view> tool_view = provider.resolve_tools(tool_context, error);
    if (!tool_view) {
        std::fprintf(stderr, "provider resolve failed: %s\n", error.c_str());
        return 1;
    }

    std::unique_ptr<common_agent_tool_runtime> tool_runtime =
        make_provider_agent_tool_runtime(*tool_view);
    common_plan_in_memory_store plan_store;
    if (!plan_store.open("", error)) {
        std::fprintf(stderr, "failed to open plan store: %s\n", error.c_str());
        return 1;
    }

    smoke_planner planner;
    smoke_executor executor;
    smoke_reflector reflector;
    common_agent_runtime runtime(plan_store, planner, executor, reflector, tool_runtime.get(), nullptr);

    common_agent_request request;
    request.prompt = "Find runtime resource evidence.";
    request.namespace_id = "runtime-smoke";
    request.session_id = "runtime-smoke-session";
    request.turn_id = "runtime-smoke-turn";
    request.plan_scope = common_plan_scope::session;
    request.enable_planning = true;
    request.enable_reflection = false;
    request.max_iterations = 1;
    request.max_tool_batches = 1;

    const auto result = runtime.run(request);
    if (!result.error.empty()) {
        std::fprintf(stderr, "runtime run failed: %s\n", result.error.c_str());
        return 1;
    }
    if (result.response.find("resource reference") == std::string::npos) {
        std::fprintf(stderr, "runtime response missing expected value: %s\n", result.response.c_str());
        return 1;
    }
    if (!result.plan_id || *result.plan_id != "runtime-smoke-plan") {
        std::fprintf(stderr, "runtime did not retain expected plan id\n");
        return 1;
    }
    bool saw_plan_create = false;
    bool saw_tool_success = false;
    bool saw_response_complete = false;
    bool saw_observation_resource = false;
    for (const auto & entry : result.trace) {
        if (entry.stage == common_runtime_trace_stage::plan &&
                entry.kind == common_runtime_trace_kind::started &&
                entry.plan_id == "runtime-smoke-plan") {
            saw_plan_create = true;
        }
        if (entry.stage == common_runtime_trace_stage::tool &&
                entry.kind == common_runtime_trace_kind::succeeded &&
                entry.tool_name == "web_search") {
            saw_tool_success = true;
        }
        if (entry.stage == common_runtime_trace_stage::response &&
                entry.kind == common_runtime_trace_kind::completed) {
            saw_response_complete = true;
        }
    }
    auto stored_plan = plan_store.get("runtime-smoke-plan", error);
    if (!stored_plan || !error.empty()) {
        std::fprintf(stderr, "runtime smoke could not reload stored plan: %s\n", error.c_str());
        return 1;
    }
    for (const auto & observation : stored_plan->observations) {
        if (observation.source == "web_search" && !observation.resource_refs.empty()) {
            saw_observation_resource = true;
            break;
        }
    }
    if (!saw_plan_create || !saw_tool_success || !saw_response_complete || !saw_observation_resource) {
        std::fprintf(stderr, "runtime trace did not include expected plan/tool/response history\n");
        return 1;
    }

    std::printf("runtime_response=%s\n", result.response.c_str());
    std::printf("tool_events=%zu\n", result.events.size());
    std::printf("trace_entries=%zu\n", result.trace.size());
    std::printf("plan_id=%s\n", result.plan_id ? result.plan_id->c_str() : "");
    return 0;
}
