#include "tools/agent/resource/agent-resource-store.h"
#include "tools/agent/tooling/agent-tool-provider.h"
#include "tools/agent/tooling/agent-tool-runtime-adapter.h"

#include "agent/agent-runtime.h"
#include "agent/tool-catalog.h"
#include "memory/memory-context.h"
#include "plan/plan-in-memory.h"

#include <chrono>
#include <cstdio>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace {

size_t count_substring_occurrences(const std::string & text, const std::string & needle) {
    if (needle.empty()) {
        return 0;
    }
    size_t count = 0;
    size_t offset = 0;
    while ((offset = text.find(needle, offset)) != std::string::npos) {
        ++count;
        offset += needle.size();
    }
    return count;
}

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

class reflection_memory_get_planner final : public common_planner {
public:
    common_plan_proposal create_plan(const common_agent_request & request, std::string & error) override {
        common_plan_proposal proposal;
        proposal.plan.id = "runtime-reflection-memory-get";
        proposal.plan.namespace_id = request.namespace_id;
        proposal.plan.session_id = request.session_id;
        proposal.plan.project_id = request.project_id;
        proposal.plan.turn_id = request.turn_id;
        proposal.plan.scope = request.plan_scope;
        proposal.plan.status = common_plan_status::active;
        proposal.plan.purpose = request.prompt;
        proposal.plan.goal = "Draft an answer before reflection.";
        proposal.plan.success_criteria = "Reach reflection with a bounded answer draft.";

        common_plan_step final_step;
        final_step.id = "answer";
        final_step.title = "Answer";
        final_step.objective = "Return a short draft answer.";
        final_step.intended_contribution = final_step.objective;
        final_step.status = common_plan_step_status::active;
        final_step.mode = common_plan_step_mode::final_response;

        proposal.plan.active_step_id = final_step.id;
        proposal.plan.steps.push_back(std::move(final_step));
        error.clear();
        return proposal;
    }
};

class reflection_memory_get_executor final : public common_action_executor {
public:
    std::string generate_draft(
            const common_agent_request &,
            const common_plan_state &,
            const std::vector<std::string> &,
            std::string & error) override {
        error.clear();
        return "Draft answer before reflection.";
    }

    std::string generate_reasoning(
            const common_agent_request &,
            const common_plan_state &,
            const common_plan_step &,
            std::string & error) override {
        error.clear();
        return "The repair step was executed before the answer was accepted.";
    }
};

class reflection_memory_get_reflector final : public common_reflection_engine {
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

        common_plan_step step;
        step.id = "repair-memory";
        step.title = "Read selected memory";
        step.objective = "Fetch the already selected memory.";
        step.intended_contribution = step.objective;
        step.mode = common_plan_step_mode::tool;
        step.selected_tool = "memory_get";
        step.tool_call = common_plan_tool_call{"memory_get", "{}"};

        common_plan_operation op;
        op.kind = common_plan_operation_kind::add_step;
        op.reason_summary = "Reflection wants to inspect a specific memory.";
        op.step = std::move(step);
        result.proposed_plan_operations.push_back(std::move(op));
        return result;
    }
};

} // namespace

int main() {
    std::vector<common_memory_hit> symbolic_hits;
    {
        common_memory_record constraint;
        constraint.id = "memory-constraint";
        constraint.kind = common_memory_kind::constraint;
        constraint.content = "Do not let the model choose host authority such as resource scope or store identity.";
        symbolic_hits.push_back({constraint, 0.9f, 0.0f, 0.0f, 0.9f, "runtime-smoke"});

        common_memory_record decision;
        decision.id = "memory-decision";
        decision.kind = common_memory_kind::decision;
        decision.content = "Use host-owned resource references for large tool outputs instead of forcing everything inline.";
        symbolic_hits.push_back({decision, 0.8f, 0.0f, 0.0f, 0.8f, "runtime-smoke"});

        common_memory_record procedure;
        procedure.id = "memory-procedure";
        procedure.kind = common_memory_kind::procedure;
        procedure.content = "Run a bounded search first, then answer using the recorded evidence.";
        symbolic_hits.push_back({procedure, 0.7f, 0.0f, 0.0f, 0.7f, "runtime-smoke"});

        common_memory_record duplicate_constraint;
        duplicate_constraint.id = "memory-constraint-duplicate";
        duplicate_constraint.kind = common_memory_kind::constraint;
        duplicate_constraint.content = "Do not let the model choose host authority such as resource scope or store identity.";
        symbolic_hits.push_back({duplicate_constraint, 0.6f, 0.0f, 0.0f, 0.6f, "runtime-smoke"});
    }
    const std::string symbolic_overlay = common_memory_render_symbolic_overlay(symbolic_hits);
    if (symbolic_overlay.find("Constraints:") == std::string::npos ||
            symbolic_overlay.find("Decisions:") == std::string::npos ||
            symbolic_overlay.find("Procedures:") == std::string::npos) {
        std::fprintf(stderr, "symbolic overlay did not render the expected sections: %s\n", symbolic_overlay.c_str());
        return 1;
    }
    if (count_substring_occurrences(
                symbolic_overlay,
                "Do not let the model choose host authority such as resource scope or store identity.") != 1) {
        std::fprintf(stderr, "symbolic overlay compaction did not deduplicate repeated constraint text: %s\n", symbolic_overlay.c_str());
        return 1;
    }
    common_memory_policy_pack policy_pack;
    policy_pack.id = "runtime-smoke-policy";
    policy_pack.purpose = "Keep host authority explicit while using resident runtime evidence.";
    policy_pack.constraints = {
        "Do not let the model choose scope or storage authority.",
        "Keep runtime evidence host-owned and bounded.",
        "Do not let the model choose scope or storage authority.",
    };
    policy_pack.decisions = {
        "Prefer resource references for larger tool payloads.",
        "Prefer resource references for larger tool payloads.",
    };
    const std::string rendered_policy_pack = common_memory_render_policy_pack(policy_pack);
    if (rendered_policy_pack.find("<policy_pack>") == std::string::npos ||
            rendered_policy_pack.find("Constraints:") == std::string::npos ||
            rendered_policy_pack.find("Decisions:") == std::string::npos) {
        std::fprintf(stderr, "policy pack did not render the expected sections: %s\n", rendered_policy_pack.c_str());
        return 1;
    }
    if (!common_memory_policy_pack_needs_compaction(policy_pack)) {
        std::fprintf(stderr, "policy pack compaction heuristic did not detect repeated items\n");
        return 1;
    }
    if (count_substring_occurrences(
                rendered_policy_pack,
                "Do not let the model choose scope or storage authority.") != 1 ||
            count_substring_occurrences(
                rendered_policy_pack,
                "Prefer resource references for larger tool payloads.") != 1) {
        std::fprintf(stderr, "policy pack compaction did not deduplicate repeated items: %s\n", rendered_policy_pack.c_str());
        return 1;
    }
    common_memory_policy_pack compact_policy_pack = common_memory_compact_policy_pack(policy_pack);
    if (common_memory_policy_pack_needs_compaction(compact_policy_pack)) {
        std::fprintf(stderr, "compacted policy pack should not still need compaction\n");
        return 1;
    }
    const auto planning_overlay_hits = common_memory_select_symbolic_overlay_hits(
        symbolic_hits,
        common_memory_overlay_stage::planning,
        3);
    const auto reasoning_overlay_hits = common_memory_select_symbolic_overlay_hits(
        symbolic_hits,
        common_memory_overlay_stage::reasoning,
        3);
    if (planning_overlay_hits.empty() || planning_overlay_hits.front().memory.kind != common_memory_kind::constraint) {
        std::fprintf(stderr, "planning overlay selection did not prioritize constraints\n");
        return 1;
    }
    if (reasoning_overlay_hits.empty() || reasoning_overlay_hits.front().memory.kind != common_memory_kind::procedure) {
        std::fprintf(stderr, "reasoning overlay selection did not prioritize procedures\n");
        return 1;
    }

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
    tool_context.async_exposed_tool_names = {"web_search"};
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
    common_agent_tool_call async_call;
    async_call.name = "web_search";
    async_call.arguments_json = R"({"query":"runtime async smoke"})";
    if (!tool_runtime->supports_async(async_call)) {
        std::fprintf(stderr, "provider runtime adapter did not expose configured async tool\n");
        return 1;
    }
    common_runtime_operation_ref async_operation;
    if (!tool_runtime->begin_async(async_call, async_operation, error) ||
            async_operation.operation_id.empty()) {
        std::fprintf(stderr, "provider runtime adapter async start failed: %s\n", error.c_str());
        return 1;
    }
    bool async_ready = false;
    common_tool_execution_result async_result;
    for (int attempt = 0; attempt < 20 && !async_ready; ++attempt) {
        if (!tool_runtime->poll_async(async_operation, async_ready, async_result, error)) {
            std::fprintf(stderr, "provider runtime adapter async poll failed: %s\n", error.c_str());
            return 1;
        }
        if (!async_ready) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }
    if (!async_ready || !async_result.ok) {
        std::fprintf(stderr, "provider runtime adapter async call did not complete successfully\n");
        return 1;
    }
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
    request.policy_pack = policy_pack;
    request.memories = symbolic_hits;
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
    bool saw_plan_created_event = false;
    bool saw_observation_event = false;
    bool saw_resource_attached_event = false;
    bool saw_resource_created_event = false;
    bool saw_tool_success = false;
    bool saw_response_complete = false;
    bool saw_observation_resource = false;
    for (const auto & event : result.events) {
        if (event.type == common_agent_event_type::plan_created &&
                event.plan_id && *event.plan_id == "runtime-smoke-plan") {
            saw_plan_created_event = true;
        }
        if (event.type == common_agent_event_type::observation_recorded &&
                event.tool_name == "web_search" &&
                !event.observation_id.empty()) {
            saw_observation_event = true;
        }
        if (event.type == common_agent_event_type::resource_created &&
                event.resource_uri.rfind("agent-resource://", 0) == 0) {
            saw_resource_created_event = true;
        }
        if (event.type == common_agent_event_type::resource_attached &&
                event.tool_name == "web_search" &&
                event.resource_uri.rfind("agent-resource://", 0) == 0) {
            saw_resource_attached_event = true;
        }
    }
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
    if (!saw_plan_created_event ||
            !saw_observation_event ||
            !saw_resource_created_event ||
            !saw_resource_attached_event) {
        std::fprintf(stderr, "runtime events did not include expected direct plan/observation/resource signals\n");
        return 1;
    }

    reflection_memory_get_planner reflection_planner;
    reflection_memory_get_executor reflection_executor;
    reflection_memory_get_reflector reflection_reflector;
    common_plan_in_memory_store reflection_plan_store;
    if (!reflection_plan_store.open("", error)) {
        std::fprintf(stderr, "failed to open reflection plan store: %s\n", error.c_str());
        return 1;
    }
    common_agent_runtime reflection_runtime(
        reflection_plan_store,
        reflection_planner,
        reflection_executor,
        reflection_reflector,
        tool_runtime.get(),
        nullptr);

    common_agent_request reflection_request;
    reflection_request.prompt = "Use reflection to repair the plan if needed.";
    reflection_request.namespace_id = "runtime-smoke";
    reflection_request.session_id = "runtime-smoke-session";
    reflection_request.turn_id = "runtime-smoke-turn-reflection";
    reflection_request.plan_scope = common_plan_scope::session;
    reflection_request.enable_planning = true;
    reflection_request.enable_reflection = true;
    reflection_request.max_iterations = 2;
    reflection_request.max_tool_batches = 1;

    const auto reflection_result = reflection_runtime.run(reflection_request);
    if (reflection_result.error.empty() ||
            reflection_result.error.find("not allowed") == std::string::npos) {
        std::fprintf(stderr, "reflection runtime did not fail closed for the disallowed memory_get repair: %s\n", reflection_result.error.c_str());
        return 1;
    }
    bool saw_policy_failure = false;
    for (const auto & trace : reflection_result.trace) {
        saw_policy_failure = saw_policy_failure ||
            (trace.stage == common_runtime_trace_stage::reflection &&
             trace.kind == common_runtime_trace_kind::failed &&
             trace.detail.find("memory_get") != std::string::npos);
    }
    if (!saw_policy_failure) {
        std::fprintf(stderr, "reflection runtime did not emit a failed reflection trace for memory_get\n");
        return 1;
    }

    std::printf("runtime_response=%s\n", result.response.c_str());
    std::printf("tool_events=%zu\n", result.events.size());
    std::printf("trace_entries=%zu\n", result.trace.size());
    std::printf("plan_id=%s\n", result.plan_id ? result.plan_id->c_str() : "");
    return 0;
}
