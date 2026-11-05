#include "tools/agent/runtime/agent-runtime-execution.h"
#include "tools/agent/runtime/agent-runtime-resident.h"
#include "tools/agent/runtime/agent-planning-selection.h"
#include "agent/thinking/research/research-workspace-factory.h"
#include "agent/thinking/research/research-controller.h"
#include "agent/runtime/agent-tool-runtime.h"
#include "memory/memory-in-memory.h"
#include "plan/plan-in-memory.h"

#include <stdexcept>

namespace {

void require(bool passed, const char * message) {
    if (!passed) throw std::runtime_error(message);
}

class unused_inference final : public common_agent_inference {
public:
    bool generate(const common_agent_generation_request &, common_agent_generation_result &) override {
        throw std::runtime_error("research binding contract must not generate text");
    }
};

class selection_inference final : public common_agent_inference {
public:
    std::vector<std::string> replies;
    std::vector<common_agent_generation_request> requests;
    bool generate(const common_agent_generation_request & request, common_agent_generation_result & result) override {
        if (requests.size() >= replies.size()) throw std::runtime_error("unexpected research selection generation");
        result.content = replies[requests.size()];
        requests.push_back(request);
        result.status = common_agent_generation_status::completed;
        result.stop_reason = common_agent_generation_stop_reason::eos;
        return true;
    }
};

class selection_runtime final : public common_agent_tool_runtime {
public:
    bool is_read_only(const std::string &) const override { return true; }
    bool is_policy_gated(const std::string &) const override { return false; }
    bool is_available(const std::string &) const override { return true; }
    bool validate(const common_agent_tool_call &, std::string & error) const override { error.clear(); return true; }
    common_tool_execution_result execute(const common_agent_tool_call &) const override {
        throw std::runtime_error("selection must not execute tools");
    }
};

} // namespace

int main() {
    {
        common_agent_request request;
        request.tool_output_format = common_agent_tool_output_format::native;
        common_agent_generation_config config;
        std::vector<std::string> sources{"stale"};
        std::string error;
        selection_inference automatic;
        automatic.replies = {R"({"sources":[]})"};
        require(select_agent_research_source(automatic, config, request, "Find relevant evidence", {"catalog"}, sources, error) &&
            sources.empty() && automatic.requests.front().json_schema.find("\"mode\"") == std::string::npos,
            "automatic source choice retained a redundant mode or stale sources");
        selection_inference named;
        named.replies = {R"({"sources":["Catalog"]})"};
        require(select_agent_research_source(named, config, request, "Search Catalog", {"catalog"}, sources, error) &&
            sources == std::vector<std::string>{"catalog"}, "named source was not resolved to its registered identity");
        request.tool_output_format = common_agent_tool_output_format::compact_dsl;
        selection_inference dsl;
        dsl.replies = {"open! selection.research_source\nsources: []"};
        require(select_agent_research_source(dsl, config, request, "Find relevant evidence", {"catalog"}, sources, error) &&
            sources.empty(), "DSL source selection could not express automatic choice");
    }
    // A resident model may serve successive turns with different host-owned
    // draft bindings. Never retain the previous turn's callback or seed.
    common_agent_runtime_turn_request resident_base;
    common_agent_research_draft_ref first_draft, second_draft;
    first_draft.draft_id = "first-draft";
    second_draft.draft_id = "second-draft";
    common_agent_research_workspace first_seed, second_seed;
    first_seed.draft_id = first_draft.draft_id;
    second_seed.draft_id = second_draft.draft_id;
    int first_calls = 0, second_calls = 0;
    set_agent_runtime_turn_research_context(resident_base, first_draft, first_seed,
        [&](const common_agent_research_workspace &, std::string &) { ++first_calls; return true; });
    const auto first_turn = make_agent_runtime_resident_turn_request(resident_base, "first", "turn-1");
    set_agent_runtime_turn_research_context(resident_base, second_draft, second_seed,
        [&](const common_agent_research_workspace &, std::string &) { ++second_calls; return true; });
    const auto second_turn = make_agent_runtime_resident_turn_request(resident_base, "second", "turn-2");
    std::string sink_error;
    require(first_turn.request.research_draft->draft_id == "first-draft" &&
        second_turn.request.research_draft->draft_id == "second-draft" &&
        second_turn.request.research_workspace_seed->draft_id == "second-draft",
        "resident turn did not replace its research draft and seed");
    require(second_turn.request.research_progress_sink(second_seed, sink_error) &&
        first_calls == 0 && second_calls == 1, "resident turn reused a prior research callback");
    set_agent_runtime_turn_research_context(resident_base, {}, {}, {});
    require(!resident_base.request.research_draft && !resident_base.request.research_workspace_seed &&
        !resident_base.request.research_progress_sink, "resident turn retained a completed turn's binding");

    common_memory_in_memory_store memories;
    common_plan_in_memory_store plans;
    unused_inference inference;
    common_agent_scope scope;
    scope.project_id = "project";
    scope.session_id = "chat";
    scope.turn_id = "next-turn";
    std::string plan_id = "plan";
    const std::string fallback_reason;
    const std::vector<common_blueprint_candidate> blueprints;
    const std::vector<common_memory_hit> hits;
    common_agent_runtime_tooling tooling;
    common_agent_orchestration_config orchestration;
    orchestration.prompt = "Find additional evidence for the paper.";
    common_agent_runtime_driver_inputs inputs{
        memories, plans, {}, {}, {}, orchestration, plan_id, scope,
        blueprints, std::nullopt, hits, common_memory_scope::session, false,
        fallback_reason, tooling};

    common_agent_research_draft_ref draft;
    draft.research_id = "research:project";
    draft.draft_id = "draft:chat";
    draft.owner_session_id = scope.session_id;
    draft.base_revision = 7;
    draft.draft_revision = 3;
    inputs.research_draft = draft;

    common_agent_request previous_request;
    common_agent_scope_apply(scope, previous_request);
    previous_request.turn_id = "previous-turn";
    previous_request.prompt = "Inspect the paper.";
    previous_request.research_draft = draft;
    previous_request.max_iterations = 4;
    previous_request.max_tool_batches = 8;
    common_agent_research_workspace previous;
    std::string error;
    require(common_agent_research_create_workspace(previous_request, previous, error),
        "could not prepare a valid previous research workspace");
    previous.iterations_completed = 2;
    inputs.research_workspace_seed = previous;
    inputs.research_progress_sink = [](const common_agent_research_workspace &, std::string & sink_error) {
        sink_error.clear(); return true;
    };

    auto execution = make_agent_runtime_driver_execution(inputs, inference);
    auto request = make_agent_runtime_driver_request(execution);
    require(request.research_draft.has_value(), "driver lost the persistent draft binding");
    require(request.research_draft->research_id == draft.research_id &&
        request.research_draft->draft_id == draft.draft_id &&
        request.research_draft->base_revision == 7 && request.research_draft->draft_revision == 3,
        "driver changed the persistent research identity or revision");
    require(request.research_workspace_seed && request.research_workspace_seed->iterations_completed == 2,
        "driver lost previously collected research state");
    require(static_cast<bool>(request.research_progress_sink), "driver lost the host progress sink");

    common_agent_research_workspace resumed;
    require(common_agent_research_create_workspace(request, resumed, error),
        "driver request could not resume the persistent research workspace");
    const auto checkpoint = make_common_agent_research_workspace_checkpoint(resumed, 3, error);
    require(!checkpoint.checkpoint_id.empty(), "resumed research produced an invalid checkpoint");
    require(checkpoint.workspace.research_id == draft.research_id &&
        checkpoint.workspace.draft_id == draft.draft_id &&
        checkpoint.workspace.base_revision == draft.base_revision &&
        checkpoint.workspace.session_id == scope.session_id,
        "checkpoint no longer belongs to the host-bound persistent draft");
    require(resumed.turn_id == scope.turn_id && resumed.iterations_completed == 2 &&
        resumed.gaps.size() >= previous.gaps.size(), "resume discarded history or kept the previous turn identity");

    // Requests rebuilt for a later slice must keep the same draft and the
    // latest checkpoint, without treating a slice as a persisted revision.
    execution.research_workspace_seed = checkpoint.workspace;
    const auto continuation = make_agent_runtime_driver_request(execution);
    require(continuation.research_workspace_seed &&
        continuation.research_workspace_seed->workspace_id == checkpoint.workspace_id &&
        continuation.research_draft->draft_revision == draft.draft_revision,
        "continuation lost the checkpoint or advanced the host-owned draft revision");

    execution.research_draft.reset();
    execution.research_workspace_seed.reset();
    const auto ordinary = make_agent_runtime_driver_request(execution);
    require(!ordinary.research_draft && !ordinary.research_workspace_seed,
        "ordinary driver requests acquired an implicit research binding");

    tooling.tools = {
        {"openalex.listWorks", "Search for scholarly works", R"({"type":"object","additionalProperties":false,"required":["search"],"properties":{"search":{"type":"string","maxLength":100}}})"},
        {"openalex.getWork", "Get a work by observed ID", R"({"type":"object","additionalProperties":false,"required":["id"],"properties":{"id":{"type":"string"}}})"},
        {"web.search", "Search the web", R"({"type":"object","additionalProperties":false,"required":["query"],"properties":{"query":{"type":"string"}}})"},
    };
    tooling.tools[0].mcp_provider_id = "openalex";
    tooling.tools[1].mcp_provider_id = "openalex";
    tooling.tools[0].collection_item_parameter = "id";
    tooling.tools[0].collection_items_pointer = "/results";
    tooling.tools[1].item_lookup_parameter = "id";
    inputs.tool_output_format = common_agent_tool_output_format::native;
    selection_inference model;
    model.replies = {R"({"sources":["OpenAlex"]})", "TOOLS: openalex", R"("openalex.listWorks")", R"({"tool":"openalex.listWorks","args":{"search":"urban cooling"}})"};
    auto selection_execution = make_agent_runtime_driver_execution(inputs, model);
    const auto selection_request = make_agent_runtime_driver_request(selection_execution);
    common_agent_research_workspace research;
    research.objective.question = "Search OpenAlex for urban cooling";
    common_agent_research_action action;
    action.instruction = research.objective.question;
    selection_runtime runtime;
    std::optional<common_agent_tool_call> call;
    std::string decision_message;
    require(selection_request.research_select_tool(action, research, runtime, nullptr, {}, call, decision_message, error), error.c_str());
    require(call && call->name == "openalex.listWorks", "research did not select the requested provider search");
    require(model.requests.size() == 4, "research did not use source, family, operation and argument generations");
    const auto prompt_contains = [](const common_agent_generation_request & request, const std::string & text) {
        for (const auto & message : request.messages) {
            if (message.content.find(text) != std::string::npos) return true;
        }
        return false;
    };
    require(prompt_contains(model.requests[2], "openalex.listWorks") &&
        !prompt_contains(model.requests[1], "web.search") && !prompt_contains(model.requests[2], "web.search"),
        "explicit provider allowed web search to replace the requested source");
    require(!prompt_contains(model.requests[2], "maxLength") &&
        model.requests[2].json_schema.find("maxLength") == std::string::npos &&
        model.requests[3].json_schema.find("maxLength") != std::string::npos &&
        model.requests[3].json_schema.find("search") != std::string::npos,
        "operation choice exposed arguments before a tool was selected or argument schema was omitted");

    // Each selection generation is charged before execution, including the
    // source classifier; an exhausted budget must not invoke the model.
    auto budget_workspace = research;
    budget_workspace.budget.max_tokens = 100;
    selection_inference budget_model;
    auto budget_execution = make_agent_runtime_driver_execution(inputs, budget_model);
    const auto budget_request = make_agent_runtime_driver_request(budget_execution);
    require(!budget_request.research_select_tool(action, budget_workspace, runtime, nullptr, {}, call, decision_message, error) &&
        error.find("budget") != std::string::npos && budget_model.requests.empty(),
        "research source selection ignored its generation budget");

    selection_inference assessment_model;
    assessment_model.replies = {R"({"status":"sufficient","confidence":0.8,"summary":"The finding addresses the question."})",
        R"({"relation":"conflict","summary":"The sources report different outcomes."})"};
    auto assessment_execution = make_agent_runtime_driver_execution(inputs, assessment_model);
    const auto assessment_request = make_agent_runtime_driver_request(assessment_execution);
    common_agent_research_gap assessment_gap{"gap", "What happened?", "", "Determine the outcome", 1};
    common_agent_research_evidence evidence_a;
    evidence_a.source_id = "source-a";
    evidence_a.statement = "Outcome A";
    common_agent_research_evidence evidence_b;
    evidence_b.source_id = "source-b";
    evidence_b.statement = "Outcome B";
    common_agent_research_assessment assessment;
    common_agent_research_source_comparison comparison;
    int used = 0;
    require(assessment_request.research_assess_evidence(assessment_gap, evidence_a, assessment, used, error) &&
        assessment.status == common_agent_research_assessment_status::sufficient,
        "model assessment was not normalized");
    require(assessment_request.research_compare_evidence(assessment_gap, {evidence_a, evidence_b}, comparison, used, error) &&
        comparison.relation == common_agent_research_comparison_relation::conflict,
        "model comparison was not normalized");

    selection_inference multiple_model;
    multiple_model.replies = {R"({"sources":["OpenAlex","web"]})",
        "TOOLS: openalex", R"("openalex.listWorks")",
        R"({"tool":"openalex.listWorks","args":{"search":"urban cooling"}})"};
    auto multiple_inputs = inputs;
    multiple_inputs.orchestration_config.prompt = "Compare OpenAlex and web sources on urban cooling";
    auto multiple_execution = make_agent_runtime_driver_execution(multiple_inputs, multiple_model);
    const auto multiple_request = make_agent_runtime_driver_request(multiple_execution);
    require(multiple_request.research_select_tool(action, research, runtime, nullptr, {}, call, decision_message, error) && call &&
        call->name == "openalex.listWorks", "multiple explicit sources were not accepted");

    selection_inference invented_model;
    invented_model.replies = {R"({"sources":["OpenAlex"]})", "TOOLS: openalex", R"("openalex.getWork")",
        R"({"tool":"openalex.getWork","args":{"id":"W999"}})",
        R"({"tool":"openalex.listWorks","args":{"search":"urban cooling"}})"};
    auto invented_execution = make_agent_runtime_driver_execution(inputs, invented_model);
    const auto invented_request = make_agent_runtime_driver_request(invented_execution);
    auto invented_action = action;
    invented_action.instruction = "Inspect the model-suggested work W999";
    require(invented_request.research_select_tool(invented_action, research, runtime, nullptr, {}, call, decision_message, error) &&
        call && call->name == "openalex.listWorks", "invented identifier did not lead to a search");

    selection_inference observed_model;
    observed_model.replies = {R"({"sources":["OpenAlex"]})", "TOOLS: openalex", R"("openalex.getWork")",
        R"({"tool":"openalex.getWork","args":{"id":"W123"}})"};
    auto observed_execution = make_agent_runtime_driver_execution(inputs, observed_model);
    const auto observed_request = make_agent_runtime_driver_request(observed_execution);
    common_agent_research_source candidate;
    candidate.source_id = "candidate";
    candidate.origin = "https://openalex.org/W123";
    candidate.authority = "openalex";
    candidate.role = "api_discovery_candidate";
    research.sources.push_back(candidate);
    require(observed_request.research_select_tool(action, research, runtime, nullptr, {}, call, decision_message, error) &&
        call && call->name == "openalex.getWork", "observed candidate was not available for lookup");

    selection_inference declining_model;
    declining_model.replies = {R"({"sources":["OpenAlex"]})", "NO_TOOLS",
        R"({"decision":"clarify","message":"Which aspect should I investigate?"})"};
    auto declining_execution = make_agent_runtime_driver_execution(inputs, declining_model);
    const auto declining_request = make_agent_runtime_driver_request(declining_execution);
    require(declining_request.research_select_tool(action, research, runtime, nullptr, {}, call, decision_message, error) && !call &&
        decision_message.find("Which aspect") != std::string::npos,
        "research cannot stop without selecting a tool");
    selection_inference unavailable_model;
    unavailable_model.replies = {R"({"sources":["missing-provider"]})"};
    auto unavailable_inputs = inputs;
    unavailable_inputs.orchestration_config.prompt = "Search missing-provider for urban cooling";
    auto unavailable_execution = make_agent_runtime_driver_execution(unavailable_inputs, unavailable_model);
    const auto unavailable_request = make_agent_runtime_driver_request(unavailable_execution);
    auto unavailable_action = action;
    unavailable_action.gap_id = "unavailable-gap";
    auto unavailable_research = research;
    unavailable_research.gaps.push_back({"unavailable-gap", "Search missing-provider for urban cooling", "", "search", 1});
    require(!unavailable_request.research_select_tool(unavailable_action, unavailable_research, runtime, nullptr, {}, call, decision_message, error) &&
        error.find("source is unavailable") != std::string::npos && unavailable_model.requests.size() == 1,
        "unavailable explicit source was silently replaced");
    // A resumed task retains its original objective, but the newest user
    // request may explicitly choose a different registered source.
    auto followup_inputs = inputs;
    followup_inputs.orchestration_config.prompt = "Use web for the follow-up search";
    selection_inference followup_model;
    followup_model.replies = {R"({"sources":["web"]})",
        "TOOLS: web", R"({"tool":"web.search","args":{"query":"urban cooling"}})"};
    auto followup_execution = make_agent_runtime_driver_execution(followup_inputs, followup_model);
    const auto followup_request = make_agent_runtime_driver_request(followup_execution);
    require(followup_request.research_select_tool(action, research, runtime, nullptr, {}, call, decision_message, error) &&
        call && call->name == "web.search", "resumed research ignored the current user source");
    require(followup_model.requests.front().messages.back().content.find(followup_inputs.orchestration_config.prompt) != std::string::npos,
        "source selector did not receive the latest user request");

    // A generated gap must not turn a provider name into a user constraint.
    selection_inference generated_gap_model;
    generated_gap_model.replies = {R"({"sources":["missing-provider"]})"};
    auto generated_gap_execution = make_agent_runtime_driver_execution(inputs, generated_gap_model);
    const auto generated_gap_request = make_agent_runtime_driver_request(generated_gap_execution);
    require(!generated_gap_request.research_select_tool(unavailable_action, unavailable_research, runtime, nullptr, {}, call, decision_message, error) &&
        error.find("not grounded in the user request") != std::string::npos,
        "a generated gap was promoted into a user-named source");

    selection_inference substituted_model;
    substituted_model.replies = {R"({"sources":["web"]})"};
    auto substituted_execution = make_agent_runtime_driver_execution(inputs, substituted_model);
    const auto substituted_request = make_agent_runtime_driver_request(substituted_execution);
    require(!substituted_request.research_select_tool(action, research, runtime, nullptr, {}, call, decision_message, error) &&
        error.find("not grounded") != std::string::npos,
        "a model-selected alternative was silently treated as the named source");
    return 0;
}
