#include "agent/thinking/research/research-contract.h"
#include "agent/thinking/research/research-controller.h"
#include "agent/thinking/research/research-workspace-factory.h"
#include "agent/thinking/research/research-identity.h"
#include "agent/thinking/research/research-assessor.h"
#include "agent/thinking/research/research-workspace.h"
#include "agent/thinking/research/research-runner.h"
#include "agent/thinking/research/research-workspace-serialization.h"
#include "agent/input-resources.h"
#include "agent/context/openalex-abstract.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>

class selected_provider_runtime final : public common_agent_tool_runtime {
public:
    mutable int executions = 0;
    bool is_read_only(const std::string &) const override { return true; }
    bool is_policy_gated(const std::string &) const override { return false; }
    bool has_execution_envelope() const override { return true; }
    bool is_available(const std::string & name) const override {
        return name == "openalex.listWorks" || name == "openalex.getWork";
    }
    bool resolve_tool_name(const std::string & name, std::string & resolved,
            std::vector<std::string> & candidates) const override {
        candidates.clear(); resolved = name; return is_available(name);
    }
    bool validate(const common_agent_tool_call & call, std::string & error) const override {
        const auto args = nlohmann::ordered_json::parse(call.arguments_json, nullptr, false);
        if (!args.is_object() || (call.name == "openalex.listWorks" && args.value("search", std::string{}).size() > 16)) {
            error = "search is too long"; return false;
        }
        error.clear(); return true;
    }
    common_tool_execution_result execute(const common_agent_tool_call & call) const override {
        ++executions;
        if (call.name == "openalex.listWorks") {
            auto result = common_tool_execution_result::success(
                R"({"result":{"results":[{"id":"https://openalex.org/W123","display_name":"Urban cooling"}]}})", "candidate");
            result.http.provider = "openalex";
            return result;
        }
        common_runtime_resource_ref ref;
        ref.uri = "agent-resource://paper";
        auto result = common_tool_execution_result::success("{}", "record", {ref});
        result.http.provider = "openalex";
        return result;
    }
};

class terminal_failure_executor final : public common_agent_research_tool_executor {
public:
    bool execute(
            const common_agent_research_action & action,
            common_agent_research_workspace &,
            common_agent_research_event & event,
            std::string &) const override {
        event.type = common_agent_research_event_type::task_failed;
        event.task_id = action.task_id;
        event.gap_id = action.gap_id;
        event.failure_code = "tool.route_denied";
        event.failure_summary = "resource.read is outside the active envelope";
        return true;
    }
};

class discovery_progress_executor final : public common_agent_research_tool_executor {
public:
    mutable int calls = 0;
    bool execute(const common_agent_research_action & action, common_agent_research_workspace &,
            common_agent_research_event & event, std::string & error) const override {
        ++calls;
        event = {};
        event.type = common_agent_research_event_type::task_completed;
        event.task_id = action.task_id;
        event.gap_id = action.gap_id;
        event.discovered_sources = 1;
        error.clear();
        return true;
    }
};

class canonical_resource_tool_runtime final : public common_agent_tool_runtime {
public:
    bool is_read_only(const std::string &) const override { return true; }
    bool is_policy_gated(const std::string &) const override { return false; }
    bool has_execution_envelope() const override { return true; }
    bool resolve_tool_name(
            const std::string & requested,
            std::string & resolved,
            std::vector<std::string> & candidates) const override {
        candidates.clear();
        if (requested != "resource.read" && requested != "web.fetch") return false;
        resolved = requested;
        return true;
    }
    bool validate(const common_agent_tool_call & call, std::string & error) const override {
        last_validated_name = call.name;
        error.clear();
        return call.name == "resource.read" || call.name == "web.fetch";
    }
    common_tool_execution_result execute(const common_agent_tool_call & call) const override {
        last_executed_name = call.name;
        last_executed_arguments = call.arguments_json;
        auto result=common_tool_execution_result::success(
            R"({"ok":true,"result":{"content":"Host-read exact source sentence."}})",
            "Readable text resource materialized.", resource.uri.empty()?
                std::vector<common_runtime_resource_ref>{}:std::vector<common_runtime_resource_ref>{resource});
        if(call.name=="web.fetch")result.http.requested_url="https://example.test/paper.pdf?token=not-for-export";
        return result;
    }

    common_runtime_resource_ref resource;
    mutable std::string last_validated_name;
    mutable std::string last_executed_name;
    mutable std::string last_executed_arguments;
};

class web_search_candidate_runtime final : public common_agent_tool_runtime {
public:
    bool is_read_only(const std::string &) const override { return true; }
    bool is_policy_gated(const std::string &) const override { return false; }
    bool is_available(const std::string & name) const override { return name == "web.search"; }
    bool resolve_tool_name(const std::string & requested, std::string & resolved,
            std::vector<std::string> & candidates) const override {
        candidates.clear();
        if (requested != "web.search") return false;
        resolved = requested;
        return true;
    }
    bool validate(const common_agent_tool_call &, std::string & error) const override {
        error.clear();
        return true;
    }
    common_tool_execution_result execute(const common_agent_tool_call &) const override {
        return common_tool_execution_result::success("{\"results\":[{\"title\":\"NASA\",\"url\":\"https://nasa.example/paper\"}]}",
            "Web search returned 1 candidate(s).");
    }
};

class openalex_record_runtime final : public common_agent_tool_runtime {
public:
    mutable std::string arguments;
    bool is_read_only(const std::string &) const override { return true; }
    bool is_policy_gated(const std::string &) const override { return false; }
    bool resolve_tool_name(const std::string & name, std::string & resolved,
            std::vector<std::string> & candidates) const override {
        candidates.clear(); resolved = name; return name == "resource.read";
    }
    bool validate(const common_agent_tool_call &, std::string & error) const override {
        error.clear(); return true;
    }
    common_tool_execution_result execute(const common_agent_tool_call & call) const override {
        arguments = call.arguments_json;
        nlohmann::ordered_json record = {{"metadata", std::string(3000, 'x')},
            {"abstract_inverted_index", {{"Cooling", {0}}, {"observed", {1}}}}};
        nlohmann::ordered_json output = {{"content", record.dump()}};
        return common_tool_execution_result::success(output.dump(), "read");
    }
};

class paged_openalex_runtime final : public common_agent_tool_runtime {
public:
    const std::string record = nlohmann::ordered_json{{"metadata", std::string(40000, 'x')},
        {"abstract_inverted_index", {{"Cooling", {0}}, {"observed", {1}}}}}.dump();
    mutable size_t calls = 0;
    bool is_read_only(const std::string &) const override { return true; }
    bool is_policy_gated(const std::string &) const override { return false; }
    bool resolve_tool_name(const std::string & name, std::string & resolved,
            std::vector<std::string> & candidates) const override {
        candidates.clear(); resolved = name; return name == "resource.read";
    }
    bool validate(const common_agent_tool_call &, std::string & error) const override { error.clear(); return true; }
    common_tool_execution_result execute(const common_agent_tool_call & call) const override {
        ++calls;
        const auto arguments = nlohmann::ordered_json::parse(call.arguments_json);
        const size_t offset = arguments.value("offset", size_t(0));
        const size_t requested = std::min(arguments.value("max_bytes", size_t(8192)), record.size() - offset);
        const size_t count = std::min<size_t>(8192, requested);
        const size_t remaining_resource = record.size() - offset - count;
        nlohmann::ordered_json range = {{"offset", offset}, {"bytes_read", count},
            {"has_more", count < requested}, {"remaining_bytes", requested - count},
            {"resource_has_more", remaining_resource > 0}, {"resource_complete", remaining_resource == 0}};
        if (remaining_resource) range["next_offset"] = offset + count;
        return common_tool_execution_result::success(nlohmann::ordered_json{
            {"content", record.substr(offset, count)}, {"range", range}}.dump(), "read");
    }
};

class paged_text_runtime final : public common_agent_tool_runtime {
public:
    const std::string text = "Unrelated opening. More context. Target evidence. Closing sentence.";
    mutable size_t calls = 0;
    bool is_read_only(const std::string &) const override { return true; }
    bool is_policy_gated(const std::string &) const override { return false; }
    bool resolve_tool_name(const std::string & name, std::string & resolved,
            std::vector<std::string> & candidates) const override {
        candidates.clear(); resolved = name; return name == "resource.read";
    }
    bool validate(const common_agent_tool_call &, std::string & error) const override { error.clear(); return true; }
    common_tool_execution_result execute(const common_agent_tool_call & call) const override {
        ++calls;
        const auto args = nlohmann::ordered_json::parse(call.arguments_json);
        const size_t offset = args.value("offset", size_t(0));
        const size_t count = std::min(args.value("max_bytes", size_t(8192)), text.size() - offset);
        common_runtime_resource_ref ref;
        ref.uri = "agent-resource://text";
        ref.size_bytes = text.size();
        ref.mime_type = "text/plain";
        return common_tool_execution_result::success(nlohmann::ordered_json{
            {"resource", {{"sha256", "0123456789abcdef"}}}, {"content", text.substr(offset, count)}, {"range", {{"offset", offset}, {"bytes_read", count},
                {"resource_complete", offset + count == text.size()}}}}.dump(), "read", {ref});
    }
};

int main() {
    {
        common_resource_continuation_position position;
        std::string validation_error;
        assert(!common_resource_continuation_position_valid(position, validation_error));
        position.resource_uri = "resource://versioned";
        position.representation = "text";
        position.content_fingerprint = "sha256:content";
        position.task_fingerprint = "task-fingerprint";
        position.next_offset = 64;
        assert(common_resource_continuation_position_valid(position, validation_error));
        position.status = common_resource_progress_status::complete;
        assert(!common_resource_continuation_position_valid(position, validation_error));
        position.more = common_resource_more::no;
        assert(common_resource_continuation_position_valid(position, validation_error));
        position.unit = common_resource_position_unit::provider_cursor;
        position.cursor_ref = "host-cursor://opaque";
        assert(!common_resource_continuation_position_valid(position, validation_error));
        position.next_offset = 0;
        assert(common_resource_continuation_position_valid(position, validation_error));
    }
    {
        const auto index = nlohmann::ordered_json::parse(R"({"study":[1],"A":[0],"works":[2]})");
        bool truncated = false;
        assert(common_agent_openalex_abstract(index, truncated) == "A study works" && !truncated);
    }
    common_agent_scope research_scope_a;
    research_scope_a.namespace_id = "research-test";
    research_scope_a.project_id = "project-shared";
    research_scope_a.session_id = "chat-a";
    common_agent_scope research_scope_b = research_scope_a;
    research_scope_b.session_id = "chat-b";
    const auto project_research_id = common_agent_research_identity(research_scope_a);
    assert(project_research_id == common_agent_research_identity(research_scope_b));
    assert(common_agent_research_draft_identity(project_research_id, "chat-a") !=
        common_agent_research_draft_identity(project_research_id, "chat-b"));
    research_scope_b.project_id.clear();
    assert(project_research_id != common_agent_research_identity(research_scope_b));

    common_agent_request factory_request;
    factory_request.prompt = "Research the bounded runtime contract";
    factory_request.turn_id = "factory-turn";
    factory_request.session_id = "factory-session";
    factory_request.namespace_id = "factory-namespace";
    factory_request.max_iterations = 3;
    factory_request.max_tool_batches = 2;
    factory_request.deliberation_policy = make_common_agent_deliberation_policy(common_agent_thinking_mode::research);
    common_agent_input_resource input_resource;
    input_resource.resource.uri = "agent-resource://factory-turn/upload-1";
    input_resource.resource.name = "notes.txt";
    input_resource.resource.mime_type = "text/plain";
    input_resource.role = "reference";
    input_resource.required = true;
    factory_request.input_resources.push_back(input_resource);
    common_memory_hit memory_hit;
    memory_hit.memory.id = "memory-procedure-1";
    memory_hit.memory.kind = common_memory_kind::procedure;
    memory_hit.memory.summary = "Use the bounded runtime contract first.";
    memory_hit.memory.confidence = 0.9f;
    memory_hit.final_score = 0.9f;
    factory_request.memories.push_back(memory_hit);
    common_agent_research_workspace factory_workspace;
    std::string factory_error;
    assert(common_agent_research_create_workspace(factory_request, factory_workspace, factory_error));
    assert(factory_workspace.workspace_id == "research:factory-turn");
    assert(factory_workspace.research_id == factory_workspace.workspace_id && factory_workspace.draft_id.empty());
    assert(factory_workspace.gaps.size() == 1);
    assert(factory_workspace.budget.max_iterations == 4);
    assert(factory_workspace.budget.max_tool_calls == 2);
    assert(factory_workspace.sources.size() == 2);
    common_agent_research_draft_ref draft_ref;
    draft_ref.research_id = "research-project-1";
    draft_ref.draft_id = "draft-chat-1";
    draft_ref.owner_session_id = "factory-session";
    draft_ref.base_revision = 3;
    draft_ref.draft_revision = 2;
    auto persisted_seed = factory_workspace;
    persisted_seed.research_id = draft_ref.research_id;
    persisted_seed.draft_id = draft_ref.draft_id;
    persisted_seed.base_revision = draft_ref.base_revision;
    persisted_seed.revision = draft_ref.draft_revision;
    persisted_seed.iterations_completed = persisted_seed.budget.max_iterations;
    persisted_seed.tool_calls = persisted_seed.budget.max_tool_calls;
    persisted_seed.no_progress_iterations = 2;
    persisted_seed.sources.front().source_id = "persisted-source";
    const auto old_workspace_id = persisted_seed.workspace_id;
    common_agent_request resumed_request = factory_request;
    resumed_request.turn_id = "factory-followup";
    resumed_request.prompt = "Check the follow-up question";
    resumed_request.research_draft = draft_ref;
    resumed_request.research_workspace_seed = persisted_seed;
    resumed_request.objective.reset();
    common_agent_research_workspace resumed_workspace;
    assert(common_agent_research_create_workspace(resumed_request, resumed_workspace, factory_error));
    assert(resumed_workspace.research_id == draft_ref.research_id && resumed_workspace.draft_id == draft_ref.draft_id);
    assert(resumed_workspace.workspace_id != old_workspace_id && resumed_workspace.revision == draft_ref.draft_revision);
    assert(resumed_workspace.objective.question == persisted_seed.objective.question);
    assert(resumed_workspace.iterations_completed == persisted_seed.iterations_completed &&
        resumed_workspace.budget.max_iterations == persisted_seed.iterations_completed + 4);
    assert(resumed_workspace.tool_calls == persisted_seed.tool_calls &&
        resumed_workspace.budget.max_tool_calls == persisted_seed.tool_calls + 2);
    assert(resumed_workspace.budget.max_sources == static_cast<int>(common_agent_research_source_count(persisted_seed)) + 2);
    assert(resumed_workspace.budget.max_candidates == static_cast<int>(common_agent_research_candidate_count(persisted_seed)) + 2);
    assert(resumed_workspace.no_progress_iterations == 0);
    assert(std::any_of(resumed_workspace.sources.begin(), resumed_workspace.sources.end(), [](const auto & source) {
        return source.source_id == "persisted-source";
    }));
    assert(std::any_of(resumed_workspace.gaps.begin(), resumed_workspace.gaps.end(), [](const auto & gap) {
        return gap.question == "Check the follow-up question";
    }));
    assert(factory_workspace.sources.front().kind == common_agent_research_source_kind::user_supplied);
    assert(factory_workspace.sources.front().resource_ref &&
        factory_workspace.sources.front().resource_ref->uri == input_resource.resource.uri);
    assert(factory_workspace.sources.front().role == "reference" &&
        factory_workspace.sources.front().required &&
        !factory_workspace.sources.front().primary_source &&
        factory_workspace.sources.front().content_hash.empty());
    assert(factory_workspace.sources.back().kind == common_agent_research_source_kind::memory);
    assert(factory_workspace.sources.back().memory_id == memory_hit.memory.id);
    assert(factory_workspace.sources.back().memory_kind == common_memory_kind::procedure);
    assert(factory_workspace.sources.back().availability ==
        common_agent_research_source_availability::materialized_context);
    assert(factory_workspace.sources.back().role == "method_context");

    common_agent_research_workspace candidate_budget_workspace;
    candidate_budget_workspace.budget.max_candidates = 5;
    candidate_budget_workspace.budget.max_sources = 1;
    for (int i = 0; i < 5; ++i) {
        common_agent_research_source candidate;
        candidate.source_id = "candidate-" + std::to_string(i);
        candidate.title = candidate.source_id;
        candidate.role = "api_discovery_candidate";
        assert(common_agent_research_add_source(candidate_budget_workspace, candidate, factory_error));
    }
    assert(common_agent_research_source_count(candidate_budget_workspace) == 0);
    assert(common_agent_research_candidate_count(candidate_budget_workspace) == 5);
    common_runtime_resource_ref candidate_resource;
    candidate_resource.uri = "agent-resource://candidate-one";
    assert(common_agent_research_materialize_candidate(candidate_budget_workspace,
        candidate_budget_workspace.sources[0], candidate_resource, factory_error));
    assert(common_agent_research_source_count(candidate_budget_workspace) == 1);
    assert(common_agent_research_candidate_count(candidate_budget_workspace) == 4);
    assert(!common_agent_research_materialize_candidate(candidate_budget_workspace,
        candidate_budget_workspace.sources[1], candidate_resource, factory_error));
    assert(candidate_budget_workspace.sources[1].role == "api_discovery_candidate" &&
        !candidate_budget_workspace.sources[1].resource_ref);

    auto draft_request = factory_request;
    draft_request.research_draft = common_agent_research_draft_ref{
        "research:shared", "draft:factory-session", "factory-session", 3, 2,
        "agent-resource://project/draft", "sha256:draft", 100};
    common_agent_research_workspace draft_workspace;
    assert(common_agent_research_create_workspace(draft_request, draft_workspace, factory_error));
    assert(draft_workspace.research_id == "research:shared" &&
        draft_workspace.draft_id == "draft:factory-session" && draft_workspace.base_revision == 3);
    draft_workspace.sources.front().resource_ref->metadata.content_hash = "sha256:source";
    draft_workspace.gaps.front().assessment = common_agent_research_gap_assessment::inconclusive;
    draft_workspace.generation_tokens_reserved = 384;
    const auto workspace_json = common_agent_research_workspace_to_json(draft_workspace);
    common_agent_research_workspace restored_workspace;
    assert(common_agent_research_workspace_from_json(workspace_json, restored_workspace, factory_error));
    assert(restored_workspace.research_id == draft_workspace.research_id &&
        restored_workspace.draft_id == draft_workspace.draft_id && restored_workspace.base_revision == 3 &&
        restored_workspace.generation_tokens_reserved == 384);
    assert(restored_workspace.objective.success_criteria == draft_workspace.objective.success_criteria &&
        restored_workspace.sources.front().resource_ref->metadata.content_hash == "sha256:source" &&
        restored_workspace.gaps.front().assessment == common_agent_research_gap_assessment::inconclusive);

    common_agent_research_workspace chunk_workspace;
    chunk_workspace.workspace_id = "research:chunk-turn";
    chunk_workspace.objective.objective_id = "chunk-objective";
    chunk_workspace.objective.question = "What result did the paper report?";
    chunk_workspace.scope.namespace_id = "chunk-namespace";
    chunk_workspace.scope.session_id = "chunk-session";
    chunk_workspace.budget.max_sources = 4;
    chunk_workspace.gaps.push_back({"chunk-gap", "What result did the paper report?", "", "the question", 1});
    common_runtime_resource_ref chunk_ref;
    chunk_ref.uri = "agent-resource://chunk-turn/paper-text";
    chunk_ref.name = "paper-text";
    chunk_ref.lineage = {"agent-resource://chunk-turn/paper-text", 2, 3, 4096, 4096, 256,
        "host.chunk.text.v1"};
    std::vector<common_plan_observation> chunk_observations{
        {"chunk-2", "resource_chunk",
            "Validated resource chunk evidence.\nfinding: Radiation increased.\nevidence: radiation flux increased",
            1.0f, {}, {chunk_ref}, 0}};
    assert(common_agent_research_seed_chunk_observations(
        chunk_workspace, chunk_observations, factory_error));
    assert(chunk_workspace.sources.size() == 1 && chunk_workspace.evidence.size() == 1);
    assert(chunk_workspace.evidence.front().origin == common_agent_research_evidence_origin::model_inference);
    assert(chunk_workspace.evidence.front().model_inferred);
    assert(chunk_workspace.evidence.front().statement == "Radiation increased.");
    assert(chunk_workspace.evidence.front().supporting_excerpt == "radiation flux increased");
    assert(chunk_workspace.evidence.front().supporting_excerpt_host_verified);
    assert(chunk_workspace.evidence.front().source_location.find("#bytes=4096+4096") != std::string::npos);
    std::vector<common_agent_research_source> projected_sources;
    std::vector<common_agent_research_evidence> projected_evidence;
    assert(common_agent_research_chunk_evidence_from_observations(
        chunk_observations, "task:projection", projected_sources, projected_evidence, factory_error));
    assert(projected_sources.size() == 1 && projected_evidence.size() == 1);
    assert(projected_evidence.front().claim_id == "task:projection");
    assert(projected_evidence.front().supporting_excerpt_host_verified);
    assert(projected_evidence.front().source_location == chunk_workspace.evidence.front().source_location);
    auto earlier_chunk = chunk_observations.front();
    earlier_chunk.id = "chunk-1-with-finding";
    earlier_chunk.resource_refs.front().lineage.chunk_index = 1;
    earlier_chunk.resource_refs.front().lineage.byte_offset = 2048;
    assert(common_agent_research_chunk_evidence_from_observations(
        {earlier_chunk, chunk_observations.front()}, "task:projection", projected_sources, projected_evidence, factory_error));
    assert(projected_evidence.size() == 2 && projected_evidence[0].evidence_id != projected_evidence[1].evidence_id);
    auto no_finding_observation = chunk_observations.front();
    no_finding_observation.id = "chunk-1";
    no_finding_observation.summary = "no relevant findings.";
    no_finding_observation.resource_refs.front().uri = "agent-resource://chunk-turn/empty-chunk";
    no_finding_observation.resource_refs.front().lineage.chunk_index = 1;
    assert(common_agent_research_chunk_evidence_from_observations(
        {no_finding_observation}, "task:projection", projected_sources, projected_evidence, factory_error));
    assert(projected_sources.empty() && projected_evidence.empty());
    assert(chunk_workspace.gaps.front().evidence_ids.size() == 1);
    assert(chunk_workspace.gaps.front().status == common_agent_research_gap_status::open);
    common_agent_research_bounded_assessor bounded_assessor;
    assert(common_agent_research_assess_materialized_evidence(
        chunk_workspace, bounded_assessor, factory_error));
    // A verified excerpt is not by itself a semantic assessment of the gap.
    assert(chunk_workspace.gaps.front().status == common_agent_research_gap_status::open);

    auto ambiguous_chunk_workspace = chunk_workspace;
    ambiguous_chunk_workspace.sources.clear();
    ambiguous_chunk_workspace.evidence.clear();
    ambiguous_chunk_workspace.gaps.front().status = common_agent_research_gap_status::open;
    ambiguous_chunk_workspace.gaps.front().evidence_ids.clear();
    ambiguous_chunk_workspace.gaps.push_back(
        {"chunk-gap-2", "What method was used?", "", "", 1});
    assert(common_agent_research_seed_chunk_observations(
        ambiguous_chunk_workspace, chunk_observations, factory_error));
    assert(ambiguous_chunk_workspace.sources.empty() && ambiguous_chunk_workspace.evidence.empty());
    assert(ambiguous_chunk_workspace.gaps.front().status == common_agent_research_gap_status::open);

    const auto input_context = common_agent_render_input_resource_context(factory_request.input_resources);
    assert(input_context.find("id=r1") != std::string::npos);
    assert(input_context.find("agent-resource://factory-turn/upload-1") == std::string::npos);
    // Workflow-generated API resources enter the same bounded research
    // acquisition path without being mislabeled as user uploads.
    factory_workspace.sources.front().kind = common_agent_research_source_kind::api_result;
    common_agent_research_controller factory_controller;
    const auto input_action = factory_controller.begin(factory_workspace, factory_error);
    assert(input_action.kind == common_agent_research_action_kind::schedule_task);
    assert(input_action.preferred_tools.size() == 1 && input_action.preferred_tools.front() == "resource.read");
    assert(input_action.instruction == input_resource.resource.uri);

    // The research controller keeps its historical underscore alias, but the
    // route-scoped runtime requires the canonical catalog name.
    canonical_resource_tool_runtime canonical_runtime;
    canonical_runtime.resource = input_resource.resource;
    common_agent_research_runtime_adapter canonical_adapter(canonical_runtime);
    auto canonical_workspace = factory_workspace;
    common_agent_research_action canonical_action;
    canonical_action.kind = common_agent_research_action_kind::schedule_task;
    canonical_action.task_id = "alias-task";
    canonical_action.gap_id = canonical_workspace.gaps.front().gap_id;
    canonical_action.instruction = input_resource.resource.uri;
    canonical_action.preferred_tools = {"resource.read"};
    common_agent_research_event canonical_event;
    assert(canonical_adapter.execute(canonical_action, canonical_workspace, canonical_event, factory_error));
    assert(canonical_runtime.last_validated_name == "resource.read");
    assert(canonical_runtime.last_executed_name == "resource.read");
    assert(canonical_workspace.evidence.size()==1);
    assert(canonical_workspace.evidence.front().statement=="Host-read exact source sentence.");
    assert(canonical_workspace.evidence.front().origin==common_agent_research_evidence_origin::direct_source);
    assert(canonical_workspace.evidence.front().supporting_excerpt_host_verified);
    assert(canonical_event.type == common_agent_research_event_type::task_completed);

    canonical_action.task_id = "fetch-task";
    canonical_runtime.resource.uri="agent-resource://new-fetch";
    canonical_runtime.resource.metadata.content_hash="sha256:source-content";
    canonical_runtime.resource.metadata.retrieved_at=1234;
    canonical_action.instruction = "https://example.test/paper.pdf";
    canonical_action.preferred_tools = {"web.fetch"};
    assert(canonical_adapter.execute(canonical_action, canonical_workspace, canonical_event, factory_error));
    assert(canonical_runtime.last_validated_name == "web.fetch");
    assert(canonical_runtime.last_executed_name == "web.fetch");
    const auto research_fetch_arguments = nlohmann::json::parse(canonical_runtime.last_executed_arguments);
    assert(research_fetch_arguments.at("url") == "https://example.test/paper.pdf");
    assert(!research_fetch_arguments.contains("max_bytes"));
    assert(canonical_workspace.sources.back().origin=="https://example.test/paper.pdf");
    assert(canonical_workspace.sources.back().content_hash=="sha256:source-content");
    assert(canonical_workspace.sources.back().retrieved_at=="1234");

    // Different requests returning identical inline content must have the
    // same content fingerprint, rather than manufacturing source diversity.
    auto inline_workspace=factory_workspace;
    inline_workspace.sources.clear();inline_workspace.evidence.clear();
    inline_workspace.gaps.front().evidence_ids.clear();
    canonical_runtime.resource={};
    canonical_action.task_id="inline-first";
    assert(canonical_adapter.execute(canonical_action,inline_workspace,canonical_event,factory_error));
    canonical_action.task_id="inline-second";
    canonical_action.instruction="https://example.test/other-request";
    assert(canonical_adapter.execute(canonical_action,inline_workspace,canonical_event,factory_error));
    assert(inline_workspace.sources.size()==2);
    assert(inline_workspace.sources[0].content_hash.rfind("sha256:",0)==0);
    assert(inline_workspace.sources[0].content_hash==inline_workspace.sources[1].content_hash);

    factory_workspace.gaps.front().status = common_agent_research_gap_status::open;
    factory_workspace.tasks.clear();
    factory_workspace.evidence.clear();
    factory_workspace.coverage = {};
    factory_workspace.sources.front().resource_ref.reset();
    const auto memory_action = factory_controller.begin(factory_workspace, factory_error);
    assert(memory_action.kind == common_agent_research_action_kind::schedule_task);
    assert(std::find(memory_action.preferred_tools.begin(), memory_action.preferred_tools.end(),
        "resource.read") == memory_action.preferred_tools.end());
    assert(std::find(memory_action.preferred_tools.begin(), memory_action.preferred_tools.end(),
        "memory.get") == memory_action.preferred_tools.end());
    factory_workspace.tasks.clear();
    factory_workspace.gaps.front().status = common_agent_research_gap_status::open;
    factory_workspace.sources.back().role = "memory_context";
    const auto scoped_memory_action = factory_controller.begin(factory_workspace, factory_error);
    assert(std::find(scoped_memory_action.preferred_tools.begin(), scoped_memory_action.preferred_tools.end(),
        "memory.get") != scoped_memory_action.preferred_tools.end());
    assert(scoped_memory_action.instruction == factory_workspace.sources.back().memory_id);

    common_agent_research_workspace workspace;
    workspace.workspace_id = "research-1";
    workspace.request_id = "request-1";
    workspace.turn_id = "turn-1";
    workspace.session_id = "session-1";
    workspace.plan_id = "plan-1";
    workspace.plan_step_id = "research-step";
    workspace.scope.namespace_id = "project";
    workspace.scope.session_id = "session-1";
    workspace.objective.objective_id = "objective-1";
    workspace.objective.question = "What scheduling gaps remain?";
    workspace.objective.success_criteria = {"claims are source-backed"};
    std::string error;
    assert(common_agent_research_add_gap(workspace, {"gap-1", "Is cancellation bounded?", "The runtime contract is unclear.", "evidence establishes whether cancellation is bounded", 1}, error));
    assert(common_agent_research_add_task(workspace, {"task-1", "gap-1", "Find cancellation paths", common_agent_research_task_kind::repository_inspection, {"repository.search"}, {}, 1, 0, 1}, error));
    common_runtime_resource_ref resource;
    resource.uri = "agent-resource://turn-1/source-1";
    resource.scope = common_runtime_resource_scope::turn;
    assert(common_agent_research_add_source(workspace, {"source-1", "runtime-operation.h", "repository", "repository", "now", "hash-1", common_agent_research_source_kind::repository_file, resource, 0.95, true}, error));
    common_agent_research_evidence evidence;
    evidence.evidence_id = "evidence-1";
    evidence.source_id = "source-1";
    evidence.claim_id = "claim-1";
    evidence.statement = "Cancellation is represented by a bounded operation state.";
    evidence.source_location = "runtime-operation.h:42";
    evidence.relation = common_agent_research_evidence_relation::supports;
    evidence.origin = common_agent_research_evidence_origin::direct_source;
    evidence.relevance = 0.95;
    evidence.confidence = 0.9;
    evidence.directly_observed = true;
    assert(common_agent_research_record_evidence(workspace, evidence, error));
    assert(common_agent_research_update_coverage(workspace, {1, 1, 0, 0, 1.0, 0.9, 1.0}, error));
    assert(common_agent_research_workspace_validate(workspace, error));
    auto invalid_verified_excerpt = workspace;
    invalid_verified_excerpt.evidence.front().supporting_excerpt_host_verified = true;
    assert(!common_agent_research_workspace_validate(invalid_verified_excerpt, error));
    assert(error.find("invalid or duplicate evidence") != std::string::npos);
    auto invalid_dependency_workspace = workspace;
    invalid_dependency_workspace.tasks.push_back({
        "task-2", "gap-1", "Use the first result", common_agent_research_task_kind::fetch,
        {"resource.read"}, {"missing-task"}, 1, 0, 1});
    assert(!common_agent_research_workspace_validate(invalid_dependency_workspace, error));
    assert(error.find("invalid task dependency") != std::string::npos);
    assert(!common_agent_research_record_evidence(workspace, workspace.evidence.front(), error));
    assert(error.find("already exists") != std::string::npos);
    workspace.evidence.push_back(workspace.evidence.front());
    assert(!common_agent_research_workspace_validate(workspace, error));
    assert(error.find("duplicate evidence") != std::string::npos);

    common_agent_research_controller controller;
    workspace.evidence.pop_back();
    workspace.gaps.front().status = common_agent_research_gap_status::open;
    workspace.tasks.clear();
    workspace.coverage = {};
    const auto first_action = controller.begin(workspace, error);
    assert(first_action.kind == common_agent_research_action_kind::schedule_task);
    const auto final_action = controller.advance(workspace, {
        common_agent_research_event_type::task_completed,
        first_action.task_id,
        first_action.gap_id,
        1,
        {"evidence-1"},
        true,
        0.9,
        "Evidence assessed against the criterion",
        {common_agent_research_assessment_status::sufficient, 0.9,
            "Evidence assessed against the criterion"}}, error);
    assert(final_action.kind == common_agent_research_action_kind::complete);
    assert(final_action.stop_reason == common_agent_research_stop_reason::sufficient_coverage);
    assert(workspace.gaps.front().assessment == common_agent_research_gap_assessment::sufficient);
    const auto research_result = controller.finalize(workspace);
    assert(research_result.complete && research_result.workspace_id == "research-1");
    assert(research_result.established_claim_ids.size() == 1);
    assert(research_result.established_claim_ids.front() == "gap-1");
    assert(research_result.sources.size() == 1 && research_result.evidence.size() == 1);

    // Two independent records do not imply agreement. Even a sufficient
    // assessment of one finding must not fabricate a source comparison.
    auto two_source_workspace = workspace;
    two_source_workspace.gaps.front().status = common_agent_research_gap_status::open;
    two_source_workspace.gaps.front().evidence_ids.clear();
    two_source_workspace.tasks.clear();
    two_source_workspace.coverage = {};
    two_source_workspace.budget.minimum_sources = 2;
    auto second_source = two_source_workspace.sources.front();
    second_source.source_id = "source-2";
    second_source.content_hash = "hash-2";
    assert(common_agent_research_add_source(two_source_workspace, second_source, error));
    auto second_evidence = two_source_workspace.evidence.front();
    second_evidence.evidence_id = "evidence-2";
    second_evidence.source_id = "source-2";
    assert(common_agent_research_record_evidence(two_source_workspace, second_evidence, error));
    const auto two_source_start = controller.begin(two_source_workspace, error);
    assert(two_source_start.kind == common_agent_research_action_kind::schedule_task);
    controller.advance(two_source_workspace, {
        common_agent_research_event_type::task_completed, two_source_start.task_id,
        two_source_start.gap_id, 2, {"evidence-1", "evidence-2"}, true, 0.9,
        "One finding was assessed", {common_agent_research_assessment_status::sufficient,
            0.9, "One finding was assessed"}}, error);
    assert(two_source_workspace.comparisons.empty());
    assert(two_source_workspace.gaps.front().status != common_agent_research_gap_status::sufficiently_answered);

    auto compared_workspace = two_source_workspace;
    compared_workspace.gaps.front().status = common_agent_research_gap_status::open;
    compared_workspace.gaps.front().evidence_ids.clear();
    compared_workspace.tasks.clear();
    compared_workspace.coverage = {};
    common_agent_research_bounded_assessor comparing_assessor({},
        [](const common_agent_research_gap &, const std::vector<common_agent_research_evidence> & items,
                common_agent_research_source_comparison & comparison) {
            assert(items.size() == 2);
            comparison.relation = common_agent_research_comparison_relation::conflict;
            comparison.summary = "The findings conflict.";
            return true;
        });
    common_agent_research_controller comparing_controller(&comparing_assessor);
    const auto comparing_start = comparing_controller.begin(compared_workspace, error);
    comparing_controller.advance(compared_workspace, {
        common_agent_research_event_type::task_completed, comparing_start.task_id,
        comparing_start.gap_id, 2, {"evidence-1", "evidence-2"}, true, 0.9,
        "Evidence assessed", {common_agent_research_assessment_status::sufficient, 0.9, "Evidence assessed"}}, error);
    assert(compared_workspace.comparisons.size() == 1 &&
        compared_workspace.comparisons.front().relation == common_agent_research_comparison_relation::conflict);
    assert(compared_workspace.comparisons.front().source_ids.size() == 2);

    auto requested_sources_workspace = workspace;
    requested_sources_workspace.gaps.front().status = common_agent_research_gap_status::open;
    requested_sources_workspace.tasks.clear();
    requested_sources_workspace.source_requirements = {{"gap-1", "source-a", true}, {"gap-1", "source-b", false}};
    common_agent_research_bounded_assessor accepting_assessor(
        [](const common_agent_research_gap &, const common_agent_research_evidence &) {
            return common_agent_research_assessment{common_agent_research_assessment_status::sufficient, 0.9, "Relevant finding."};
        });
    assert(common_agent_research_assess_materialized_evidence(requested_sources_workspace, accepting_assessor, error));
    assert(requested_sources_workspace.gaps.front().status == common_agent_research_gap_status::open);
    requested_sources_workspace.source_requirements.back().searched = true;
    assert(common_agent_research_assess_materialized_evidence(requested_sources_workspace, accepting_assessor, error));
    assert(requested_sources_workspace.gaps.front().status == common_agent_research_gap_status::sufficiently_answered);
    common_agent_research_workspace source_requirement_roundtrip;
    assert(common_agent_research_workspace_from_json(common_agent_research_workspace_to_json(requested_sources_workspace),
        source_requirement_roundtrip, error));
    assert(source_requirement_roundtrip.source_requirements.size() == 2 &&
        source_requirement_roundtrip.source_requirements.back().searched);

    auto assessment_workspace = workspace;
    assessment_workspace.gaps.front().status = common_agent_research_gap_status::open;
    assessment_workspace.gaps.front().evidence_ids.clear();
    assessment_workspace.tasks.clear();
    assessment_workspace.coverage = {};
    const auto assessment_start = controller.begin(assessment_workspace, error);
    assert(assessment_start.kind == common_agent_research_action_kind::schedule_task);
    const auto assessment_action = controller.advance(assessment_workspace, {
        common_agent_research_event_type::task_completed,
        assessment_start.task_id,
        assessment_start.gap_id,
        1,
        {"evidence-1"},
        true,
        0.9,
        "tool completed but criterion was not assessed",
        {common_agent_research_assessment_status::insufficient, 0.2,
            "evidence is not sufficient for the completion criterion"}}, error);
    assert(assessment_action.kind == common_agent_research_action_kind::schedule_task);
    assert(assessment_workspace.gaps.front().status == common_agent_research_gap_status::investigating);

    common_agent_research_bounded_assessor assessor;
    const auto sufficient_assessment = assessor.assess(
        {"gap", "", "", "repository result", 1},
        {"evidence", "source", "gap", "repository search returned a match", "repository", common_agent_research_evidence_relation::supports,
            common_agent_research_evidence_origin::normalized_tool_result, 0.8, 0.9});
    assert(sufficient_assessment.status == common_agent_research_assessment_status::inconclusive);
    const auto insufficient_assessment = assessor.assess(
        {"gap", "", "", "web result", 1},
        {"evidence-2", "source", "gap", "repository search returned a match", "repository", common_agent_research_evidence_relation::supports,
            common_agent_research_evidence_origin::normalized_tool_result, 0.8, 0.9});
    assert(insufficient_assessment.status == common_agent_research_assessment_status::inconclusive);

    auto criteria_request = factory_request;
    criteria_request.input_resources.clear();
    criteria_request.memories.clear();
    criteria_request.objective = common_agent_objective{};
    criteria_request.objective->success_criteria = {"verify source", "record limitation"};
    common_agent_research_workspace criteria_workspace;
    assert(common_agent_research_create_workspace(criteria_request, criteria_workspace, factory_error));
    assert(criteria_workspace.gaps.size() == 2);
    assert(criteria_workspace.gaps[0].completion_criterion == "verify source");
    assert(criteria_workspace.gaps[1].completion_criterion == "record limitation");

    const auto cancelled_action = controller.advance(workspace, {
        common_agent_research_event_type::cancelled,
        {}, {}, 0, {}}, error);
    assert(cancelled_action.kind == common_agent_research_action_kind::complete);
    assert(cancelled_action.stop_reason == common_agent_research_stop_reason::cancelled);

    auto budget_limited_workspace = workspace;
    budget_limited_workspace.budget.max_iterations = 0;
    const auto budget_action = controller.begin(budget_limited_workspace, error);
    assert(budget_action.kind == common_agent_research_action_kind::complete);
    assert(budget_action.stop_reason == common_agent_research_stop_reason::budget_exhausted);

    auto retry_workspace = workspace;
    retry_workspace.gaps.front().status = common_agent_research_gap_status::open;
    retry_workspace.tasks.clear();
    retry_workspace.evidence.clear();
    retry_workspace.coverage = {};
    retry_workspace.budget.max_iterations = 2;
    const auto retry_start = controller.begin(retry_workspace, error);
    assert(retry_start.kind == common_agent_research_action_kind::schedule_task);
    common_agent_research_event retry_event;
    retry_event.type = common_agent_research_event_type::task_failed;
    retry_event.task_id = retry_start.task_id;
    retry_event.gap_id = retry_start.gap_id;
    retry_event.retryable = true;
    const auto retry_action = controller.advance(retry_workspace, retry_event, error);
    assert(retry_action.kind == common_agent_research_action_kind::schedule_task);
    assert(retry_action.task_id != retry_start.task_id);
    assert(retry_action.instruction != retry_start.instruction);
    assert(retry_action.preferred_tools != retry_start.preferred_tools);

    auto failed_workspace = factory_workspace;
    failed_workspace.gaps.front().status = common_agent_research_gap_status::open;
    failed_workspace.gaps.front().evidence_ids.clear();
    failed_workspace.tasks.clear();
    failed_workspace.evidence.clear();
    failed_workspace.budget.max_iterations = 1;
    failed_workspace.budget.max_tool_calls = 1;
    failed_workspace.no_progress_iterations = 0;
    failed_workspace.coverage = {};
    terminal_failure_executor failure_executor;
    common_agent_research_runner research_runner;
    std::string run_error;
    const auto failed_result = research_runner.run(
        failed_workspace, failure_executor, run_error);
    assert(!failed_result.complete);
    assert(run_error.find("tool.route_denied") != std::string::npos);
    assert(run_error.find("outside the active envelope") != std::string::npos);

    // A resumed workspace's iteration limit is a total ceiling, including
    // completed work from prior turns, not a fresh per-run allowance.
    auto resumed_budget = factory_workspace;
    resumed_budget.gaps.front().status = common_agent_research_gap_status::open;
    resumed_budget.tasks.clear();
    resumed_budget.evidence.clear();
    resumed_budget.iterations_completed = 2;
    resumed_budget.tool_calls = 0;
    resumed_budget.budget.max_iterations = 3;
    resumed_budget.budget.max_tool_calls = 8;
    resumed_budget.no_progress_iterations = 0;
    resumed_budget.coverage = {};
    discovery_progress_executor progress_executor;
    run_error.clear();
    const auto resumed_result = research_runner.run(resumed_budget, progress_executor, run_error);
    assert(run_error.empty() && !resumed_result.complete &&
        resumed_result.stop_reason == common_agent_research_stop_reason::budget_exhausted);
    assert(progress_executor.calls == 1 && resumed_budget.iterations_completed == 3);

    // A successful web search is discovery only. Its generic candidate-count
    // summary must not be promoted to a source-backed supporting claim.
    common_agent_research_workspace discovery_workspace;
    discovery_workspace.workspace_id="research:discovery-only";
    discovery_workspace.objective.objective_id="discovery-question";
    discovery_workspace.objective.question="Find a primary source";
    discovery_workspace.scope.namespace_id="discovery-ns";
    discovery_workspace.scope.session_id="discovery-session";
    discovery_workspace.gaps.push_back({"discovery-gap","Find a primary source","","read an actual source",1});
    common_agent_research_action discovery_action;
    discovery_action.kind=common_agent_research_action_kind::schedule_task;
    discovery_action.task_id="discovery-task";
    discovery_action.gap_id="discovery-gap";
    discovery_action.instruction="Find a primary source";
    discovery_action.preferred_tools={"web.search"};
    web_search_candidate_runtime candidate_runtime;
    common_agent_research_runtime_adapter discovery_adapter(candidate_runtime);
    common_agent_research_event discovery_event;
    assert(discovery_adapter.execute(discovery_action,discovery_workspace,discovery_event,run_error));
    assert(discovery_event.type==common_agent_research_event_type::task_completed);
    assert(!discovery_event.gap_sufficiently_answered && discovery_event.evidence_count==0);
    assert(discovery_workspace.sources.size()==1 && discovery_workspace.evidence.empty());
    assert(discovery_workspace.sources.front().role=="discovery_candidate" && discovery_event.discovered_sources==1);
    assert(discovery_event.assessment.status==common_agent_research_assessment_status::insufficient);

    // The controller must progress from a discovered candidate to an
    // authorized fetch without turning the search result into evidence.
    auto discovery_flow = discovery_workspace;
    discovery_flow.sources.clear();
    const auto search_action = controller.begin(discovery_flow, run_error);
    assert(search_action.kind == common_agent_research_action_kind::schedule_task);
    common_agent_research_event search_event;
    assert(discovery_adapter.execute(search_action, discovery_flow, search_event, run_error));
    const auto fetch_action = controller.advance(discovery_flow, search_event, run_error);
    assert(fetch_action.kind == common_agent_research_action_kind::schedule_task);
    assert(fetch_action.preferred_tools == std::vector<std::string>{"web.fetch"});
    assert(fetch_action.instruction == "https://nasa.example/paper" && discovery_flow.evidence.empty());

    auto record_workspace = discovery_workspace;
    record_workspace.sources.clear();
    common_agent_research_source api_source;
    api_source.source_id = "api-source";
    api_source.title = "OpenAlex record";
    api_source.kind = common_agent_research_source_kind::api_result;
    api_source.resource_ref.emplace();
    api_source.resource_ref->uri = "agent-resource://record";
    api_source.resource_ref->size_bytes = 8192;
    assert(common_agent_research_add_source(record_workspace, api_source, run_error));
    auto read_action = discovery_action;
    read_action.preferred_tools = {"resource.read"};
    read_action.instruction = api_source.resource_ref->uri;
    openalex_record_runtime record_runtime;
    common_agent_research_runtime_adapter record_adapter(record_runtime);
    common_agent_research_event record_event;
    assert(record_adapter.execute(read_action, record_workspace, record_event, run_error));
    assert(nlohmann::ordered_json::parse(record_runtime.arguments)["max_bytes"] == 8192);
    assert(record_workspace.evidence.size() == 1 &&
        record_workspace.evidence.front().statement == "OpenAlex abstract: Cooling observed");
    assert(!record_workspace.evidence.front().supporting_excerpt_host_verified);
    auto paged_workspace = discovery_workspace;
    paged_workspace.sources.clear();
    paged_openalex_runtime paged_runtime;
    api_source.resource_ref->size_bytes = paged_runtime.record.size();
    assert(common_agent_research_add_source(paged_workspace, api_source, run_error));
    common_agent_research_runtime_adapter paged_adapter(paged_runtime);
    common_agent_research_event paged_event;
    assert(paged_adapter.execute(read_action, paged_workspace, paged_event, run_error));
    assert(paged_runtime.calls > 1 && paged_workspace.tool_calls == static_cast<int>(paged_runtime.calls - 1));
    assert(paged_workspace.evidence.size() == 1 &&
        paged_workspace.evidence.front().statement == "OpenAlex abstract: Cooling observed");
    // Unknown source size makes the initial requested interval smaller than
    // the resource. has_more is false at that interval boundary, but the host
    // must follow resource_complete/next_offset until the record is complete.
    auto unknown_size_workspace = discovery_workspace;
    unknown_size_workspace.sources.clear();
    auto unknown_size_source = api_source;
    unknown_size_source.resource_ref->size_bytes = 0;
    assert(common_agent_research_add_source(unknown_size_workspace, unknown_size_source, run_error));
    assert(paged_adapter.execute(read_action, unknown_size_workspace, paged_event, run_error));
    assert(unknown_size_workspace.evidence.size() == 1 &&
        unknown_size_workspace.evidence.front().statement == "OpenAlex abstract: Cooling observed");
    auto limited_workspace = discovery_workspace;
    limited_workspace.sources.clear();
    limited_workspace.budget.max_tool_calls = 1;
    limited_workspace.tool_calls = 1; // The runner charges the first call before invoking the adapter.
    assert(common_agent_research_add_source(limited_workspace, api_source, run_error));
    assert(paged_adapter.execute(read_action, limited_workspace, paged_event, run_error));
    assert(limited_workspace.evidence.empty() && !paged_event.gap_sufficiently_answered);

    // Research uses the host's slice extractor for every page, including the
    // configured overlap, and does not promote partially analyzed evidence.
    auto text_workspace = discovery_workspace;
    text_workspace.sources.clear();
    text_workspace.evidence.clear();
    text_workspace.budget.max_tool_calls = 16;
    text_workspace.tool_calls = 1;
    text_workspace.gaps.front().question = "Target evidence";
    text_workspace.gaps.front().completion_criterion = "Target evidence";
    common_agent_research_source text_source;
    text_source.source_id = "text-source";
    text_source.title = "Text source";
    text_source.kind = common_agent_research_source_kind::web_page;
    text_source.resource_ref.emplace();
    text_source.resource_ref->uri = "agent-resource://text";
    text_source.resource_ref->size_bytes = 67;
    text_source.resource_ref->mime_type = "text/plain";
    assert(common_agent_research_add_source(text_workspace, text_source, run_error));
    paged_text_runtime text_runtime;
    auto extractor = [](const std::string &, const common_agent_generation_resource_chunk & chunk,
            common_agent_resource_chunk_extraction & extraction, int & used, std::string & error) {
        used = 10;
        const std::string response = chunk.text.find("Target evidence.") == std::string::npos
            ? "status: no_findings" : "status: findings\nevidence: Target evidence.";
        return common_agent_parse_resource_chunk_extraction(response, chunk, extraction, error);
    };
    // Chunk extraction proves that the evidence is faithfully tied to the
    // source, but it does not prove semantic relevance. Supply an explicit
    // deterministic assessor for this fixture instead of relying on the
    // production default (which must remain inconclusive without a model).
    common_agent_research_bounded_assessor text_assessor(
        [](const common_agent_research_gap &, const common_agent_research_evidence & evidence) {
            return common_agent_research_assessment{
                evidence.statement == "Target evidence."
                    ? common_agent_research_assessment_status::sufficient
                    : common_agent_research_assessment_status::inconclusive,
                evidence.statement == "Target evidence." ? 0.9 : 0.0,
                evidence.statement == "Target evidence." ? "Fixture evidence matches the target."
                    : "The fixture contains no target evidence."};
        });
    common_agent_research_runtime_adapter text_adapter(text_runtime, &text_assessor, {}, 24, 20, extractor);
    auto text_action = read_action;
    text_action.instruction = "agent-resource://text";
    common_agent_research_event text_event;
    assert(text_adapter.execute(text_action, text_workspace, text_event, run_error));
    assert(text_runtime.calls > 1 && text_event.gap_sufficiently_answered);
    assert(text_workspace.evidence.size() == 1 && text_workspace.evidence.front().supporting_excerpt_host_verified);
    auto partial_workspace = text_workspace;
    partial_workspace.evidence.clear();
    partial_workspace.tool_calls = 1;
    partial_workspace.budget.max_tool_calls = 2;
    assert(text_adapter.execute(text_action, partial_workspace, text_event, run_error));
    assert(!text_event.gap_sufficiently_answered);
    for (const auto & evidence : partial_workspace.evidence)
        assert(evidence.relation == common_agent_research_evidence_relation::contextualizes);
    auto checkpoint_workspace = discovery_workspace;
    checkpoint_workspace.sources.clear();
    checkpoint_workspace.evidence.clear();
    checkpoint_workspace.budget.max_tool_calls = 16;
    checkpoint_workspace.tool_calls = 1;
    checkpoint_workspace.gaps.front().question = "Target evidence";
    checkpoint_workspace.gaps.front().completion_criterion = "Target evidence";
    assert(common_agent_research_add_source(checkpoint_workspace, text_source, run_error));
    common_agent_research_workspace saved_checkpoint;
    int saves = 0;
    auto interrupting_sink = [&](const common_agent_research_workspace & current, std::string & sink_error) {
        ++saves;
        if (!current.resource_progress.empty() &&
                current.resource_progress.front().position.next_offset == 48 &&
                current.generation_tokens_reserved >= 1152) {
            sink_error = "simulated checkpoint interruption before the third chunk";
            return false;
        }
        saved_checkpoint = current; sink_error.clear(); return true;
    };
    common_agent_research_runtime_adapter interrupted_adapter(text_runtime, &text_assessor, {}, 24, 20, extractor, interrupting_sink);
    assert(!interrupted_adapter.execute(text_action, checkpoint_workspace, text_event, run_error));
    assert(saved_checkpoint.resource_progress.size() == 1 && saved_checkpoint.resource_progress.front().position.next_offset == 48);
    assert(checkpoint_workspace.resource_progress.front().position.next_offset == 48);
    assert(checkpoint_workspace.evidence.size() == saved_checkpoint.evidence.size());
    const auto serialized_checkpoint = common_agent_research_workspace_to_json(saved_checkpoint);
    common_agent_research_workspace reloaded_checkpoint;
    assert(common_agent_research_workspace_from_json(serialized_checkpoint, reloaded_checkpoint, run_error));
    auto durable_sink = [](const common_agent_research_workspace &, std::string & sink_error) { sink_error.clear(); return true; };
    common_agent_research_runtime_adapter resumed_adapter(text_runtime, &text_assessor, {}, 24, 20, extractor, durable_sink);
    auto changed_source = reloaded_checkpoint;
    changed_source.sources.front().content_hash = "sha256:different";
    assert(!resumed_adapter.execute(text_action, changed_source, text_event, run_error));
    assert(changed_source.resource_progress.front().position.status == common_resource_progress_status::stale);
    auto changed_task = reloaded_checkpoint;
    changed_task.gaps.front().question = "A different research question";
    assert(!resumed_adapter.execute(text_action, changed_task, text_event, run_error));
    assert(changed_task.resource_progress.front().position.status == common_resource_progress_status::stale);
    common_agent_research_runtime_adapter incompatible_adapter(text_runtime, &text_assessor, {}, 20, 12, extractor, durable_sink);
    auto changed_policy = reloaded_checkpoint;
    assert(!incompatible_adapter.execute(text_action, changed_policy, text_event, run_error));
    assert(changed_policy.resource_progress.front().position.status == common_resource_progress_status::stale);
    assert(resumed_adapter.execute(text_action, reloaded_checkpoint, text_event, run_error));
    assert(reloaded_checkpoint.resource_progress.front().position.status == common_resource_progress_status::complete);
    assert(text_event.gap_sufficiently_answered && reloaded_checkpoint.evidence.size() == 1);
    auto planned_workspace = discovery_workspace;
    planned_workspace.sources.clear();
    planned_workspace.evidence.clear();
    planned_workspace.budget.max_tool_calls = 16;
    planned_workspace.tool_calls = 1;
    planned_workspace.gaps.front().question = "Target evidence";
    planned_workspace.gaps.front().completion_criterion = "Target evidence";
    assert(common_agent_research_add_source(planned_workspace, text_source, run_error));
    size_t planned_calls = 0;
    auto host_plan = [&](const std::string & uri, common_agent_resource_chunk_plan & plan, std::string & plan_error) {
        ++planned_calls;
        plan.source_uri = uri;
        plan.content_fingerprint = "sha256:0123456789abcdef";
        plan.source_bytes = text_runtime.text.size();
        for (size_t next = 0; next < plan.source_bytes;) {
            const size_t overlap = next ? std::min<size_t>(20, next) : 0;
            const size_t fresh = std::min<size_t>(24, plan.source_bytes - next);
            plan.ranges.push_back({next - overlap, overlap + fresh, overlap});
            next += fresh;
        }
        plan_error.clear(); return true;
    };
    common_agent_research_runtime_adapter planned_adapter(text_runtime, &text_assessor, {}, 24, 20,
        extractor, durable_sink, host_plan);
    assert(planned_adapter.execute(text_action, planned_workspace, text_event, run_error));
    assert(planned_calls == 1 && text_event.gap_sufficiently_answered);
    assert(planned_workspace.resource_progress.front().position.query_fingerprint == "host-text-window-v2");
    assert(planned_workspace.resource_progress.front().position.next_offset == text_runtime.text.size());
    auto planned_interrupted = discovery_workspace;
    planned_interrupted.sources.clear();
    planned_interrupted.evidence.clear();
    planned_interrupted.budget.max_tool_calls = 16;
    planned_interrupted.tool_calls = 1;
    planned_interrupted.gaps.front().question = "Target evidence";
    planned_interrupted.gaps.front().completion_criterion = "Target evidence";
    assert(common_agent_research_add_source(planned_interrupted, text_source, run_error));
    common_agent_research_workspace planned_saved;
    size_t planned_saves = 0;
    auto stop_after_first = [&](const common_agent_research_workspace & current, std::string & sink_error) {
        ++planned_saves;
        if (!current.resource_progress.empty() &&
                current.resource_progress.front().position.next_offset == 24 &&
                current.generation_tokens_reserved >= 768) {
            sink_error = "simulated interruption before the second chunk";
            return false;
        }
        planned_saved = current; sink_error.clear(); return true;
    };
    common_agent_research_runtime_adapter planned_first(text_runtime, &text_assessor, {}, 24, 20,
        extractor, stop_after_first, host_plan);
    assert(!planned_first.execute(text_action, planned_interrupted, text_event, run_error));
    assert(planned_saved.resource_progress.front().position.next_offset == 24);
    common_agent_research_workspace planned_restored;
    assert(common_agent_research_workspace_from_json(common_agent_research_workspace_to_json(planned_saved),
        planned_restored, run_error));
    common_agent_research_runtime_adapter planned_resume(text_runtime, &text_assessor, {}, 24, 20,
        extractor, durable_sink, host_plan);
    assert(planned_resume.execute(text_action, planned_restored, text_event, run_error));
    assert(text_event.gap_sufficiently_answered && planned_restored.evidence.size() == 1);
    assert(planned_restored.resource_progress.front().position.next_offset == text_runtime.text.size());
    common_runtime_resource_ref retained_record = *api_source.resource_ref;
    retained_record.uri = "agent-resource://retained-record";
    common_agent_research_rebind_resource(record_workspace, record_workspace.sources.front(), retained_record);
    assert(record_workspace.evidence.front().source_location ==
        "agent-resource://retained-record#/abstract_inverted_index");

    auto confidence_workspace = workspace;
    confidence_workspace.gaps.front().status = common_agent_research_gap_status::open;
    confidence_workspace.gaps.front().evidence_ids.clear();
    confidence_workspace.budget.minimum_confidence = 0.95;
    confidence_workspace.coverage = {};
    const auto confidence_start = controller.begin(confidence_workspace, error);
    assert(confidence_start.kind == common_agent_research_action_kind::schedule_task);
    const auto confidence_action = controller.advance(confidence_workspace, {
        common_agent_research_event_type::task_completed,
        confidence_start.task_id,
        confidence_start.gap_id,
        1,
        {"evidence-1"}}, error);
    assert(confidence_action.kind == common_agent_research_action_kind::schedule_task);

    auto insufficient_workspace = workspace;
    insufficient_workspace.gaps.front().status = common_agent_research_gap_status::open;
    insufficient_workspace.gaps.front().evidence_ids.clear();
    insufficient_workspace.coverage = {};
    const auto insufficient_start = controller.begin(insufficient_workspace, error);
    assert(insufficient_start.kind == common_agent_research_action_kind::schedule_task);
    const auto insufficient_action = controller.advance(insufficient_workspace, {
        common_agent_research_event_type::task_completed,
        insufficient_start.task_id,
        insufficient_start.gap_id,
        1,
        {"evidence-1"},
        false,
        0.9}, error);
    assert(insufficient_action.kind == common_agent_research_action_kind::schedule_task);
    assert(insufficient_workspace.gaps.front().status != common_agent_research_gap_status::sufficiently_answered);
    // The host selection boundary repairs invalid arguments and carries only
    // observed provider candidates to a subsequent lookup.
    auto selected_workspace = discovery_workspace;
    selected_workspace.sources.clear();
    selected_workspace.evidence.clear();
    // The selected candidate becomes the materialized source in place.
    selected_workspace.budget.max_sources = 1;
    selected_provider_runtime selected_runtime;
    int repairs = 0;
    auto select_provider = [&](const common_agent_research_action &, const common_agent_research_workspace & current,
            const common_agent_tool_runtime &, const common_agent_tool_call * previous, const std::string & validation,
            std::optional<common_agent_tool_call> & call, std::string &, std::string & select_error) {
        select_error.clear();
        if (previous) {
            assert(previous->name == "openalex.listWorks" && validation == "search is too long");
            ++repairs;
            call = common_agent_tool_call{"openalex.listWorks", R"({"search":"urban cooling"})"};
        } else if (current.sources.empty()) {
            call = common_agent_tool_call{"openalex.listWorks", R"({"search":"this query is deliberately much too long"})"};
        } else {
            assert(current.sources.front().origin == "https://openalex.org/W123");
            call = common_agent_tool_call{"openalex.getWork", R"({"id":"W123"})"};
        }
        if (call->name == "openalex.listWorks") {
            call->collection_item_parameter = "id";
            call->collection_items_pointer = "/results";
        } else if (call->name == "openalex.getWork") {
            call->item_lookup_parameter = "id";
        }
        return true;
    };
    common_agent_research_runtime_adapter selected_adapter(selected_runtime, nullptr, {}, 0, 0, {}, {}, {}, select_provider);
    auto selected_action = discovery_action;
    selected_action.preferred_tools = {"repository.search", "web.search"};
    common_agent_research_event selected_event;
    assert(selected_adapter.execute(selected_action, selected_workspace, selected_event, run_error));
    assert(repairs == 1 && selected_runtime.executions == 1 && selected_workspace.evidence.empty());
    assert(selected_workspace.sources.size() == 1 && selected_workspace.sources.front().role == "api_discovery_candidate");
    assert(common_agent_research_source_count(selected_workspace) == 0);
    assert(common_agent_research_candidate_count(selected_workspace) == 1);
    common_agent_research_workspace selected_restored;
    assert(common_agent_research_workspace_from_json(common_agent_research_workspace_to_json(selected_workspace), selected_restored, run_error));
    selected_action.task_id = "lookup-task";
    assert(selected_adapter.execute(selected_action, selected_restored, selected_event, run_error));
    assert(selected_runtime.executions == 2 && selected_restored.evidence.empty());
    assert(selected_restored.sources.size() == 1 && selected_restored.sources.back().resource_ref->uri == "agent-resource://paper");
    assert(common_agent_research_source_count(selected_restored) == 1);
    assert(common_agent_research_candidate_count(selected_restored) == 0);

    auto decline = [](const common_agent_research_action &, const common_agent_research_workspace &,
            const common_agent_tool_runtime &, const common_agent_tool_call *, const std::string &,
            std::optional<common_agent_tool_call> & call, std::string &, std::string & select_error) {
        call.reset(); select_error.clear(); return true;
    };
    common_agent_research_runtime_adapter no_action(selected_runtime, nullptr, {}, 0, 0, {}, {}, {}, decline);
    assert(no_action.execute(selected_action, selected_restored, selected_event, run_error));
    assert(selected_event.type == common_agent_research_event_type::no_action &&
        selected_event.failure_code == "research.no_action" && selected_runtime.executions == 2);
    auto forbidden = [](const common_agent_research_action &, const common_agent_research_workspace &,
            const common_agent_tool_runtime &, const common_agent_tool_call *, const std::string &,
            std::optional<common_agent_tool_call> & call, std::string &, std::string & select_error) {
        call = common_agent_tool_call{"web.search", R"({"query":"other"})"};
        select_error.clear(); return true;
    };
    common_agent_research_runtime_adapter denied(selected_runtime, nullptr, {}, 0, 0, {}, {}, {}, forbidden);
    assert(denied.execute(selected_action, selected_restored, selected_event, run_error));
    assert(selected_event.type == common_agent_research_event_type::task_failed &&
        selected_runtime.executions == 2);

    auto provider_checkpoint_workspace = discovery_workspace;
    provider_checkpoint_workspace.sources.clear();
    provider_checkpoint_workspace.evidence.clear();
    provider_checkpoint_workspace.budget.max_iterations = 4;
    provider_checkpoint_workspace.budget.max_tool_calls = 8;
    common_agent_research_workspace saved_after_discovery;
    bool stop_after_checkpoint = false;
    auto checkpoint = [&](const common_agent_research_workspace & current, std::string & checkpoint_error) {
        saved_after_discovery = current;
        stop_after_checkpoint = true;
        checkpoint_error.clear(); return true;
    };
    const auto interrupted = research_runner.run(provider_checkpoint_workspace, selected_adapter, run_error,
        [&] { return stop_after_checkpoint; }, {}, {}, checkpoint);
    assert(!interrupted.complete && interrupted.stop_reason == common_agent_research_stop_reason::cancelled);
    assert(saved_after_discovery.sources.size() == 1 &&
        saved_after_discovery.sources.front().role == "api_discovery_candidate");
    return 0;
}
