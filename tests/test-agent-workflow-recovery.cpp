#include "tools/agent/runtime/agent-runtime-execution.h"
#include "tools/agent/tooling/agent-tool-provider.h"
#include "tools/agent/resource/agent-resource-store.h"
#include "agent/agent-bootstrap.h"
#include "agent/agent-package-json.h"
#include "agent/template-library.h"
#include "memory/memory-in-memory.h"
#include "plan/plan-in-memory.h"
#include "plan/plan-scheduler.h"
#include "test-plan-route-entry.h"
#ifdef LLAMA_RECOVERY_TEST_SQLITE
#include "plan/sqlite/plan-sqlite.h"
#endif

#include <algorithm>
#include <filesystem>
#include <functional>
#include <stdexcept>

namespace {
using json = nlohmann::ordered_json;
void check(bool condition, const std::string & reason) {
    if (!condition) throw std::runtime_error(reason);
}

// These are ordinary configured workflows. No provider-name routing or live
// network/model is involved in exercising the production runtime driver.
const char * package_json = R"({
 "schema_version":2,"name":"recovery-contract","version":"v1",
 "workflows":[
  {"id":"discover","workflow_ref":"workflow://catalog/discover","workflow_revision":"v1",
   "graph_revision":"discover-v1","family":"openapi","allowed_tools":["catalog.list"],
   "start_state":"start","terminal_states":["listed"],
   "transitions":[{"id":"search","from":"start","to":"listed","tool":"catalog.list","arguments":"{\"query\":\"green space\"}"}],
   "continuation_links":[{"from_transition":"search","from_terminal_state":"listed",
    "workflow_ref":"workflow://catalog/evidence","workflow_revision":"v1",
    "bindings":[{"transition":"get","tool":"catalog.get","target":"/id","source":"/result/host_item_candidates","candidate_id":"/id"}]}]},
  {"id":"evidence","workflow_ref":"workflow://catalog/evidence","workflow_revision":"v1",
   "graph_revision":"evidence-v1","family":"openapi","resource_read_policy":"chunked",
   "allowed_tools":["catalog.get","web.fetch","resource.inspect","resource.read"],
   "start_state":"start","terminal_states":["unavailable","read"],
   "transitions":[
    {"id":"get","from":"start","to":"record","tool":"catalog.get","arguments":"{}"},
    {"id":"depth","from":"record","to":"fetchable","kind":"model_choice","choices":[
     {"id":"fetch","description":"Inspect required full text","to":"fetchable","requires":["/result/full_text_url"]},
     {"id":"try_another","description":"Full text is required but no usable location exists; retain shallow evidence and request another candidate","to":"unavailable"}]},
    {"id":"fetch","from":"fetchable","to":"fetched","tool":"web.fetch","arguments":"{}",
     "failure_terminal_state":"unavailable","handled_http_statuses":[403,404],
     "bindings":[{"target":"/url","from":"get","source":"/result/full_text_url"}]},
    {"id":"inspect","from":"fetched","to":"inspected","tool":"resource.inspect","arguments":"{}",
     "bindings":[{"target":"/uri","from":"fetch","source":"/resources/0/uri"}]},
    {"id":"read","from":"inspected","to":"read","tool":"resource.read","arguments":"{\"representation\":\"text\",\"max_bytes\":512}",
     "bindings":[{"target":"/uri","from":"fetch","source":"/resources/0/uri"}]}],
   "continuation_links":[{"from_terminal_state":"unavailable","max_uses":1,
    "workflow_ref":"workflow://catalog/evidence","workflow_revision":"v1",
    "bindings":[{"transition":"get","tool":"catalog.get","target":"/id","source":"/handoff/value/result/host_item_candidates","candidate_id":"/id"}]}]}
 ],
 "blueprints":[
  {"id":"discover","goal":"Discover candidate works","success_criteria":"Returned identities are preserved",
   "workflow_policy":"required","workflow_bindings":[{"workflow_ref":"workflow://catalog/discover","workflow_revision":"v1"}],
   "steps":[{"id":"discover","title":"Discover","objective":"Find candidate works"}]},
  {"id":"evidence","goal":"Inspect a selected work","success_criteria":"Inspect text or disclose the access limitation",
   "workflow_policy":"required","workflow_bindings":[{"workflow_ref":"workflow://catalog/evidence","workflow_revision":"v1"}],
   "steps":[{"id":"inspect","title":"Inspect","objective":"Inspect the selected evidence"}]}
 ]})";

class source_view final : public agent_tool_view {
public:
    std::vector<common_chat_tool> tools = {
        {"catalog.list", "Find candidate works", R"({"type":"object","required":["query"],"properties":{"query":{"type":"string"}},"additionalProperties":false})"},
        {"catalog.get", "Read one returned work", R"({"type":"object","required":["id"],"properties":{"id":{"type":"string"},"language":{"type":"string"}},"additionalProperties":false})"},
        {"web.fetch", "Fetch the returned full text", R"({"type":"object","required":["url"],"properties":{"url":{"type":"string"}},"additionalProperties":false})"},
        {"resource.inspect", "Inspect a resource", R"({"type":"object","required":["uri"],"properties":{"uri":{"type":"string"}},"additionalProperties":false})"},
        {"resource.read", "Read resource text", R"({"type":"object","required":["uri"],"properties":{"uri":{"type":"string"},"representation":{"type":"string"},"max_bytes":{"type":"integer"},"offset":{"type":"integer"}},"additionalProperties":false})"},
    };
    std::vector<std::string> lookups, fetched;
    size_t searches = 0, inspections = 0, reads = 0;
    bool second_blocked = false, singleton = false;
    size_t missing_urls = 0;
    common_runtime_resource_ref full_text;
    std::string text;

    const std::vector<common_chat_tool> & chat_tools() const override { return tools; }
    bool exposes_tool(const std::string & name) const override {
        return std::any_of(tools.begin(), tools.end(), [&](const auto & tool) { return tool.name == name; });
    }
    bool is_read_only(const std::string &) const override { return true; }
    bool is_policy_gated(const std::string &) const override { return false; }
    bool validate(const agent_tool_call & call, std::string & error) const override {
        error.clear(); return exposes_tool(call.name) && json::parse(call.arguments_json, nullptr, false).is_object();
    }
    agent_tool_result call(const agent_tool_call & call, std::string & error) override {
        check(validate(call, error), "runtime dispatched an unauthorized or invalid call");
        const auto args = json::parse(call.arguments_json);
        agent_tool_result result;
        result.ok = true; result.tool_call_id = call.id; result.tool_name = call.name;
        if (call.name == "catalog.list") {
            ++searches;
            auto candidates = json::array({{{"id","W1"},{"title","First work"}}});
            if (!singleton) { candidates.push_back({{"id","W2"},{"title","Second work"}}); candidates.push_back({{"id","W3"},{"title","Third work"}}); }
            result.content_json = json{{"host_item_parameter","id"},{"host_item_candidates",candidates}}.dump();
        } else if (call.name == "catalog.get") {
            check(args.value("language", std::string{}) == "en", "route change discarded an execution host constraint");
            const auto id = args.at("id").get<std::string>();
            check(id == "W1" || id == "W2", "lookup did not use an untried observed identity");
            check(std::find(lookups.begin(), lookups.end(), id) == lookups.end(), "candidate was looked up twice");
            lookups.push_back(id);
            result.content_json = json{{"id",id},{"abstract_text","Only the abstract is currently available."},
                {"full_text_url","https://example.org/"+id+".txt"}}.dump();
            if (lookups.size() <= missing_urls) {
                auto record = json::parse(result.content_json);
                record.erase("full_text_url");
                result.content_json = record.dump();
            }
        } else if (call.name == "web.fetch") {
            const auto url = args.at("url").get<std::string>();
            fetched.push_back(url);
            const bool first = url == "https://example.org/W1.txt";
            if (first || second_blocked) {
                result.ok = false; result.failure_code = "tool.web_fetch.request_failed";
                result.failure_class = common_tool_failure_class::network;
                result.http.status = first ? 403 : 404;
                result.safe_summary = "Full text is unavailable";
                result.raw_diagnostic = "HTTP status " + std::to_string(result.http.status);
            } else {
                check(url == "https://example.org/W2.txt", "fetch used an invented URL");
                result.resource_refs = {full_text};
                result.content_json = json{{"resource_ref",full_text.uri},{"status",200}}.dump();
            }
        } else if (call.name == "resource.inspect") {
            ++inspections; check(args.at("uri") == full_text.uri, "inspected a different resource");
            result.content_json = json{{"uri",full_text.uri},{"size_bytes",text.size()}}.dump();
            result.resource_refs = {full_text};
        } else if (call.name == "resource.read") {
            ++reads; check(args.at("uri") == full_text.uri, "read a different resource");
            const size_t offset = args.value("offset", size_t{0});
            const size_t limit = args.value("max_bytes", size_t{512});
            result.content_json = json{{"uri",full_text.uri},{"text",text.substr(offset,limit)},
                {"offset",offset},{"has_more",offset+limit<text.size()}}.dump();
            result.resource_refs = {full_text};
        }
        if (result.ok) {
            json envelope{{"ok",true},{"result",json::parse(result.content_json)}};
            if (!result.resource_refs.empty()) {
                envelope["resources"] = json::array();
                for (const auto & resource : result.resource_refs)
                    envelope["resources"].push_back({{"uri",resource.uri}});
            }
            result.content_json = envelope.dump();
        }
        error.clear(); return result;
    }
    bool supports_async_call(const std::string &) const override { return false; }
    bool begin_call_async(const agent_tool_call &, agent_tool_pending_call &, std::string &) override { return false; }
    bool poll_call_async(const agent_tool_pending_call &, bool &, agent_tool_result &, std::string &) override { return false; }
    bool cancel_call_async(const agent_tool_pending_call &, std::string &) override { return false; }
};

class model final : public common_agent_inference {
public:
    explicit model(source_view & view) : view(view) {}
    source_view & view;
    size_t selections = 0, extraction_calls = 0, draft_calls = 0, review_calls = 0;
    bool generate(const common_agent_generation_request & request, common_agent_generation_result & result) override {
        result = {}; result.status = common_agent_generation_status::completed;
        result.stop_reason = common_agent_generation_stop_reason::eos; result.decoded_tokens = 1;
        if (request.purpose == common_agent_generation_purpose::operation_selection) {
            const auto schema = json::parse(request.json_schema);
            check(schema.contains("properties") && schema["properties"].contains("id"), "unexpected semantic operation or argument generation");
            const auto ids = schema["properties"]["id"]["enum"].get<std::vector<std::string>>();
            check(!ids.empty(), "no observed candidates");
            if (std::find(ids.begin(), ids.end(), "try_another") != ids.end()) {
                result.content = json{{"id",std::find(ids.begin(),ids.end(),"fetch") != ids.end() ? "fetch" : "try_another"}}.dump();
                return true;
            }
            for (const auto & prior : view.lookups)
                check(std::find(ids.begin(), ids.end(), prior) == ids.end(), "used candidate reached the model again");
            ++selections; result.content = json{{"id",ids.front()}}.dump();
        } else if (request.purpose == common_agent_generation_purpose::resource_chunk_extraction) {
            check(request.resource_chunk && !request.resource_chunk->segments.empty(), "extraction lacks a host slice");
            ++extraction_calls;
            const auto & paragraph = request.resource_chunk->segments.front().text;
            const auto end = paragraph.find('.');
            result.content = end == std::string::npos ? "status: no_findings" :
                "status: findings\nevidence: " + paragraph.substr(0,end+1);
        } else if (request.purpose == common_agent_generation_purpose::reflection) {
            ++review_calls;
            check(view.lookups.size() == (view.singleton ? 1u : 2u), "reflection ran before the required fallback");
            std::string context;
            for(const auto & message:request.messages) context+=message.content;
            check(context.find("\"mandatory_remaining\":false")!=std::string::npos,
                "review did not receive host continuation closure");
            if(view.second_blocked || view.singleton)
                check(context.find("\"partial_allowed\":true")!=std::string::npos,
                    "bounded source failure was not presented as an allowed qualified partial answer");
            result.content = R"({"decision":"accept","assurance_action":"accept","ready_to_answer":true})";
        } else if (request.purpose == common_agent_generation_purpose::draft) {
            ++draft_calls;
            check(view.lookups.size() == (view.singleton ? 1u : 2u), "draft bypassed required candidate recovery");
            if (!view.second_blocked && !view.singleton)
                check(view.inspections && view.reads && extraction_calls > 1, "draft preceded chunked full-text analysis");
            result.content = view.second_blocked || view.singleton ? "Abstract evidence retained; full-text attempts were blocked." :
                "The second work was read and analyzed from the retrieved full text.";
        } else throw std::runtime_error("unexpected inference purpose in workflow recovery contract");
        return true;
    }
};

class observing_store final : public common_plan_store {
public:
    explicit observing_store(common_plan_store & backend) : backend(backend) {}
    common_plan_store & backend;
    std::function<void(const common_plan_operation &)> after_commit;
    bool open(const std::string & path, std::string & error) override { return backend.open(path,error); }
    void close() override { backend.close(); }
    bool create(const common_plan_state & plan, std::string & error) override { return backend.create(plan,error); }
    std::optional<common_plan_state> get(const std::string & id, std::string & error) override { return backend.get(id,error); }
    std::vector<common_plan_state> list(std::string & error) override { return backend.list(error); }
    std::vector<common_plan_event> history(const std::string & id, std::string & error) override { return backend.history(id,error); }
    bool erase(const std::string & id, std::string & error) override { return backend.erase(id,error); }
    bool apply(const common_plan_operation & op, common_plan_state & plan, std::string & error) override {
        if (!backend.apply(op,plan,error)) return false;
        if (after_commit) after_commit(op);
        return true;
    }
};

void run_recovery(common_plan_store & backend, const std::string & path, bool blocked, bool singleton, bool restart,
        size_t missing_urls = 0) {
    observing_store plans(backend);
    common_memory_in_memory_store memories;
    std::string error;
    check(memories.open("",error) && plans.open(path,error), error);
    common_agent_bootstrap_package package;
    check(common_agent_package_parse_json(package_json,package,error), error);
    common_agent_bootstrap_config bootstrap;
    bootstrap.namespace_id = "local"; bootstrap.session_id = "default"; bootstrap.install_procedures = false;
    common_agent_bootstrap_result installed;
    check(common_agent_install_bootstrap_package(memories,plans,bootstrap,package,{},installed,error), error);
    std::vector<common_blueprint_candidate> blueprints;
    for (const auto & id : installed.installed_blueprint_ids) {
        const auto definition = plans.get(id,error); check(definition.has_value(), error);
        blueprints.push_back(common_agent_template_candidate(id.substr(id.rfind(':')+1),*definition));
    }
    auto blobs = std::make_shared<agent_in_memory_blob_store>();
    agent_catalogued_resource_store resources(blobs,std::make_unique<agent_in_memory_resource_catalog>());
    agent_resource_put_request put;
    put.name = "paper.txt"; put.scope = common_runtime_resource_scope::session;
    put.session_id = "default"; put.turn_id = "recovery";
    for (size_t i=0; i<12; ++i)
        put.text += "The study compared tree cover and health outcomes in cohort " + std::to_string(i) +
            ". The analysis reported a sustained association while acknowledging observational limitations.\n\n";
    agent_resource_descriptor resource;
    check(resources.put_text(put,resource,error),error);
    source_view view;
    view.second_blocked = blocked; view.singleton = singleton; view.text = put.text;
    view.missing_urls = missing_urls;
    view.full_text.uri = resource.uri; view.full_text.name = resource.name;
    view.full_text.mime_type = "text/plain"; view.full_text.size_bytes = resource.size_bytes;
    model inference(view);
    common_agent_runtime_tooling tooling;
    tooling.tools = view.tools; tooling.tool_view = &view; tooling.profile_tools_active = true;
    tooling.resource_runtime.store = &resources; tooling.resource_runtime.turn_id = "recovery";
    common_agent_scope scope;
    scope.namespace_id = "local"; scope.session_id = "default"; scope.turn_id = "recovery";
    scope.plan_scope = common_plan_scope::turn;
    std::string plan_id;
    std::vector<common_memory_hit> hits;
    common_agent_runtime_policy policy;
    policy.max_iterations = 16; policy.max_tool_rounds = 16;
    policy.enable_reflection = true; policy.max_reflection_rounds = 2;
    common_agent_runtime_config config;
    config.max_continuations = singleton ? 1 : 2;
    config.generation_config.n_predict = 256;
    config.generation_config.generation_trace = true;
    config.generation_config.context_budgets.resource_chunk_max_bytes = 512;
    config.generation_config.context_budgets.resource_chunk_overlap_bytes = 0;
    config.context_budgets = config.generation_config.context_budgets;
    common_agent_orchestration_config orchestration;
    orchestration.agent_plan = "off"; orchestration.agent_blueprint = "discover";
    orchestration.prompt = "Find candidate works, inspect the full text, and try another work if access fails.";
    const auto make_execution = [&]() {
        common_agent_runtime_driver_execution execution{memories,plans,inference,policy,config,orchestration,
            plan_id,scope,blueprints,std::nullopt,hits,common_memory_scope::session,false,tooling};
        execution.tool_output_format = common_agent_tool_output_format::native;
        execution.tool_argument_bindings.push_back({"catalog.get", R"({"language":"en"})", "host-configuration", false});
        return execution;
    };
    auto execution = make_execution();
    auto cancellation = std::make_shared<common_agent_runtime_cancellation_state>();
    if (restart) {
        execution.execution_control.cancellation = cancellation;
        plans.after_commit = [&](const common_plan_operation & op) {
            if (op.workflow_continuation_record && op.workflow_continuation_record->source_terminal_state == "unavailable")
                cancellation->request_cancel("stop after atomic fallback checkpoint");
        };
    }
    common_agent_result result;
    const bool first_ok = run_agent_runtime_driver(execution,result,error);
    if (restart) {
        check(!first_ok && cancellation->is_cancelled() && view.lookups == std::vector<std::string>{"W1"},
            "interruption did not occur between committed W2 work and its execution: " + error);
        plans.after_commit = {};
        plans.close(); check(plans.open(path,error),error);
        auto resumed = make_execution();
        check(run_agent_runtime_driver(resumed,result,error),error);
    } else check(first_ok,error);
    check(result.error.empty() && !result.response.empty(), "recovery failed to reach a bounded final answer: " + result.error);
    check(view.searches == 1 && view.lookups == (singleton ? std::vector<std::string>{"W1"} : std::vector<std::string>{"W1","W2"}),
        "recovery repeated discovery or failed to inspect a distinct candidate");
    check(view.fetched.size() == view.lookups.size() - std::min(missing_urls,view.lookups.size()),
        "fetch count does not match the host-observed full-text locations");
    check(inference.selections == view.lookups.size(), "resume repeated model candidate selection");
    check(inference.draft_calls && inference.review_calls, "recovery did not reach answer review");
    const auto saved = plans.get(plan_id,error); check(saved.has_value(),error);
    check(saved->workflow_continuation_history.size() == (singleton ? 1u : 2u), "no-op handoff consumed a transition");
    if (missing_urls) {
        size_t unavailable_choices = 0;
        for (const auto & observation : saved->observations) {
            if (observation.source != "workflow_terminal_choice") continue;
            const auto outcome = json::parse(observation.summary);
            if (outcome.value("choice_id",std::string{}) != "try_another") continue;
            ++unavailable_choices;
            check(outcome.value("terminal_state",std::string{}) == "unavailable" &&
                outcome.contains("handoff") && !outcome.contains("failure") &&
                !outcome.contains("http_status"), "missing URL became a fabricated HTTP failure or lost its frontier");
        }
        check(unavailable_choices == std::min(missing_urls,view.lookups.size()), "missing locator outcomes were lost");
    }
    if (!blocked && !singleton) {
        size_t chunks = 0;
        for (const auto & observation : saved->observations) if (observation.source == "resource_chunk") ++chunks;
        check(chunks > 1, "successful fallback did not persist chunk analysis");
    }
    plans.close();
}
} // namespace

int main() {
    common_plan_in_memory_store atomic;
    std::string error;
    check(atomic.open("",error),error);
    write_atomic_route_entry(atomic);
    verify_atomic_route_entry(atomic);
    common_plan_in_memory_store successful;
    run_recovery(successful,"",false,false,false);
    common_plan_in_memory_store blocked;
    run_recovery(blocked,"",true,false,false);
    common_plan_in_memory_store exhausted;
    run_recovery(exhausted,"",true,true,false);
    common_plan_in_memory_store missing_then_success;
    run_recovery(missing_then_success,"",false,false,false,1);
    common_plan_in_memory_store both_missing;
    run_recovery(both_missing,"",true,false,false,2);
    common_plan_in_memory_store no_alternative;
    run_recovery(no_alternative,"",true,true,false,1);
#ifdef LLAMA_RECOVERY_TEST_SQLITE
    const auto path = std::filesystem::temp_directory_path() / "llama-workflow-recovery.sqlite";
    std::filesystem::remove(path);
    common_plan_sqlite_store persisted;
    run_recovery(persisted,path.string(),true,false,true,1);
    std::filesystem::remove(path);
    run_recovery(persisted,path.string(),true,false,true);
    std::filesystem::remove(path);
#endif
}
