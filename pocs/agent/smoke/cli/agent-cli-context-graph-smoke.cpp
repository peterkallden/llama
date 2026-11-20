#include "agent/tooling/catalog/tool-catalog.h"
#include "agent/tooling/adapters/families/repository-adapters.h"
#include <fstream>
#include "agent/tooling/schema/tool-output-codec.h"
#include "agent/tooling/schema/tool-dsl-grammar.h"
#include "tools/agent/runtime/agent-server-context-host.h"
#include "tools/agent/tooling/agent-tool-provider.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>

using json = nlohmann::ordered_json;

class smoke_store final : public common_agent_data_store {
public:
    common_context_graph_state graph;
    bool execute(const std::string &, const std::string &, std::string &, std::string &) override { return false; }
    bool get_context_graph(const common_agent_scope &, common_context_graph_state & out, std::string & error) override {
        out = graph; error.clear(); return true;
    }
    common_agent_publication_status publish_context_graph(const common_agent_scope &, uint64_t expected,
            const common_context_graph_state & in, common_context_graph_state & out, std::string & error) override {
        if (graph.revision != expected) { error = "revision conflict"; return common_agent_publication_status::conflict; }
        graph = in; out = graph; error.clear(); return common_agent_publication_status::published;
    }
};

int main(int argc, char ** argv) {
    std::string model;
    if (const char * path = std::getenv("LLAMA_AGENT_MODEL")) model = path;
    int gpu_layers = -1;
    std::string repository = std::filesystem::current_path().string(), trace_path;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) model = argv[++i];
        else if (arg == "--n-gpu-layers" && i + 1 < argc) gpu_layers = std::stoi(argv[++i]);
        else if (arg == "--repository" && i + 1 < argc) repository = argv[++i];
        else if (arg == "--trace-file" && i + 1 < argc) trace_path = argv[++i];
        else if (arg == "--no-model") model.clear();
        else { std::cerr << "usage: " << argv[0] << " [--no-model | --model PATH] [--n-gpu-layers N] [--repository PATH] [--trace-file PATH]\n"; return 2; }
    }
    std::ofstream trace_file;
    if (!trace_path.empty()) { trace_file.open(trace_path); if (!trace_file) { std::cerr << "cannot open trace file\n"; return 1; } }
    std::ostream & trace = trace_path.empty() ? std::cout : trace_file;
    const auto traced_call = [&](agent_tool_view & tools, const agent_tool_call & call, std::string & error) {
        trace << json{{"event", "tool.request"}, {"id", call.id}, {"tool", call.name}, {"arguments", json::parse(call.arguments_json)}}.dump() << std::endl;
        auto result = tools.call(call, error);
        trace << json{{"event", "tool.result"}, {"id", call.id}, {"tool", call.name}, {"ok", result.ok},
            {"content", result.content_json}, {"error", error}}.dump() << std::endl;
        return result;
    };
    const auto source = common_repository_observation(repository, "CMakeLists.txt");
    if (source.value("status", std::string{}) != "observed") { std::cerr << "repository observation unavailable\n"; return 1; }
    trace << json{{"event", "source.observed"}, {"source", source}}.dump() << std::endl;
    std::string error;
    common_tool_catalog catalog; common_tool_bootstrap_result bootstrap;
    common_tool_profile profile; profile.id = "source-graph-smoke"; profile.allow_policy_gated_writes = true;
    for (const auto * name : {"graph.find", "graph.get", "graph.expand", "graph.search", "graph.link.propose", "graph.describe.propose", "repository.read"})
        profile.members.push_back({name, 1, true, "{}"});
    if (!catalog.bootstrap(profile.id, bootstrap, error, {}, {{profile.id, profile}})) { std::cerr << error << '\n'; return 1; }
    smoke_store store;
    native_agent_tool_provider provider(catalog, [&](const agent_tool_context & context, common_native_tool_bindings & bindings, std::string &) {
        bindings.data_store = &store; bindings.tool_scope = context.scope; bindings.repository_root = repository; return true;
    });
    agent_tool_context context; context.profile_id = profile.id; context.scope.namespace_id = "smoke";
    context.scope.project_id = "project"; context.max_calls = 48; agent_tool_context_apply_policy_gated_writes(context, true);
    auto view = provider.resolve_tools(context, error);
    if (!view) { std::cerr << error << '\n'; return 1; }
    const auto setup = traced_call(*view, {"setup", "graph.link.propose", R"({"from":{"kind":"component","name":"planning"},"to":{"kind":"file","name":"selection","path":"src/selection.cpp"},"relation":"depends_on","rationale":"inspect selection"})"}, error);
    if (!setup.ok) { std::cerr << "setup failed: " << error << '\n'; return 1; }
    auto authored_arguments = json::parse(R"({"from":{"kind":"actor","name":"Alice","source_ref":"case:a/alice"},"to":{"kind":"resource","name":"Paper","source_ref":"paper:a"},"relation":"authored","rationale":"source credits Alice"})");
    authored_arguments["sources"] = std::vector<common_source_reference>{source.get<common_source_reference>()};
    const auto authored = traced_call(*view, {"authored", "graph.link.propose", authored_arguments.dump()}, error);
    if (!authored.ok) { std::cerr << "authorship setup failed: " << error << '\n'; return 1; }
    const auto authored_result = json::parse(authored.content_json, nullptr, false);
    if (authored_result.is_discarded() || !authored_result.contains("result")) return 1;
    const auto edge = authored_result["result"]["edge"];
    const auto revision = authored_result["result"]["revision"];
    const auto description = traced_call(*view, {"describe", "graph.describe.propose", json{
        {"id", edge["from"]}, {"description", "Author of the research paper"},
        {"expected_revision", revision}, {"source_revision", "fixture-v1"}, {"evidence_refs", {"paper:a"}}}.dump()}, error);
    if (!description.ok) { std::cerr << "description proposal failed: " << error << '\n'; return 1; }
    auto read_context = context;
    agent_tool_context_apply_policy_gated_writes(read_context, false);
    auto read_view = provider.resolve_tools(read_context, error);
    if (!read_view) { std::cerr << "read-only view unavailable: " << error << '\n'; return 1; }
    const auto denied = traced_call(*read_view, {"denied", "graph.describe.propose", json{
        {"id", edge["from"]}, {"description", "Not authorized"}, {"expected_revision", revision}}.dump()}, error);
    if (denied.ok) { std::cerr << "read-only view allowed a description proposal\n"; return 1; }
    const auto described = traced_call(*view, {"find-description", "graph.find", R"({"query":"research paper"})"}, error);
    if (!described.ok) { std::cerr << "description lookup failed: " << error << '\n'; return 1; }
    const auto described_result = json::parse(described.content_json, nullptr, false);
    if (described_result.is_discarded() || described_result["result"]["nodes"].empty() ||
            described_result["result"]["nodes"][0]["id"] != edge["from"]) return 1;

    // Standalone notes need neither an invented edge nor a second store.
    const auto current_revision = json::parse(description.content_json)["result"]["revision"];
    const std::string long_note = "Purpose: standalone evidence. Invariant: proposed only. Dependencies: none known. Change hazard: verify sources. " + std::string(600, 'x');
    const auto isolated = traced_call(*view, {"isolated", "graph.describe.propose", json{
        {"node", {{"kind", "component"}, {"name", "isolated comprehension subject"}}},
        {"description", long_note}, {"source_revision", "fixture-v1"},
        {"evidence_refs", {"paper:a"}}, {"expected_revision", current_revision}}.dump()}, error);
    if (!isolated.ok) { std::cerr << "standalone proposal failed\n"; return 1; }
    const auto saved_isolated = json::parse(isolated.content_json)["result"];
    const auto isolated_id = saved_isolated["node"]["id"];
    const auto isolated_revision = saved_isolated["revision"];
    const auto replay = traced_call(*view, {"replay-isolated", "graph.describe.propose", json{
        {"id", isolated_id}, {"description", long_note}, {"source_revision", "fixture-v1"},
        {"evidence_refs", {"paper:a"}}, {"expected_revision", isolated_revision}}.dump()}, error);
    if (!replay.ok || !json::parse(replay.content_json)["result"].value("reused", false)) return 1;
    const auto isolated_read = traced_call(*read_view, {"read-isolated", "graph.get", json{{"id", isolated_id}}.dump()}, error);
    if (!isolated_read.ok || json::parse(isolated_read.content_json)["result"]["node"]["description"]["text"] != long_note) return 1;
    const auto isolated_expand = traced_call(*read_view, {"expand-isolated", "graph.expand", json{{"id", isolated_id}, {"depth", 1}}.dump()}, error);
    if (!isolated_expand.ok || !json::parse(isolated_expand.content_json)["result"]["edges"].empty()) return 1;

    common_agent_tool_call selected{"graph.search", R"({"relation":"authored"})"};
    if (!model.empty()) {
        if (!std::filesystem::is_regular_file(model)) { std::cerr << "model unavailable\n"; return 77; }
        const auto & available = view->chat_tools();
        const auto it = std::find_if(available.begin(), available.end(), [](const auto & tool) { return tool.name == "graph.search"; });
        if (it == available.end()) { std::cerr << "graph.search is unavailable\n"; return 1; }
        std::vector<std::string> unsupported;
        const auto instructions = common_render_model_tool_output_instructions(common_agent_tool_output_format::compact_dsl, {*it}, &unsupported, error);
        if (!error.empty() || !unsupported.empty()) { std::cerr << "tool contract unavailable: " << error << '\n'; return 1; }
        auto host = std::make_shared<common_agent_server_context_host>();
        common_agent_server_context_host_config config;
        config.context_key.load_key.model = model; config.context_key.load_key.n_gpu_layers = gpu_layers;
        config.context_key.load_key.fit_params = true; config.context_key.n_parallel = 1; config.context_key.n_sequences = 1;
        config.context_key.n_ctx = 2048; config.context_key.n_threads = 4; config.verbosity = LOG_LEVEL_DEBUG;
        if (!host->start(config, error)) { std::cerr << "model host failed: " << error << '\n'; return 1; }
        common_agent_inference_session session;
        if (!host->build_inference_session(session, error) || !session.inference) { std::cerr << error << '\n'; return 1; }
        common_agent_generation_request request;
        request.purpose = common_agent_generation_purpose::operation_selection;
        request.trace_id = "context-graph-smoke"; request.options.n_predict = 96; request.options.n_threads = 4;
        request.grammar = common_build_tool_dsl_grammar({*it}, common_tool_dsl_grammar_form::action, error);
        if (request.grammar.empty() || !error.empty()) { std::cerr << "tool grammar unavailable: " << error << '\n'; return 1; }
        request.messages = {{"system", instructions + "\nTool names are case sensitive. Use graph.search with only its relation argument. Do not add endpoint, kind, status, or limit filters. Return only the tool call."},
            {"user", "Find all graph relationships with the exact relation name authored. No other filters were requested. Emit the tool call now."}};
        json messages = json::array();
        for (const auto & message : request.messages) messages.push_back({{"role", message.role}, {"content", message.content}});
        trace << json{{"messages", messages}, {"event", "generation.request"}, {"model", model}, {"gpu_layers", gpu_layers},
            {"grammar", request.grammar}, {"instructions", instructions}, {"trace_id", request.trace_id}}.dump() << std::endl;
        const auto generated = session.inference->generate_result(request);
        trace << json{{"event", "generation.result"}, {"content", generated.content},
            {"ok", common_agent_generation_succeeded(generated)}}.dump() << std::endl;
        std::cout << "model_output=" << generated.content << '\n';
        if (!common_agent_generation_succeeded(generated) ||
                !common_parse_model_tool_call(common_agent_tool_output_format::compact_dsl, generated.content, {*it}, selected, error)) {
            std::cerr << "model did not produce a valid graph call: " << error << '\n'; return 1;
        }
    }
    if (selected.name != "graph.search") { std::cerr << "unexpected model tool " << selected.name << '\n'; return 1; }
    const auto found = traced_call(*view, {"find", selected.name, selected.arguments_json}, error);
    const auto result = json::parse(found.content_json, nullptr, false);
    if (!found.ok || result.is_discarded() || !result.contains("result") || result["result"].value("edges", json::array()).empty()) {
        std::cerr << "graph search returned no matching relations: " << error << '\n'; return 1;
    }
    const auto & currentness = result["result"]["edges"][0]["assertions"][0]["sources"][0]["currentness"];
    if (currentness.value("status", std::string{}) != (source.value("local_changes", false) ? "unknown" : "same_revision")) {
        std::cerr << "source currentness did not match the observed repository\n"; return 1;
    }
    for (const auto & field : {"from", "to"}) {
        const auto args = json{{field, edge[field]}, {"relation", "authored"}}.dump();
        const auto search = traced_call(*view, {"relations", "graph.search", args}, error);
        const auto payload = json::parse(search.content_json, nullptr, false);
        if (!search.ok || payload.is_discarded() || payload["result"]["edges"].size() != 1) {
            std::cerr << "authorship relation search failed: " << error << '\n'; return 1;
        }
    }
    const auto expanded = traced_call(*view, {"paper", "graph.expand", json{{"id",edge["to"]}}.dump()}, error);
    if (!expanded.ok) { std::cerr << "paper expansion failed: " << error << '\n'; return 1; }
    for (int i = 0; i < 6; ++i) {
        const auto added = traced_call(*view, {"candidate", "graph.link.propose", json{
            {"from", {{"kind", "component"}, {"name", "fixture-candidate-" + std::to_string(i)}}},
            {"to", {{"id", edge["to"]}}}, {"relation", "related_to"}, {"rationale", "selection fixture"}}.dump()}, error);
        if (!added.ok) { std::cerr << "candidate setup failed: " << error << '\n'; return 1; }
    }
    const auto limited = traced_call(*view, {"limited", "graph.find", R"({"query":"fixture-candidate","limit":5})"}, error);
    const auto limited_result = json::parse(limited.content_json, nullptr, false);
    if (!limited.ok || limited_result.is_discarded() || limited_result["result"]["nodes"].size() != 5 ||
            limited_result["result"]["truncated"] != true) {
        std::cerr << "bounded graph lookup did not report further matches: " << error << '\n'; return 1;
    }
    std::cout << "context graph smoke passed; mode=" << (model.empty() ? "no-model" : "model") << '\n';
    return 0;
}
