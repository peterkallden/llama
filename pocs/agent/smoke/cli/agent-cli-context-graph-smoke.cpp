#include "agent/tooling/catalog/tool-catalog.h"
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
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) model = argv[++i];
        else if (arg == "--n-gpu-layers" && i + 1 < argc) gpu_layers = std::stoi(argv[++i]);
        else if (arg == "--no-model") model.clear();
        else { std::cerr << "usage: " << argv[0] << " [--no-model | --model PATH] [--n-gpu-layers N]\n"; return 2; }
    }
    std::string error;
    common_tool_catalog catalog; common_tool_bootstrap_result bootstrap;
    if (!catalog.bootstrap("graph-project", bootstrap, error)) { std::cerr << error << '\n'; return 1; }
    smoke_store store;
    native_agent_tool_provider provider(catalog, [&](const agent_tool_context & context, common_native_tool_bindings & bindings, std::string &) {
        bindings.data_store = &store; bindings.tool_scope = context.scope; return true;
    });
    agent_tool_context context; context.profile_id = "graph-project"; context.scope.namespace_id = "smoke";
    context.scope.project_id = "project"; context.max_calls = 8; agent_tool_context_apply_policy_gated_writes(context, true);
    auto view = provider.resolve_tools(context, error);
    if (!view) { std::cerr << error << '\n'; return 1; }
    const auto setup = view->call({"setup", "graph.link.propose", R"({"from":{"kind":"component","name":"planning"},"to":{"kind":"file","name":"selection","path":"src/selection.cpp"},"relation":"depends_on","rationale":"inspect selection"})"}, error);
    if (!setup.ok) { std::cerr << "setup failed: " << error << '\n'; return 1; }
    const auto authored = view->call({"authored", "graph.link.propose", R"({"from":{"kind":"actor","name":"Alice","source_ref":"case:a/alice"},"to":{"kind":"resource","name":"Paper","source_ref":"paper:a"},"relation":"authored","rationale":"source credits Alice"})"}, error);
    if (!authored.ok) { std::cerr << "authorship setup failed: " << error << '\n'; return 1; }
    const auto authored_result = json::parse(authored.content_json, nullptr, false);
    if (authored_result.is_discarded() || !authored_result.contains("result")) return 1;
    const auto edge = authored_result["result"]["edge"];

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
        config.context_key.n_ctx = 2048; config.context_key.n_threads = 4; config.verbosity = LOG_LEVEL_WARN;
        if (!host->start(config, error)) { std::cerr << "model host failed: " << error << '\n'; return 1; }
        common_agent_inference_session session;
        if (!host->build_inference_session(session, error) || !session.inference) { std::cerr << error << '\n'; return 1; }
        common_agent_generation_request request;
        request.purpose = common_agent_generation_purpose::operation_selection;
        request.trace_id = "context-graph-smoke"; request.options.n_predict = 96; request.options.n_threads = 4;
        request.grammar = common_build_tool_dsl_grammar({*it}, common_tool_dsl_grammar_form::action, error);
        if (request.grammar.empty() || !error.empty()) { std::cerr << "tool grammar unavailable: " << error << '\n'; return 1; }
        request.messages = {{"system", instructions + "\nTool names are case sensitive. Use graph.search with its relation argument. Do not invent endpoint IDs. Return only the tool call."},
            {"user", "Find all graph relationships with the exact relation name authored. No source or target node was provided, so omit from and to. Emit the tool call now."}};
        const auto generated = session.inference->generate_result(request);
        std::cout << "model_output=" << generated.content << '\n';
        if (!common_agent_generation_succeeded(generated) ||
                !common_parse_model_tool_call(common_agent_tool_output_format::compact_dsl, generated.content, {*it}, selected, error)) {
            std::cerr << "model did not produce a valid graph call: " << error << '\n'; return 1;
        }
    }
    if (selected.name != "graph.search") { std::cerr << "unexpected model tool " << selected.name << '\n'; return 1; }
    const auto found = view->call({"find", selected.name, selected.arguments_json}, error);
    const auto result = json::parse(found.content_json, nullptr, false);
    if (!found.ok || result.is_discarded() || !result.contains("result") || result["result"].value("edges", json::array()).empty()) {
        std::cerr << "graph search returned no matching relations: " << error << '\n'; return 1;
    }
    for (const auto & field : {"from", "to"}) {
        const auto args = json{{field, edge[field]}, {"relation", "authored"}}.dump();
        const auto search = view->call({"relations", "graph.search", args}, error);
        const auto payload = json::parse(search.content_json, nullptr, false);
        if (!search.ok || payload.is_discarded() || payload["result"]["edges"].size() != 1) {
            std::cerr << "authorship relation search failed: " << error << '\n'; return 1;
        }
    }
    const auto expanded = view->call({"paper", "graph.expand", json{{"id",edge["to"]}}.dump()}, error);
    if (!expanded.ok) { std::cerr << "paper expansion failed: " << error << '\n'; return 1; }
    std::cout << "context graph smoke passed; mode=" << (model.empty() ? "no-model" : "model") << '\n';
    return 0;
}
