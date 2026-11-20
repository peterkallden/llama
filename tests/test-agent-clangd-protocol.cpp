#include "agent-clangd-protocol.h"
#include "agent-clangd-session.h"
#include "agent-clangd-provider.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <fstream>
#include <sstream>
#include <thread>
#include <utility>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

// Keep Windows CI failures actionable: MSVC's assert abort is surfaced by CTest
// as STATUS_STACK_BUFFER_OVERRUN, which otherwise hides the failed expression.
#undef assert
#define assert(...) do { \
    if (!(__VA_ARGS__)) { \
        std::fprintf(stderr, "assertion failed at %s:%d: %s\n", __FILE__, __LINE__, #__VA_ARGS__); \
        std::fflush(stderr); \
        std::abort(); \
    } \
} while (false)

static int fake_clangd() {
#ifdef _WIN32
    // LSP framing is byte-oriented. The CRT's default text mode rewrites
    // CRLF on stdin/stdout and corrupts Content-Length framed messages.
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
#endif
    agent_clangd_message_decoder decoder;
    std::string error;
    agent_clangd_json message;
    auto send = [](const agent_clangd_json & value) {
        const auto encoded = agent_clangd_encode_message(value);
        std::fwrite(encoded.data(), 1, encoded.size(), stdout);
        std::fflush(stdout);
    };
    for (;;) {
        if (!decoder.pop(message)) {
            const int byte = std::getchar();
            if (byte == EOF) return 0;
            if (!decoder.feed(std::string(1, static_cast<char>(byte)), error)) return 1;
            continue;
        }
        if (!message.contains("method")) continue;
        const auto method = message["method"].get<std::string>();
        if (const char * log_path = std::getenv("LLAMA_TEST_CLANGD_LOG")) {
            std::ofstream log(log_path, std::ios::app);
            log << message.dump() << '\n';
        }
        if (std::getenv("LLAMA_TEST_CLANGD_CONTEXT") && (method == "textDocument/documentSymbol" || method == "textDocument/references")) {
            using json = agent_clangd_json;
            const auto range = [](int first, int column, int last, int end_column) {
                return json{{"start", {{"line", first}, {"character", column}}}, {"end", {{"line", last}, {"character", end_column}}}};
            };
            const auto function = json{{"name", "fixture"}, {"kind", 12}, {"detail", "int ()"},
                {"range", range(1, 0, 4, 0)}, {"selectionRange", range(1, 4, 1, 11)}};
            const auto symbols = json::array({{{"name", "demo"}, {"kind", 3}, {"range", range(0, 0, 5, 0)},
                {"selectionRange", range(0, 10, 0, 14)}, {"children", json::array({function})}}});
            const auto result = method == "textDocument/documentSymbol" ? symbols : json::array({{
                {"uri", message["params"]["textDocument"]["uri"]}, {"range", range(2, 8, 2, 15)}}});
            send({{"jsonrpc", "2.0"}, {"id", message["id"]}, {"result", result}});
            continue;
        }
        if (method == "initialize") send({{"jsonrpc", "2.0"}, {"id", message["id"]}, {"result", {{"capabilities", agent_clangd_json::object()}}}});
        else if (method == "workspace/symbol") {
            send({{"jsonrpc", "2.0"}, {"id", 999}, {"method", "window/workDoneProgress/create"}, {"params", {{"token", "index"}}}});
            send(agent_clangd_notification("$/progress", {{"token", "index"}, {"value", {{"kind", "begin"}, {"title", "Background Index"}}}}));
            send(agent_clangd_notification("$/progress", {{"token", "index"}, {"value", {{"kind", "report"}}}}));
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            send(agent_clangd_notification("$/progress", {{"token", "index"}, {"value", {{"kind", "end"}}}}));
            send({{"jsonrpc", "2.0"}, {"id", message["id"]}, {"result", agent_clangd_json::array()}});
        } else if (method == "textDocument/references" || method == "textDocument/prepareCallHierarchy" ||
                method == "textDocument/documentSymbol")
            send({{"jsonrpc", "2.0"}, {"id", message["id"]}, {"result", agent_clangd_json::array()}});
        else if (method == "shutdown") send({{"jsonrpc", "2.0"}, {"id", message["id"]}, {"result", nullptr}});
        else if (method == "exit") return 0;
    }
}

int main(int argc, char ** argv) {
    if (argc > 1 && std::string(argv[1]) == "--offset-encoding=utf-16") return fake_clangd();
    const auto request = agent_clangd_request(7, "textDocument/definition", {{"textDocument", {{"uri", "file:///workspace/main.cpp"}}}});
    const auto encoded = agent_clangd_encode_message(request);
    agent_clangd_message_decoder decoder;
    std::string error;
    assert(decoder.feed(encoded.substr(0, 11), error));
    assert(decoder.feed(encoded.substr(11), error));
    agent_clangd_json decoded;
    assert(decoder.pop(decoded));
    assert(decoded["id"] == 7 && decoded["method"] == "textDocument/definition");

    const auto notification = agent_clangd_encode_message(agent_clangd_notification("initialized"));
    assert(decoder.feed(notification + encoded, error));
    assert(decoder.pop(decoded) && decoded["method"] == "initialized" && !decoded.contains("id"));
    assert(decoder.pop(decoded) && decoded["id"] == 7);

    agent_clangd_message_decoder limited(8);
    assert(!limited.feed(agent_clangd_encode_message({{"jsonrpc", "2.0"}, {"result", "too long"}}), error));

    agent_clangd_session unavailable({"__llama_missing_clangd__", "", "", 100, 100, "", "", {}});
    assert(unavailable.status_snapshot()["session"] == "not_running");
    assert(unavailable.status_snapshot()["start_on"] == "semantic_query");
    assert(!unavailable.request("workspace/symbol", {{"query", "needle"}}, decoded, error));
    assert(unavailable.status_snapshot()["index_activity"] == "unknown");

    {
        agent_clangd_session fake({std::filesystem::absolute(argv[0]).string(), "", "", 5000, 1000, "", "", {}});
        assert(fake.status_snapshot()["index_activity"] == "unknown");
        bool success = false;
        std::thread worker([&] { agent_clangd_json reply; std::string failure; success = fake.request("workspace/symbol", {{"query", "test"}}, reply, failure); });
        bool observed = false;
        // The test executable also serves as clangd and starts under the same
        // sanitizer instrumentation. Allow process startup before asserting on
        // the progress notification rather than assuming it fits in 500 ms.
        for (int i = 0; i < 1000; ++i) {
            if (fake.status_snapshot()["index_activity"] == "active") { observed = true; break; }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        assert(observed);
        const auto during = fake.status_snapshot();
        assert(during["session"] == "initialized" && during["observation_may_be_stale"] == true);
        worker.join();
        assert(success && fake.status_snapshot()["index_activity"] == "ended");
    }

    {
        const auto root = std::filesystem::temp_directory_path() /
            ("llama-clangd-sync-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(root);
        const auto file = root / "main.cpp";
        const auto log_path = root / "lsp.log";
        { std::ofstream output(file, std::ios::binary); output << "int old_value;\n"; }
#ifdef _WIN32
        _putenv_s("LLAMA_TEST_CLANGD_LOG", log_path.string().c_str());
#else
        setenv("LLAMA_TEST_CLANGD_LOG", log_path.string().c_str(), 1);
#endif
        {
            agent_clangd_session fake({std::filesystem::absolute(argv[0]).string(), root.string(), "", 5000, 1000, "", "", {}});
            const auto uri = agent_clangd_file_uri(file);
            assert(fake.open_document(uri, "cpp", "int old_value;\n", error));
            { std::ofstream output(file, std::ios::binary); output << "int new_value;\n"; }
            assert(fake.apply_file_changes({{agent_project_file_change_kind::changed, file.string(), {}, 1}}, error));
            std::filesystem::remove(file);
            assert(fake.apply_file_changes({{agent_project_file_change_kind::removed, file.string(), {}, 2}}, error));
            fake.stop_for_refresh();
        }
#ifdef _WIN32
        _putenv_s("LLAMA_TEST_CLANGD_LOG", "");
#else
        unsetenv("LLAMA_TEST_CLANGD_LOG");
#endif
        std::ifstream input(log_path);
        const std::string lsp_log((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
        assert(lsp_log.find("textDocument/didOpen") != std::string::npos);
        assert(lsp_log.find("textDocument/didChange") != std::string::npos);
        assert(lsp_log.find("workspace/didChangeWatchedFiles") != std::string::npos);
        assert(lsp_log.find("textDocument/didClose") != std::string::npos);
        bool saw_versioned_change = false, saw_delete_notification = false;
        std::istringstream messages(lsp_log);
        std::string line;
        while (std::getline(messages, line)) {
            const auto message = agent_clangd_json::parse(line, nullptr, false);
            assert(message.is_object());
            if (message.value("method", std::string()) == "textDocument/didChange") {
                const auto & params = message.at("params");
                assert(params.at("textDocument").at("uri") == agent_clangd_file_uri(file));
                assert(params.at("textDocument").at("version") == 2);
                assert(params.at("contentChanges").at(0).at("text") == "int new_value;\n");
                saw_versioned_change = true;
            }
            if (message.value("method", std::string()) == "workspace/didChangeWatchedFiles") {
                for (const auto & change : message.at("params").at("changes"))
                    if (change.at("uri") == agent_clangd_file_uri(file) && change.at("type") == 3)
                        saw_delete_notification = true;
            }
        }
        assert(saw_versioned_change && saw_delete_notification);
        input.close();
        std::error_code cleanup_error;
        std::filesystem::remove_all(root, cleanup_error);
        assert(!cleanup_error);
    }

    {
        namespace fs = std::filesystem;
        const auto root = fs::temp_directory_path() / ("llama-clangd-context-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::create_directories(root);
        { std::ofstream file(root / "source.cpp"); file << "namespace demo {\nint fixture() {\n return fixture();\n}\n}\n"; }
#ifdef _WIN32
        _putenv_s("LLAMA_TEST_CLANGD_CONTEXT", "1");
#else
        setenv("LLAMA_TEST_CLANGD_CONTEXT", "1", 1);
#endif
        {
            agent_clangd_code_provider provider({fs::absolute(argv[0]).string(), root.string(), "", 5000, 1000, "", "", {}});
            const auto compact = provider.references(R"({"symbol":"fixture","path_hint":"source.cpp","line":2})");
            assert(compact.ok && !agent_clangd_json::parse(compact.output)["results"][0].contains("context"));
            const auto contextual = provider.references(R"({"symbol":"fixture","path_hint":"source.cpp","line":2,"include_context":true})");
            assert(contextual.ok);
            const auto item = agent_clangd_json::parse(contextual.output)["results"][0];
            assert(item["context_status"] == "available" && item["context"]["lines"].size() == 5);
            assert(item["enclosing_function"]["name"] == "fixture" && item["enclosing_function"]["container_name"] == "demo");
            assert(item["enclosing_function_status"] == "found");
            const auto symbols = provider.symbol(R"({"symbol":"fixture","path_hint":"source.cpp"})");
            assert(symbols.ok);
            const auto definition = agent_clangd_json::parse(symbols.output)["definitions"][0];
            assert(definition["container_name"] == "demo" && definition["detail"] == "int ()");
        }
#ifdef _WIN32
        _putenv_s("LLAMA_TEST_CLANGD_CONTEXT", "");
#else
        unsetenv("LLAMA_TEST_CLANGD_CONTEXT");
#endif
        fs::remove_all(root);
    }
    {
        namespace fs = std::filesystem;
        const auto root = fs::temp_directory_path() /
            ("llama-clangd-database-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::create_directories(root);
        const auto source_file = root / "source.cpp";
        const auto database = root / "source-db.json";
        const auto cached = root / "cache" / "compile_commands.json";
        { std::ofstream source(source_file); source << "int fixture;\n"; }
        const agent_clangd_json entries = agent_clangd_json::array({{
            {"directory", root.string()}, {"file", source_file.string()}, {"command", "c++ -c source.cpp"}}});
        { std::ofstream output(database); output << entries.dump(); }
        agent_clangd_session_config config{fs::absolute(argv[0]).string(), root.string(), cached.parent_path().string(), 5000, 1000, database.string(), cached.string(), {}};
        {
            agent_clangd_code_provider provider(std::move(config));
            assert(provider.refresh().ok);
            assert(fs::is_regular_file(cached));
            // Both public position forms reach the provider unchanged. A stale
            // line or column must explain the actual problem to MCP clients.
            const auto reference = provider.references(R"({"symbol":"fixture","path_hint":"source.cpp","line":1,"column":5})");
            assert(reference.ok);
            assert(provider.references(R"({"symbol":"fixture","definition_path":"source.cpp","definition_line":1})").ok);
            const auto calls = provider.call_hierarchy(R"({"symbol":"fixture","definition_path":"source.cpp","definition_line":1,"definition_column":5})");
            assert(calls.ok);
            assert(provider.call_hierarchy(R"({"symbol":"fixture","path_hint":"source.cpp","line":1})").ok);
            const auto stale = provider.references(R"({"symbol":"missing","definition_path":"source.cpp","definition_line":1})");
            assert(!stale.ok && stale.safe_summary.find("location may be stale") != std::string::npos);
            const auto column = provider.call_hierarchy(R"({"symbol":"fixture","path_hint":"source.cpp","line":1,"column":6})");
            assert(!column.ok && column.safe_summary.find("UTF-16 column") != std::string::npos);
            const auto no_location = provider.references(R"({"symbol":"fixture"})");
            assert(!no_location.ok && no_location.safe_summary.find("requires a source location") != std::string::npos);
            { std::ofstream output(database); output << "invalid json"; }
            assert(!provider.refresh().ok);
            assert(provider.symbol(R"({"symbol":"fixture"})").ok);
            const auto status = provider.status_snapshot();
            assert(status["compile_commands_may_be_stale"] == true);
            assert(status["compile_commands_sync_error"].is_string());
            std::ifstream copy(cached);
            assert(agent_clangd_json::parse(copy) == entries);
            copy.close();
        }
        std::error_code cleanup_error;
        fs::remove_all(root, cleanup_error);
        assert(!cleanup_error);
    }

    if (std::getenv("LLAMA_AGENT_CLANGD_LIVE_TEST") != nullptr) {
        agent_clangd_session live({"clangd", "", "", 5000, 5000, "", "", {}});
        error.clear();
        if (!live.request("workspace/symbol", {{"query", "common_schema_normalize"}}, decoded, error)) {
            std::fprintf(stderr, "clangd live request failed: %s\n", error.c_str());
            return 2;
        }
    }
    return 0;
}
