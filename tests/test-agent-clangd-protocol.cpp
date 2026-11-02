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

static int fake_clangd() {
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
        if (method == "initialize") send({{"jsonrpc", "2.0"}, {"id", message["id"]}, {"result", {{"capabilities", agent_clangd_json::object()}}}});
        else if (method == "workspace/symbol") {
            send({{"jsonrpc", "2.0"}, {"id", 999}, {"method", "window/workDoneProgress/create"}, {"params", {{"token", "index"}}}});
            send(agent_clangd_notification("$/progress", {{"token", "index"}, {"value", {{"kind", "begin"}, {"title", "Background Index"}}}}));
            send(agent_clangd_notification("$/progress", {{"token", "index"}, {"value", {{"kind", "report"}}}}));
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            send(agent_clangd_notification("$/progress", {{"token", "index"}, {"value", {{"kind", "end"}}}}));
            send({{"jsonrpc", "2.0"}, {"id", message["id"]}, {"result", agent_clangd_json::array()}});
        } else if (method == "shutdown") send({{"jsonrpc", "2.0"}, {"id", message["id"]}, {"result", nullptr}});
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
    assert(!unavailable.request("workspace/symbol", {{"query", "needle"}}, decoded, error));
    assert(unavailable.status_snapshot()["index_activity"] == "unknown");

    {
        agent_clangd_session fake({std::filesystem::absolute(argv[0]).string(), "", "", 5000, 1000, "", "", {}});
        assert(fake.status_snapshot()["index_activity"] == "unknown");
        bool success = false;
        std::thread worker([&] { agent_clangd_json reply; std::string failure; success = fake.request("workspace/symbol", {{"query", "test"}}, reply, failure); });
        bool observed = false;
        for (int i = 0; i < 100; ++i) {
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
        { std::ofstream output(file); output << "int old_value;\n"; }
#ifdef _WIN32
        _putenv_s("LLAMA_TEST_CLANGD_LOG", log_path.string().c_str());
#else
        setenv("LLAMA_TEST_CLANGD_LOG", log_path.string().c_str(), 1);
#endif
        {
            agent_clangd_session fake({std::filesystem::absolute(argv[0]).string(), root.string(), "", 5000, 1000, "", "", {}});
            const auto uri = agent_clangd_file_uri(file);
            assert(fake.open_document(uri, "cpp", "int old_value;\n", error));
            { std::ofstream output(file); output << "int new_value;\n"; }
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
        std::filesystem::remove_all(root);
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
            { std::ofstream output(database); output << "invalid json"; }
            assert(!provider.refresh().ok);
            assert(provider.symbol(R"({"symbol":"fixture"})").ok);
            const auto status = provider.status_snapshot();
            assert(status["compile_commands_may_be_stale"] == true);
            assert(status["compile_commands_sync_error"].is_string());
            std::ifstream copy(cached);
            assert(agent_clangd_json::parse(copy) == entries);
        }
        fs::remove_all(root);
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
