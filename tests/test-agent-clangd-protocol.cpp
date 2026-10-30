#include "agent-clangd-protocol.h"
#include "agent-clangd-session.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <thread>

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

    agent_clangd_session unavailable({"__llama_missing_clangd__", "", "", 100, 100});
    assert(unavailable.status_snapshot()["session"] == "not_running");
    assert(!unavailable.request("workspace/symbol", {{"query", "needle"}}, decoded, error));
    assert(unavailable.status_snapshot()["index_activity"] == "unknown");

    {
        agent_clangd_session fake({std::filesystem::absolute(argv[0]).string(), "", "", 5000, 1000});
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

    if (std::getenv("LLAMA_AGENT_CLANGD_LIVE_TEST") != nullptr) {
        agent_clangd_session live({"clangd", "", "", 5000, 5000});
        error.clear();
        if (!live.request("workspace/symbol", {{"query", "common_schema_normalize"}}, decoded, error)) {
            std::fprintf(stderr, "clangd live request failed: %s\n", error.c_str());
            return 2;
        }
    }
    return 0;
}
