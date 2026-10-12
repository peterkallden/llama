#include "agent-clangd-protocol.h"
#include "agent-clangd-session.h"

#include <cassert>

int main() {
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
    assert(!unavailable.request("workspace/symbol", {{"query", "needle"}}, decoded, error));
    return 0;
}
