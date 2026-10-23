#pragma once

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <deque>
#include <string>

using agent_clangd_json = nlohmann::ordered_json;

class agent_clangd_message_decoder {
public:
    explicit agent_clangd_message_decoder(size_t max_message_bytes = 4 * 1024 * 1024);

    bool feed(const char * data, size_t size, std::string & error);
    bool feed(const std::string & data, std::string & error) { return feed(data.data(), data.size(), error); }
    bool pop(agent_clangd_json & message);
    size_t buffered_bytes() const { return buffer_.size(); }

private:
    size_t max_message_bytes_;
    std::string buffer_;
    std::deque<agent_clangd_json> messages_;
};

std::string agent_clangd_encode_message(const agent_clangd_json & message);
agent_clangd_json agent_clangd_request(int64_t id, const std::string & method, const agent_clangd_json & params = agent_clangd_json::object());
agent_clangd_json agent_clangd_notification(const std::string & method, const agent_clangd_json & params = agent_clangd_json::object());
