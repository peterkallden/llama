#include "agent-clangd-protocol.h"

#include <algorithm>

namespace {
constexpr size_t max_header_bytes = 8192;
}

agent_clangd_message_decoder::agent_clangd_message_decoder(size_t max_message_bytes)
    : max_message_bytes_(max_message_bytes) {}

bool agent_clangd_message_decoder::feed(const char * data, size_t size, std::string & error) {
    if (size == 0) return true;
    if (buffer_.size() + size > max_message_bytes_ + max_header_bytes) {
        error = "clangd protocol buffer exceeds its limit";
        return false;
    }
    buffer_.append(data, size);

    while (true) {
        const auto header_end = buffer_.find("\r\n\r\n");
        if (header_end == std::string::npos) {
            if (buffer_.size() > max_header_bytes) {
                error = "clangd protocol header exceeds its limit";
                return false;
            }
            return true;
        }
        const auto header = buffer_.substr(0, header_end);
        const auto prefix = std::string("Content-Length:");
        const auto length_start = header.find(prefix);
        if (length_start == std::string::npos) {
            error = "clangd protocol message has no Content-Length";
            return false;
        }
        auto value_start = length_start + prefix.size();
        while (value_start < header.size() && (header[value_start] == ' ' || header[value_start] == '\t')) ++value_start;
        const auto value_end = header.find('\r', value_start);
        const auto value = header.substr(value_start, value_end == std::string::npos ? std::string::npos : value_end - value_start);
        size_t content_length = 0;
        try {
            size_t parsed = 0;
            content_length = std::stoull(value, &parsed);
            if (parsed != value.size()) throw std::invalid_argument("trailing data");
        } catch (...) {
            error = "clangd protocol Content-Length is invalid";
            return false;
        }
        if (content_length > max_message_bytes_) {
            error = "clangd protocol message exceeds its limit";
            return false;
        }
        const auto message_start = header_end + 4;
        if (buffer_.size() < message_start + content_length) return true;
        const auto payload = buffer_.substr(message_start, content_length);
        const auto message = agent_clangd_json::parse(payload, nullptr, false);
        if (message.is_discarded() || !message.is_object()) {
            error = "clangd protocol payload is not a JSON object";
            return false;
        }
        messages_.push_back(message);
        buffer_.erase(0, message_start + content_length);
    }
}

bool agent_clangd_message_decoder::pop(agent_clangd_json & message) {
    if (messages_.empty()) return false;
    message = std::move(messages_.front());
    messages_.pop_front();
    return true;
}

std::string agent_clangd_encode_message(const agent_clangd_json & message) {
    const auto payload = message.dump();
    return "Content-Length: " + std::to_string(payload.size()) + "\r\n\r\n" + payload;
}

agent_clangd_json agent_clangd_request(int64_t id, const std::string & method, const agent_clangd_json & params) {
    return {{"jsonrpc", "2.0"}, {"id", id}, {"method", method}, {"params", params}};
}

agent_clangd_json agent_clangd_notification(const std::string & method, const agent_clangd_json & params) {
    return {{"jsonrpc", "2.0"}, {"method", method}, {"params", params}};
}
