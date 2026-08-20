#pragma once

#include "../daemon/agent-daemon-jsonl-protocol.h"

#include <functional>
#include <nlohmann/json.hpp>
#include <string>

class agent_daemon_client_admin {
public:
    using request_sender =
        std::function<bool(
            const nlohmann::ordered_json & request,
            nlohmann::ordered_json & response,
            std::string & error)>;

    explicit agent_daemon_client_admin(request_sender send_request);

    bool status(agent_daemon_jsonl_status_response & response, std::string & error) const;
    bool list_sessions(agent_daemon_jsonl_status_response & response, std::string & error) const;
    bool get_session(
        const std::string & session_id,
        const std::string & namespace_id,
        agent_daemon_jsonl_status_response & response,
        std::string & error) const;
    bool list_resources(
        const agent_daemon_jsonl_list_resources_request & request,
        agent_daemon_jsonl_listing_response & response,
        std::string & error) const;
    bool list_memories(
        const agent_daemon_jsonl_list_memories_request & request,
        agent_daemon_jsonl_listing_response & response,
        std::string & error) const;
    bool list_plans(
        const agent_daemon_jsonl_list_plans_request & request,
        agent_daemon_jsonl_listing_response & response,
        std::string & error) const;
    bool read_resource(
        const agent_daemon_jsonl_read_resource_request & request,
        agent_daemon_jsonl_resource_response & response,
        std::string & error) const;
    bool put_resource(
        const agent_daemon_jsonl_put_resource_request & request,
        agent_daemon_jsonl_resource_response & response,
        std::string & error) const;
    bool drain(agent_daemon_jsonl_lifecycle_response & response, std::string & error) const;
    bool reset_session(
        const std::string & session_id,
        const std::string & namespace_id,
        agent_daemon_jsonl_lifecycle_response & response,
        std::string & error) const;
    bool close_session(
        const std::string & session_id,
        const std::string & namespace_id,
        agent_daemon_jsonl_lifecycle_response & response,
        std::string & error) const;
    bool shutdown(agent_daemon_jsonl_lifecycle_response & response, std::string & error) const;

private:
    request_sender send_request_;
};
