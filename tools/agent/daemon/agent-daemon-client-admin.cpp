#include "agent-daemon-client-admin.h"

using json = nlohmann::ordered_json;

namespace {

template <typename Response, typename Parser>
bool parse_response_or_append_message(
        const json & message,
        Response & response,
        std::string & error,
        const Parser & parser) {
    if (parser(message, response, error)) {
        return true;
    }
    error += ": " + message.dump();
    return false;
}

} // namespace

agent_daemon_client_admin::agent_daemon_client_admin(request_sender send_request) :
        send_request_(std::move(send_request)) {
}

bool agent_daemon_client_admin::status(
        agent_daemon_jsonl_status_response & response,
        std::string & error) const {
    json message;
    if (!send_request_(make_agent_daemon_jsonl_status_request({}), message, error)) {
        return false;
    }
    return parse_response_or_append_message(
        message,
        response,
        error,
        parse_agent_daemon_jsonl_status_response);
}

bool agent_daemon_client_admin::list_sessions(
        agent_daemon_jsonl_status_response & response,
        std::string & error) const {
    json message;
    if (!send_request_(make_agent_daemon_jsonl_list_sessions_request({}), message, error)) {
        return false;
    }
    if (!parse_response_or_append_message(
                message,
                response,
                error,
                parse_agent_daemon_jsonl_status_response)) {
        return false;
    }
    if (response.event != "sessions_listed") {
        error = "unexpected daemon list_sessions response: " + message.dump();
        return false;
    }
    return true;
}

bool agent_daemon_client_admin::get_session(
        const std::string & session_id,
        const std::string & namespace_id,
        agent_daemon_jsonl_status_response & response,
        std::string & error) const {
    json message;
    if (!send_request_(
                make_agent_daemon_jsonl_get_session_request({
                    session_id,
                    namespace_id,
                }),
                message,
                error)) {
        return false;
    }
    if (!parse_response_or_append_message(
                message,
                response,
                error,
                parse_agent_daemon_jsonl_status_response)) {
        return false;
    }
    if (response.event != "session_found") {
        error = "unexpected daemon get_session response: " + message.dump();
        return false;
    }
    return true;
}

bool agent_daemon_client_admin::list_resources(
        const agent_daemon_jsonl_list_resources_request & request,
        agent_daemon_jsonl_listing_response & response,
        std::string & error) const {
    json message;
    if (!send_request_(make_agent_daemon_jsonl_list_resources_request(request), message, error)) {
        return false;
    }
    if (!parse_response_or_append_message(
                message,
                response,
                error,
                parse_agent_daemon_jsonl_listing_response)) {
        return false;
    }
    if (response.event != "resources_listed") {
        error = "unexpected daemon list_resources response: " + message.dump();
        return false;
    }
    return true;
}

bool agent_daemon_client_admin::list_memories(
        const agent_daemon_jsonl_list_memories_request & request,
        agent_daemon_jsonl_listing_response & response,
        std::string & error) const {
    json message;
    if (!send_request_(make_agent_daemon_jsonl_list_memories_request(request), message, error)) {
        return false;
    }
    if (!parse_response_or_append_message(
                message,
                response,
                error,
                parse_agent_daemon_jsonl_listing_response)) {
        return false;
    }
    if (response.event != "memories_listed") {
        error = "unexpected daemon list_memories response: " + message.dump();
        return false;
    }
    return true;
}

bool agent_daemon_client_admin::list_plans(
        const agent_daemon_jsonl_list_plans_request & request,
        agent_daemon_jsonl_listing_response & response,
        std::string & error) const {
    json message;
    if (!send_request_(make_agent_daemon_jsonl_list_plans_request(request), message, error)) {
        return false;
    }
    if (!parse_response_or_append_message(
                message,
                response,
                error,
                parse_agent_daemon_jsonl_listing_response)) {
        return false;
    }
    if (response.event != "plans_listed") {
        error = "unexpected daemon list_plans response: " + message.dump();
        return false;
    }
    return true;
}

bool agent_daemon_client_admin::read_resource(
        const agent_daemon_jsonl_read_resource_request & request,
        agent_daemon_jsonl_resource_response & response,
        std::string & error) const {
    json message;
    if (!send_request_(make_agent_daemon_jsonl_read_resource_request(request), message, error)) {
        return false;
    }
    if (!parse_response_or_append_message(
                message,
                response,
                error,
                parse_agent_daemon_jsonl_resource_response)) {
        return false;
    }
    if (response.event != "resource_read") {
        error = "unexpected daemon read_resource response: " + message.dump();
        return false;
    }
    return true;
}

bool agent_daemon_client_admin::put_resource(
        const agent_daemon_jsonl_put_resource_request & request,
        agent_daemon_jsonl_resource_response & response,
        std::string & error) const {
    json message;
    if (!send_request_(make_agent_daemon_jsonl_put_resource_request(request), message, error)) {
        return false;
    }
    if (!parse_response_or_append_message(
                message,
                response,
                error,
                parse_agent_daemon_jsonl_resource_response)) {
        return false;
    }
    if (response.event != "resource_created") {
        error = "unexpected daemon put_resource response: " + message.dump();
        return false;
    }
    return true;
}

bool agent_daemon_client_admin::drain(
        agent_daemon_jsonl_lifecycle_response & response,
        std::string & error) const {
    json message;
    if (!send_request_(make_agent_daemon_jsonl_drain_request({}), message, error)) {
        return false;
    }
    if (!parse_response_or_append_message(
                message,
                response,
                error,
                parse_agent_daemon_jsonl_lifecycle_response)) {
        return false;
    }
    if (response.event != "drain") {
        error = "unexpected daemon drain response: " + message.dump();
        return false;
    }
    return true;
}

bool agent_daemon_client_admin::reset_session(
        const std::string & session_id,
        const std::string & namespace_id,
        agent_daemon_jsonl_lifecycle_response & response,
        std::string & error) const {
    json message;
    if (!send_request_(
                make_agent_daemon_jsonl_reset_session_request(
                    session_id,
                    namespace_id),
                message,
                error)) {
        return false;
    }
    if (!parse_response_or_append_message(
                message,
                response,
                error,
                parse_agent_daemon_jsonl_lifecycle_response)) {
        return false;
    }
    if (response.event != "session_reset") {
        error = "unexpected daemon session_reset response: " + message.dump();
        return false;
    }
    return true;
}

bool agent_daemon_client_admin::close_session(
        const std::string & session_id,
        const std::string & namespace_id,
        agent_daemon_jsonl_lifecycle_response & response,
        std::string & error) const {
    json message;
    if (!send_request_(
                make_agent_daemon_jsonl_close_session_request(
                    session_id,
                    namespace_id),
                message,
                error)) {
        return false;
    }
    if (!parse_response_or_append_message(
                message,
                response,
                error,
                parse_agent_daemon_jsonl_lifecycle_response)) {
        return false;
    }
    if (response.event != "session_closed") {
        error = "unexpected daemon session_closed response: " + message.dump();
        return false;
    }
    return true;
}

bool agent_daemon_client_admin::shutdown(
        agent_daemon_jsonl_lifecycle_response & response,
        std::string & error) const {
    json message;
    if (!send_request_(make_agent_daemon_jsonl_shutdown_request({}), message, error)) {
        return false;
    }
    if (!parse_response_or_append_message(
                message,
                response,
                error,
                parse_agent_daemon_jsonl_lifecycle_response)) {
        return false;
    }
    if (response.event != "shutdown") {
        error = "unexpected daemon shutdown response: " + message.dump();
        return false;
    }
    return true;
}
