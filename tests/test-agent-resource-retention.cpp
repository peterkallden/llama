#include "resource/resource-retention.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <map>

class retention_store final : public agent_resource_store {
public:
    bool put_text(const agent_resource_put_request & request, agent_resource_descriptor & out,
            std::string & error) override {
        auto bytes = request;
        bytes.bytes = request.text;
        return put_bytes(bytes, out, error);
    }
    bool put_bytes(const agent_resource_put_request & request, agent_resource_descriptor & out,
            std::string & error) override {
        out.uri = "resource://" + std::to_string(++next_id);
        out.name = request.name;
        out.description = request.description;
        out.mime_type = request.mime_type;
        out.size_bytes = request.bytes.size();
        out.scope = request.scope;
        out.namespace_id = request.namespace_id;
        out.session_id = request.session_id;
        out.project_id = request.project_id;
        out.turn_id = request.turn_id;
        out.sha256 = "resource-sha";
        out.metadata = request.metadata;
        out.lineage = request.lineage;
        resources[out.uri] = out;
        payloads[out.uri] = request.bytes;
        error.clear();
        return true;
    }
    bool read_bytes(const std::string & uri, const agent_resource_read_authority & authority,
            size_t max_bytes, std::string & out, std::string & error) const override {
        if (!allowed(uri, authority) || payloads.at(uri).size() > max_bytes) {
            error = "resource authority denied";
            return false;
        }
        out = payloads.at(uri);
        error.clear();
        return true;
    }
    bool read_bytes_range(const std::string &, const agent_resource_read_authority &, size_t,
            size_t, std::string &, std::string & error) const override {
        error = "not used";
        return false;
    }
    bool read_text(const std::string & uri, const agent_resource_read_authority & authority,
            size_t max_bytes, std::string & out, std::string & error) const override {
        return read_bytes(uri, authority, max_bytes, out, error);
    }
    bool read_text_range(const std::string &, const agent_resource_read_authority &, size_t,
            size_t, std::string &, std::string & error) const override {
        error = "not used";
        return false;
    }
    bool stat(const std::string & uri, const agent_resource_read_authority & authority,
            agent_resource_descriptor & out, std::string & error) const override {
        if (!allowed(uri, authority)) {
            error = "resource authority denied";
            return false;
        }
        out = resources.at(uri);
        error.clear();
        return true;
    }
    bool list(const agent_resource_read_authority &, std::vector<agent_resource_descriptor> &,
            std::string & error) const override {
        error = "not used";
        return false;
    }
    std::map<std::string, agent_resource_descriptor> resources;
    std::map<std::string, std::string> payloads;
private:
    bool allowed(const std::string & uri, const agent_resource_read_authority & authority) const {
        const auto found = resources.find(uri);
        if (found == resources.end() || found->second.namespace_id != authority.namespace_id) return false;
        const auto & resource = found->second;
        if (resource.scope == common_runtime_resource_scope::project)
            return resource.project_id == authority.project_id;
        if (resource.scope == common_runtime_resource_scope::session)
            return resource.session_id == authority.session_id;
        return resource.session_id == authority.session_id && resource.turn_id == authority.turn_id;
    }
    size_t next_id = 0;
};

int main() {
    retention_store store;
    agent_resource_put_request original_request;
    original_request.name = "notes.txt";
    original_request.mime_type = "text/plain";
    original_request.bytes = "retained evidence";
    original_request.bytes_are_authoritative = true;
    original_request.scope = common_runtime_resource_scope::turn;
    original_request.namespace_id = "tenant";
    original_request.session_id = "chat-a";
    original_request.project_id = "project-a";
    original_request.turn_id = "turn-1";
    agent_resource_descriptor original;
    std::string error;
    assert(store.put_bytes(original_request, original, error));

    agent_resource_read_authority authority;
    authority.namespace_id = "tenant";
    authority.session_id = "chat-a";
    authority.project_id = "project-a";
    authority.turn_id = "turn-1";
    agent_resource_retention_target chat_target;
    chat_target.scope = common_runtime_resource_scope::session;
    chat_target.namespace_id = "tenant";
    chat_target.session_id = "chat-a";
    chat_target.project_id = "project-a";
    chat_target.source_tool = "research.workspace.persist";
    chat_target.max_bytes = 1024;
    agent_resource_descriptor chat_resource;
    bool copied = false;
    assert(retain_agent_resource(store, original, authority, chat_target, chat_resource, copied, error));
    assert(copied && chat_resource.scope == common_runtime_resource_scope::session &&
        chat_resource.lineage.parent_uri == original.uri && store.payloads.at(chat_resource.uri) == "retained evidence");

    agent_resource_retention_target project_target = chat_target;
    project_target.scope = common_runtime_resource_scope::project;
    project_target.project_id = "project-a";
    project_target.source_tool = "research.workspace.publish";
    agent_resource_descriptor project_resource;
    assert(retain_agent_resource(store, chat_resource, authority, project_target, project_resource, copied, error));
    assert(copied && project_resource.scope == common_runtime_resource_scope::project &&
        project_resource.lineage.parent_uri == chat_resource.uri);

    assert(retain_agent_resource(store, project_resource, authority, chat_target, chat_resource, copied, error));
    assert(!copied && chat_resource.uri == project_resource.uri);

    auto too_small = chat_target;
    too_small.max_bytes = 2;
    agent_resource_descriptor rejected;
    assert(!retain_agent_resource(store, original, authority, too_small, rejected, copied, error));
    assert(error.find("byte limit") != std::string::npos);
    authority.session_id = "chat-b";
    assert(!retain_agent_resource(store, original, authority, chat_target, rejected, copied, error));
    return 0;
}
