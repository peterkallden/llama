#pragma once

#include "agent/context/conversation-context.h"
#include "resource/resource-contract.h"
#include "tools/agent/resource/agent-resource-store.h"

#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <vector>

class agent_session_context_store {
public:
    agent_session_context_store(agent_resource_store * store, std::string namespace_id,
            std::string session_id, bool persist);

    bool load(std::vector<common_agent_conversation_segment> & segments,
            std::optional<common_agent_context_checkpoint> & checkpoint,
            uint64_t & epoch, std::string & error) const;
    bool append(common_agent_conversation_segment segment, std::string & error);
    bool save_checkpoint(common_agent_context_checkpoint & checkpoint, std::string & error);
    bool begin_new_epoch(std::string & error);
    bool persistent() const { return store != nullptr && persist; }

private:
    bool list_internal(std::vector<agent_resource_descriptor> & resources, std::string & error) const;
    bool put_internal(const std::string & name, const std::string & key,
            const nlohmann::ordered_json & payload, agent_resource_descriptor & out,
            std::string & error) const;

    agent_resource_store * store = nullptr;
    std::string namespace_id;
    std::string session_id;
    bool persist = true;
    mutable std::vector<common_agent_conversation_segment> transient_segments;
    mutable std::optional<common_agent_context_checkpoint> transient_checkpoint;
    mutable uint64_t transient_epoch = 1;
    mutable uint64_t transient_checkpoint_revision = 0;
};
