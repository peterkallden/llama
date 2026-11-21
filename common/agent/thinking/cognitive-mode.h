#pragma once

#include <string>

enum class common_agent_cognitive_mode { explore, frame, execute, reflect };

struct common_agent_cognitive_state {
    common_agent_cognitive_mode mode = common_agent_cognitive_mode::frame;
    std::string reason = "initial task framing";
};

const char * common_agent_cognitive_mode_name(common_agent_cognitive_mode mode);

// Modes are descriptive runtime state. This transition table does not grant
// route, planning, or tool authority; existing host policies remain decisive.
bool common_agent_cognitive_transition_allowed(
        common_agent_cognitive_mode from,
        common_agent_cognitive_mode to);

bool common_agent_cognitive_transition(
        common_agent_cognitive_state & state,
        common_agent_cognitive_mode to,
        std::string reason);

std::string common_agent_cognitive_mode_instruction(common_agent_cognitive_mode mode);
