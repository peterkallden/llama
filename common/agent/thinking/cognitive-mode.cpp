#include "cognitive-mode.h"

#include <utility>

const char * common_agent_cognitive_mode_name(common_agent_cognitive_mode mode) {
    switch (mode) {
        case common_agent_cognitive_mode::explore: return "explore";
        case common_agent_cognitive_mode::frame: return "frame";
        case common_agent_cognitive_mode::execute: return "execute";
        case common_agent_cognitive_mode::reflect: return "reflect";
    }
    return "frame";
}

bool common_agent_cognitive_transition_allowed(
        common_agent_cognitive_mode from,
        common_agent_cognitive_mode to) {
    switch (from) {
        case common_agent_cognitive_mode::explore:
            return to == common_agent_cognitive_mode::frame;
        case common_agent_cognitive_mode::frame:
            return to == common_agent_cognitive_mode::execute ||
                to == common_agent_cognitive_mode::explore;
        case common_agent_cognitive_mode::execute:
            return to == common_agent_cognitive_mode::reflect;
        case common_agent_cognitive_mode::reflect:
            return to == common_agent_cognitive_mode::execute ||
                to == common_agent_cognitive_mode::frame ||
                to == common_agent_cognitive_mode::explore;
    }
    return false;
}

bool common_agent_cognitive_transition(
        common_agent_cognitive_state & state,
        common_agent_cognitive_mode to,
        std::string reason) {
    if (!common_agent_cognitive_transition_allowed(state.mode, to)) return false;
    state.mode = to;
    state.reason = std::move(reason);
    return true;
}

std::string common_agent_cognitive_mode_instruction(common_agent_cognitive_mode mode) {
    switch (mode) {
        case common_agent_cognitive_mode::explore:
            return "Host-assigned cognitive mode: explore. Gather or compare relevant evidence; do not silently change the selected route or claim that evidence is sufficient.";
        case common_agent_cognitive_mode::frame:
            return "Host-assigned cognitive mode: frame. Clarify task structure and plan within the selected route; any route change requires explicit host authorization.";
        case common_agent_cognitive_mode::execute:
            return "Host-assigned cognitive mode: execute. Advance the validated plan within its current route and use only the tools already exposed by the host.";
        case common_agent_cognitive_mode::reflect:
            return "Host-assigned cognitive mode: reflect. Assess the result against the request and verified evidence; request only supported repair, replan, or research actions.";
    }
    return "Host-assigned cognitive mode: frame.";
}
