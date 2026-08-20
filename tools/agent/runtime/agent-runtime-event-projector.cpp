#include "agent-runtime-event-projector.h"

common_agent_event_sink make_agent_runtime_event_projector(
        common_agent_daemon_event_sink daemon_sink,
        common_agent_event_context context) {
    return [daemon_sink = std::move(daemon_sink), context = std::move(context)](
            const common_agent_event & event) {
        std::string detail = common_agent_event_type_name(event.type);
        if (!event.detail.empty()) detail += ": " + event.detail;
        if (event.plan_id.has_value()) detail += " plan=" + *event.plan_id;
        if (!event.step_id.empty()) detail += " step=" + event.step_id;
        if (!event.tool_name.empty()) detail += " tool=" + event.tool_name;
        if (!event.observation_id.empty()) detail += " observation=" + event.observation_id;
        common_agent_event_emitter(daemon_sink, context).emit(
            common_agent_daemon_event_type::agent_runtime_event,
            std::move(detail));
    };
}
