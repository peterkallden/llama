#pragma once
#include "agent/contracts/agent-request.h"

// Advance only along this plan's verified publications. A live store read
// validates that baseline; it never silently rebases onto another writer.
bool common_agent_refresh_investigation_context(common_agent_request &,
    const common_plan_state *, std::string & error);
std::string common_agent_render_investigation_context(const common_agent_request &);

// Project configured, scope-authorized connector identities into tagged model
// fields. The execution adapter always rechecks live source authorization.
bool common_agent_project_investigation_source_choices(std::vector<common_chat_tool> &,
    const std::vector<std::string> & source_ids, std::string & error);

// Bind only the selected, verified case baseline and schema-declared fields.
// Call refresh first; this helper never reads or chooses a new case revision.
bool common_agent_bind_investigation_tool(common_agent_request &, const std::string & tool_name,
    const std::string & input_schema, std::string & error);
