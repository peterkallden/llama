#pragma once

#include "common.h"
#include "agent/agent-inference.h"
#include "agent/agent-prepared-generation.h"
#include "server-task.h"

task_params make_server_task_params_from_prepared_generation(
    const common_params & params_base,
    const common_agent_generation_request & request,
        const common_agent_prepared_generation & prepared,
        const std::vector<llama_logit_bias> & logit_bias_eog);

bool server_context_agent_generation_supports_flydelta(
        const common_agent_generation_request & request,
        std::string & error);

// An active FlyDelta overlay is request-scoped model state. Until the server
// graph can prove per-sequence overlay-aware KV reuse, prompt/KV reuse must be
// disabled for that request. Baseline requests remain eligible for the normal
// host cache policy.
bool server_context_agent_generation_requires_fresh_prompt_kv(
        const common_agent_generation_request & request);
