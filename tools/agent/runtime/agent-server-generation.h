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
