#pragma once

#include "agent/agent-continuation.h"
#include "agent/agent-generation.h"
#include "agent/contracts/agent-events.h"
#include "agent/contracts/agent-failures.h"
#include "agent/contracts/agent-learning.h"
#include "agent/thinking/research/research-contract.h"
#include "agent/thinking/research/research-verifier.h"
#include "memory/memory-candidate.h"
#include "runtime/runtime-trace.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

struct common_agent_result {
    std::string response;
    std::string error;
    int total_decoded_tokens = 0;
    int response_decoded_tokens = 0;
    int reasoning_decoded_tokens = 0;
    common_agent_generation_status response_generation_status = common_agent_generation_status::errored;
    common_agent_generation_stop_reason response_stop_reason = common_agent_generation_stop_reason::error;
    std::vector<std::string> memory_ids;
    std::optional<common_memory_candidate> learned_memory_candidate;
    std::string memory_learning_summary;
    size_t memory_learning_related_count = 0;
    std::vector<common_learning_signal> learning_signals;
    std::vector<common_agent_failure> failures;
    std::optional<std::string> plan_id;
    uint64_t plan_version = 0;
    bool reflected = false;
    bool revised = false;
    bool limit_reached = false;
    std::optional<common_agent_continuation_checkpoint> continuation_checkpoint;
    std::vector<common_agent_generation_record> generation_records;
    std::optional<common_agent_research_result> research_result;
    std::optional<common_agent_research_workspace_checkpoint> research_workspace_checkpoint;
    std::optional<common_agent_research_verification> research_verification;
    std::vector<common_agent_event> events;
    std::vector<common_runtime_trace_entry> trace;
};
