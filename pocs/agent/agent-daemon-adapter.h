#pragma once

#include "../common/cli-config.h"

#include "agent-daemon-service.h"
#include "agent-runtime-assembly.h"
#include "agent-runtime-execution.h"
#include "agent-plan-orchestration.h"

#include <nlohmann/json.hpp>

#include <memory>
#include <string>

struct daemon_options {
    std::string model;
    std::string embedding_model;
    std::string backend = "auto";
    std::string memory_db;
    std::string plan_backend = "auto";
    std::string plan_db;
    std::string default_mode = "chat";
    int n_predict = 64;
    int n_gpu_layers = 0;
    std::string planning_mode = "off";
    std::string reflection_mode = "off";
    std::string memory_learn = "off";
    std::string agent_plan = "off";
    bool memory_learn_show_candidate = false;
    float memory_learn_min_confidence = 0.75f;
    float memory_learn_min_reuse = 0.65f;
    bool plan_show_summary = false;
    bool agent_trace = false;
};

bool parse_mode(
    const std::string & value,
    common_agent_runtime_host_mode & mode);

bool parse_agent_daemon_args(int argc, char ** argv, daemon_options & options);
void print_agent_daemon_usage(const char * argv0);

bool initialize_agent_daemon_environment(
    const daemon_options & options,
    common_agent_daemon_runtime & runtime,
    std::string & error);

bool parse_agent_daemon_command(
    const nlohmann::ordered_json & parsed,
    const daemon_options & options,
    common_agent_runtime_host_mode default_mode,
    common_agent_daemon_command & command,
    std::string & error);

nlohmann::ordered_json make_agent_daemon_ready_response(const daemon_options & options);
nlohmann::ordered_json make_agent_daemon_error_response(const std::string & error);
nlohmann::ordered_json make_agent_daemon_command_response(
    const common_agent_daemon_command_result & result);
