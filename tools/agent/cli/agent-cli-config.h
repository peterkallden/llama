#pragma once

#include "tools/agent/cli/agent-cli-options.h"

#include <string>

void print_agent_usage(const char * argv0, const char * command_name = "run");
bool parse_agent_run_args(int argc, char ** argv, args & out);
bool validate_agent_memory_scope(const args & a, std::string & error);
