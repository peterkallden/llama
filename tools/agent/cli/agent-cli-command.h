#pragma once

#include "tools/agent/cli/agent-cli-options.h"
#include "memory/memory-store.h"

int run_agent_command_main(const char * argv0, int argc, char ** argv);
int run_memory_chat_command(const char * argv0, common_memory_store & store, args a);
