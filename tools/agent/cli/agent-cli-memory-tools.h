#pragma once

#include "chat.h"
#include "tools/agent/cli/agent-cli-options.h"
#include "memory/memory-store.h"

#include <string>
#include <vector>

void apply_memory_scope(const args & a, common_memory_record & record);
void apply_memory_scope(const args & a, common_memory_query & query);

bool ensure_memory_cli_embedding(
    const args & a,
    const std::string & text,
    std::vector<float> & embedding,
    const char * label,
    std::string & error);

bool ensure_memory_cli_embedding_from_model(
    const std::string & model_path,
    int n_gpu_layers,
    const std::string & text,
    std::vector<float> & embedding,
    const char * label,
    std::string & error);

common_chat_tool memory_search_tool_definition();
common_chat_tool memory_remember_tool_definition();

std::string memory_search_tool_result(
    common_memory_store & store,
    const args & a,
    const std::string & arguments);

std::string memory_remember_tool_result(
    common_memory_store & store,
    const args & a,
    const std::string & arguments);
