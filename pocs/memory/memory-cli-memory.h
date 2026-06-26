#pragma once

#include "chat.h"
#include "common/cli-config.h"
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
