#pragma once

#include "agent/data-store.h"

#include <memory>

std::unique_ptr<common_agent_data_store> make_agent_data_store(
    const common_agent_data_store_config & config,
    std::string & error);
