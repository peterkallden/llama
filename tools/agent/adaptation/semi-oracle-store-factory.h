#pragma once

#include "agent/adaptation/semi-oracle-store.h"

#include <memory>

std::unique_ptr<common_semi_oracle_store> make_agent_semi_oracle_store(
        const std::string & backend, const std::string & path, std::string & error);
