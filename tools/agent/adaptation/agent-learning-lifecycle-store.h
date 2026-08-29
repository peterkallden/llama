#pragma once

#include "agent/adaptation/lifecycle-store.h"

#include <memory>
#include <string>

// Uses the same backend selection policy as the transaction ledger, but keeps
// lifecycle events in their own relation/table.
std::unique_ptr<common_learning_lifecycle_store>
make_agent_learning_lifecycle_store(
        const std::string & backend,
        const std::string & path,
        std::string & error);
