#pragma once

#include "agent/adaptation/learning-transaction.h"

#include <memory>
#include <string>

// Creates the host-side persistence adapter for the adaptation ledger.
// `auto` means in-memory without a path, then Cozo, then SQLite for a
// persistent path. JSONL is explicit and intended for portable export/debug.
std::unique_ptr<common_learning_transaction_store>
make_agent_learning_transaction_store(
        const std::string & backend,
        const std::string & path,
        std::string & error);
