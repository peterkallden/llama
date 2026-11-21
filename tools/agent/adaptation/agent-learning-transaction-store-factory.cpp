#include "agent-learning-transaction-store.h"

#ifdef LLAMA_AGENT_ADAPTATION_USE_COZO
#include "agent-learning-transaction-store-cozo.h"
#endif
#ifdef LLAMA_AGENT_ADAPTATION_USE_SQLITE
#include "agent-learning-transaction-store-sqlite.h"
#endif

#include <memory>

std::unique_ptr<common_learning_transaction_store>
make_agent_learning_transaction_store(
        const std::string & requested_backend,
        const std::string & path,
        std::string & error) {
    error.clear();
    common_learning_transaction_backend backend;
    if (!parse_common_learning_transaction_backend(requested_backend.empty() ? "auto" : requested_backend, backend)) {
        error = "unknown adaptation transaction backend: " + requested_backend;
        return nullptr;
    }
    if (backend == common_learning_transaction_backend::automatic) {
        if (path.empty()) backend = common_learning_transaction_backend::in_memory;
#ifdef LLAMA_AGENT_ADAPTATION_USE_COZO
        else backend = common_learning_transaction_backend::cozo;
#elif defined(LLAMA_AGENT_ADAPTATION_USE_SQLITE)
        else backend = common_learning_transaction_backend::sqlite;
#else
        else {
            error = "no persistent adaptation transaction backend is compiled in";
            return nullptr;
        }
#endif
    }
    if (backend == common_learning_transaction_backend::in_memory) {
        if (!path.empty()) { error = "in-memory adaptation transaction backend does not accept a path"; return nullptr; }
        return std::make_unique<common_learning_in_memory_transaction_store>();
    }
    if (backend == common_learning_transaction_backend::jsonl) {
        auto store = std::make_unique<common_learning_jsonl_transaction_store>();
        if (!store->open(path, error)) return nullptr;
        return store;
    }
    if (path.empty()) {
        error = "persistent adaptation transaction backend requires a path";
        return nullptr;
    }
    if (backend == common_learning_transaction_backend::cozo) {
#ifdef LLAMA_AGENT_ADAPTATION_USE_COZO
        auto store = std::make_unique<common_agent_cozo_learning_transaction_store>();
        if (!store->open(path, error)) return nullptr;
        return store;
#else
        error = "adaptation transaction Cozo backend requires a build with LLAMA_MEMORY_COZO";
        return nullptr;
#endif
    }
    if (backend == common_learning_transaction_backend::sqlite) {
#ifdef LLAMA_AGENT_ADAPTATION_USE_SQLITE
        auto store = std::make_unique<common_agent_sqlite_learning_transaction_store>();
        if (!store->open(path, error)) return nullptr;
        return store;
#else
        error = "adaptation transaction SQLite backend requires a build with LLAMA_AGENT_STORAGE_SQLITE";
        return nullptr;
#endif
    }
    error = "unsupported adaptation transaction backend";
    return nullptr;
}
