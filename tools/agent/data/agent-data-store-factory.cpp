#include "agent-data-store-factory.h"

#ifdef LLAMA_MEMORY_USE_COZO
#include "agent-data-store-cozo.h"
#endif
#ifdef LLAMA_AGENT_DATA_USE_SQLITE
#include "agent-data-store-sqlite.h"
#endif

#include <memory>

std::unique_ptr<common_agent_data_store> make_agent_data_store(
        const common_agent_data_store_config & config,
        std::string & error) {
    std::string backend = config.backend;
    if (backend == "auto") {
        backend = config.path.empty() ? "disabled" : "cozo";
    }
    if (backend == "disabled" || backend == "none") {
        error.clear();
        return nullptr;
    }
    if (backend == "cozo") {
#ifdef LLAMA_MEMORY_USE_COZO
        if (config.path.empty()) {
            error = "data store cozo backend requires a path";
            return nullptr;
        }
        auto store = std::make_unique<common_agent_cozo_data_store>();
        if (!store->open(config.path, error)) return nullptr;
        return store;
#else
        error = "data store cozo backend requires a build with LLAMA_MEMORY_COZO";
        return nullptr;
#endif
    }
    if (backend == "sqlite") {
#ifdef LLAMA_AGENT_DATA_USE_SQLITE
        if (config.path.empty()) {
            error = "data store sqlite backend requires a path";
            return nullptr;
        }
        auto store = std::make_unique<common_agent_sqlite_data_store>();
        if (!store->open(config.path, error)) return nullptr;
        return store;
#else
        error = "data store sqlite backend requires a build with LLAMA_AGENT_STORAGE_SQLITE";
        return nullptr;
#endif
    }
    error = "unknown data store backend: " + backend;
    return nullptr;
}
