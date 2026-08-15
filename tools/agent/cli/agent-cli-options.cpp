#include "agent-cli-options.h"

#include "memory/memory-in-memory.h"

#ifdef LLAMA_MEMORY_USE_COZO
#include "memory/cozo/memory-cozo.h"
#endif

#ifdef LLAMA_MEMORY_POC_USE_AGENT_TOOLS
#include "plan/plan-in-memory.h"
#ifdef LLAMA_PLAN_USE_COZO
#include "plan/cozo/plan-cozo.h"
#endif
#endif

bool resolve_agent_profile(args & a, std::string & error) {
    std::string tool_profile;
    std::string thinking_mode;
    std::string memory_learn;
    if (a.agent_profile == "default") {
        tool_profile = "memory"; thinking_mode = "reflective"; memory_learn = "off";
    } else if (a.agent_profile == "learning") {
        tool_profile = "memory"; thinking_mode = "reflective"; memory_learn = "post-turn";
    } else if (a.agent_profile == "research") {
        tool_profile = "research"; thinking_mode = "research"; memory_learn = "off";
    } else if (a.agent_profile == "safe") {
        tool_profile = "memory-read"; thinking_mode = "reflective"; memory_learn = "off";
    } else if (a.agent_profile == "static") {
        tool_profile.clear(); thinking_mode = "reflective"; memory_learn = "off";
    } else {
        error = "--agent-profile must be default, learning, research, safe, or static";
        return false;
    }
    if (!a.tool_profile_explicit) {
        a.tool_profile = std::move(tool_profile);
        if (a.thinking_mode_explicit && a.thinking_mode == "deliberate") {
            a.tool_profile = "analysis";
        }
    }
    a.agent_runtime = a.agent_profile != "static";
    if (!a.thinking_mode_explicit) a.thinking_mode = std::move(thinking_mode);
    if (!a.memory_learn_explicit) a.memory_learn = std::move(memory_learn);
    error.clear();
    return true;
}

std::unique_ptr<common_memory_store> make_memory_store(const args & a, std::string & error) {
    std::string backend = a.backend;
    if (backend == "auto") backend = a.memory_db.empty() ? "in-memory" : "cozo";
    if (backend == "in-memory" && !a.memory_db.empty()) {
        error = "--memory-db requires --backend cozo or the default auto backend";
        return nullptr;
    }
    if (backend == "in-memory") {
        error.clear();
        return std::make_unique<common_memory_in_memory_store>();
    }
    if (backend == "cozo") {
#ifdef LLAMA_MEMORY_USE_COZO
        if (a.memory_db.empty()) {
            error = "--backend cozo requires --memory-db PATH";
            return nullptr;
        }
        error.clear();
        return std::make_unique<common_memory_cozo_store>();
#else
        error = "this binary was built without LLAMA_MEMORY_COZO";
        return nullptr;
#endif
    }
    error = "unknown memory backend: " + backend;
    return nullptr;
}

bool open_memory_store(common_memory_store & store, const args & a, std::string & error) {
    return store.open(a.memory_db, error);
}

#ifdef LLAMA_MEMORY_POC_USE_AGENT_TOOLS
std::unique_ptr<common_plan_store> make_plan_store(const args & a, std::string & error) {
    std::string backend = a.plan_backend;
    if (backend == "auto") backend = a.plan_db.empty() ? "in-memory" : "cozo";
    if (backend == "in-memory" && !a.plan_db.empty()) {
        error = "--plan-db requires --plan-backend cozo or the default auto backend";
        return nullptr;
    }
    if (backend == "in-memory") {
        error.clear();
        return std::make_unique<common_plan_in_memory_store>();
    }
    if (backend == "cozo") {
#ifdef LLAMA_PLAN_USE_COZO
        if (a.plan_db.empty()) {
            error = "--plan-backend cozo requires --plan-db PATH";
            return nullptr;
        }
        error.clear();
        return std::make_unique<common_plan_cozo_store>();
#else
        error = "this binary was built without LLAMA_PLAN_COZO";
        return nullptr;
#endif
    }
    error = "unknown plan backend: " + backend;
    return nullptr;
}
#endif
