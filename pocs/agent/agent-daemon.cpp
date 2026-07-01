#include "agent-runtime-host.h"
#include "agent-runtime-session-host.h"

#include "log.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>

using json = nlohmann::ordered_json;

namespace {

struct daemon_options {
    std::string model;
    std::string embedding_model;
    std::string backend = "auto";
    std::string memory_db;
    std::string plan_backend = "auto";
    std::string plan_db;
    std::string default_mode = "chat";
    int n_predict = 64;
    int n_gpu_layers = 0;
    std::string planning_mode = "off";
    std::string reflection_mode = "off";
    std::string memory_learn = "off";
    std::string agent_plan = "off";
    bool memory_learn_show_candidate = false;
    float memory_learn_min_confidence = 0.75f;
    float memory_learn_min_reuse = 0.65f;
    bool plan_show_summary = false;
    bool agent_trace = false;
};

bool parse_args(int argc, char ** argv, daemon_options & options) {
    for (int i = 1; i < argc; ++i) {
        auto need_value = [&](const char * name) -> const char * {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", name);
                return nullptr;
            }
            return argv[++i];
        };

        if (std::strcmp(argv[i], "--model") == 0) {
            const char * value = need_value(argv[i]); if (!value) return false; options.model = value;
        } else if (std::strcmp(argv[i], "--embedding-model") == 0) {
            const char * value = need_value(argv[i]); if (!value) return false; options.embedding_model = value;
        } else if (std::strcmp(argv[i], "--backend") == 0) {
            const char * value = need_value(argv[i]); if (!value) return false; options.backend = value;
        } else if (std::strcmp(argv[i], "--memory-db") == 0) {
            const char * value = need_value(argv[i]); if (!value) return false; options.memory_db = value;
        } else if (std::strcmp(argv[i], "--plan-backend") == 0) {
            const char * value = need_value(argv[i]); if (!value) return false; options.plan_backend = value;
        } else if (std::strcmp(argv[i], "--plan-db") == 0) {
            const char * value = need_value(argv[i]); if (!value) return false; options.plan_db = value;
        } else if (std::strcmp(argv[i], "--default-mode") == 0) {
            const char * value = need_value(argv[i]); if (!value) return false; options.default_mode = value;
        } else if (std::strcmp(argv[i], "-n") == 0 || std::strcmp(argv[i], "--n-predict") == 0) {
            const char * value = need_value(argv[i]); if (!value) return false; options.n_predict = std::stoi(value);
        } else if (std::strcmp(argv[i], "-ngl") == 0 || std::strcmp(argv[i], "--n-gpu-layers") == 0) {
            const char * value = need_value(argv[i]); if (!value) return false; options.n_gpu_layers = std::stoi(value);
        } else if (std::strcmp(argv[i], "--planning-mode") == 0) {
            const char * value = need_value(argv[i]); if (!value) return false; options.planning_mode = value;
        } else if (std::strcmp(argv[i], "--reflection-mode") == 0) {
            const char * value = need_value(argv[i]); if (!value) return false; options.reflection_mode = value;
        } else if (std::strcmp(argv[i], "--memory-learn") == 0) {
            const char * value = need_value(argv[i]); if (!value) return false; options.memory_learn = value;
        } else if (std::strcmp(argv[i], "--memory-learn-show-candidate") == 0) {
            options.memory_learn_show_candidate = true;
        } else if (std::strcmp(argv[i], "--memory-learn-min-confidence") == 0) {
            const char * value = need_value(argv[i]); if (!value) return false; options.memory_learn_min_confidence = std::stof(value);
        } else if (std::strcmp(argv[i], "--memory-learn-min-reuse") == 0) {
            const char * value = need_value(argv[i]); if (!value) return false; options.memory_learn_min_reuse = std::stof(value);
        } else if (std::strcmp(argv[i], "--agent-plan") == 0) {
            const char * value = need_value(argv[i]); if (!value) return false; options.agent_plan = value;
        } else if (std::strcmp(argv[i], "--plan-show-summary") == 0) {
            options.plan_show_summary = true;
        } else if (std::strcmp(argv[i], "--agent-trace") == 0) {
            options.agent_trace = true;
        } else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            return false;
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", argv[i]);
            return false;
        }
    }

    if (options.model.empty()) {
        std::fprintf(stderr, "--model is required\n");
        return false;
    }

    if (options.default_mode != "chat" && options.default_mode != "mini") {
        std::fprintf(stderr, "--default-mode must be chat or mini\n");
        return false;
    }
    if (options.backend != "auto" && options.backend != "in-memory" && options.backend != "cozo") {
        std::fprintf(stderr, "--backend must be auto, in-memory, or cozo\n");
        return false;
    }
    if (options.plan_backend != "auto" && options.plan_backend != "in-memory" && options.plan_backend != "cozo") {
        std::fprintf(stderr, "--plan-backend must be auto, in-memory, or cozo\n");
        return false;
    }
    if (options.planning_mode != "off" && options.planning_mode != "mini") {
        std::fprintf(stderr, "--planning-mode must be off or mini\n");
        return false;
    }
    if (options.reflection_mode != "off" && options.reflection_mode != "always") {
        std::fprintf(stderr, "--reflection-mode must be off or always\n");
        return false;
    }
    if (options.memory_learn != "off" && options.memory_learn != "post-turn") {
        std::fprintf(stderr, "--memory-learn must be off or post-turn\n");
        return false;
    }
    if (options.agent_plan != "off" && options.agent_plan != "auto") {
        std::fprintf(stderr, "--agent-plan must be off or auto\n");
        return false;
    }
    if (options.memory_learn_min_confidence < 0.0f || options.memory_learn_min_confidence > 1.0f ||
            options.memory_learn_min_reuse < 0.0f || options.memory_learn_min_reuse > 1.0f) {
        std::fprintf(stderr, "memory learning thresholds must be between 0 and 1\n");
        return false;
    }

    return true;
}

void usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s --model MODEL [--default-mode chat|mini] [--planning-mode off|mini] [--reflection-mode off|always]\n"
        "         [--embedding-model MODEL] [--backend auto|in-memory|cozo] [--memory-db PATH]\n"
        "         [--plan-backend auto|in-memory|cozo] [--plan-db PATH] [--memory-learn off|post-turn] [--memory-learn-min-confidence F] [--memory-learn-min-reuse F]\n"
        "         [--memory-learn-show-candidate] [--agent-plan off|auto] [--agent-trace] [--plan-show-summary] [--n-predict N] [-ngl N]\n",
        argv0);
}

bool parse_mode(
        const std::string & value,
        common_agent_runtime_host_mode & mode) {
    if (value == "chat") {
        mode = common_agent_runtime_host_mode::chat;
        return true;
    }
    if (value == "mini") {
        mode = common_agent_runtime_host_mode::mini;
        return true;
    }
    return false;
}

bool parse_memory_scope(
        const std::string & value,
        common_memory_scope & scope) {
    return common_memory_scope_parse(value, scope);
}

bool parse_plan_scope(
        const std::string & value,
        common_plan_scope & scope) {
    if (value == "turn")    { scope = common_plan_scope::turn; return true; }
    if (value == "session") { scope = common_plan_scope::session; return true; }
    if (value == "project") { scope = common_plan_scope::project; return true; }
    if (value == "global")  { scope = common_plan_scope::global; return true; }
    return false;
}

json make_error_response(const std::string & error) {
    return {
        {"ok", false},
        {"error", error},
    };
}

json make_turn_response(const common_agent_runtime_daemon_turn_result & result) {
    json response = {
        {"ok", result.ok},
        {"runtime_reused", result.runtime_reused},
        {"limit_reached", result.limit_reached},
        {"reflected", result.reflected},
        {"revised", result.revised},
        {"response", result.response},
        {"total_decoded_tokens", result.total_decoded_tokens},
        {"event_count", result.event_count},
        {"memory_learning_related_count", result.memory_learning_related_count},
        {"memory_learning_summary", result.memory_learning_summary},
    };
    if (!result.plan_id.empty()) {
        response["plan_id"] = result.plan_id;
    }
    if (!result.error.empty()) {
        response["error"] = result.error;
    }
    return response;
}

args make_runtime_args(const daemon_options & options) {
    args runtime_args;
    runtime_args.prompt = "";
    runtime_args.model = options.model;
    runtime_args.embedding_model = options.embedding_model;
    runtime_args.backend = options.backend;
    runtime_args.memory_db = options.memory_db;
    runtime_args.plan_backend = options.plan_backend;
    runtime_args.plan_db = options.plan_db;
    runtime_args.n_predict = options.n_predict;
    runtime_args.n_gpu_layers = options.n_gpu_layers;
    runtime_args.planning_mode = options.planning_mode;
    runtime_args.agent_plan = options.agent_plan;
    runtime_args.agent_blueprint = "off";
    runtime_args.reflection_mode = options.reflection_mode;
    runtime_args.memory_learn = options.memory_learn;
    runtime_args.memory_learn_show_candidate = options.memory_learn_show_candidate;
    runtime_args.memory_learn_min_confidence = options.memory_learn_min_confidence;
    runtime_args.memory_learn_min_reuse = options.memory_learn_min_reuse;
    runtime_args.plan_show_summary = options.plan_show_summary;
    runtime_args.agent_trace = options.agent_trace;
    return runtime_args;
}

common_agent_runtime_resident_request_config make_resident_request_config(
        const daemon_options & options) {
    return {
        "",
        "",
        "",
        "",
        options.model,
        options.n_predict,
        options.n_gpu_layers,
        false,
        "server-context",
        common_memory_scope::session,
        common_plan_scope::turn,
    };
}

common_agent_runtime_session_host_build_config make_session_host_build_config(
        common_memory_store & memory_store,
        common_plan_store & plan_store,
        const daemon_options & options) {
    auto runtime_args = make_runtime_args(options);
    return {
        memory_store,
        plan_store,
        make_resident_request_config(options),
        make_agent_runtime_policy(runtime_args),
        make_agent_runtime_config(runtime_args),
        make_agent_orchestration_config(runtime_args),
        common_memory_scope::session,
        true,
        {},
        {},
        false,
        nullptr,
    };
}

bool open_daemon_memory_store(
        const daemon_options & options,
        std::unique_ptr<common_memory_store> & store,
        std::string & error) {
    const auto store_args = make_runtime_args(options);
    store = make_memory_store(store_args, error);
    if (!store) {
        return false;
    }
    return open_memory_store(*store, store_args, error);
}

bool open_daemon_plan_store(
        const daemon_options & options,
        std::unique_ptr<common_plan_store> & store,
        std::string & error) {
    const auto store_args = make_runtime_args(options);
    store = make_plan_store(store_args, error);
    if (!store) {
        return false;
    }
    return store->open(store_args.plan_db, error);
}

} // namespace

int main(int argc, char ** argv) {
    daemon_options options;
    if (!parse_args(argc, argv, options)) {
        usage(argv[0]);
        return 2;
    }
    common_log_set_verbosity_thold(LOG_LEVEL_WARN);

    std::unique_ptr<common_memory_store> memory_store;
    std::unique_ptr<common_plan_store> plan_store;
    std::string error;
    if (!open_daemon_memory_store(options, memory_store, error)) {
        std::fprintf(stderr, "failed to open daemon memory store: %s\n", error.c_str());
        return 1;
    }
    if (!open_daemon_plan_store(options, plan_store, error)) {
        std::fprintf(stderr, "failed to open daemon plan store: %s\n", error.c_str());
        return 1;
    }

    common_agent_runtime_host_mode default_mode = common_agent_runtime_host_mode::chat;
    if (!parse_mode(options.default_mode, default_mode)) {
        std::fprintf(stderr, "unsupported default mode: %s\n", options.default_mode.c_str());
        return 2;
    }

    common_agent_runtime_daemon_host daemon(make_agent_runtime_daemon_config(
        make_session_host_build_config(*memory_store, *plan_store, options)));

    std::cout << json({
        {"ok", true},
        {"event", "ready"},
        {"default_mode", options.default_mode},
        {"protocol_version", 1},
        {"capabilities", json::array({
            "chat",
            "mini",
            "planning",
            "reflection",
            "memory_learning",
            "scoped_sessions",
        })},
    }).dump() << std::endl;

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) {
            continue;
        }

        const auto parsed = json::parse(line, nullptr, false);
        if (parsed.is_discarded() || !parsed.is_object()) {
            std::cout << make_error_response("invalid JSON request").dump() << std::endl;
            continue;
        }

        if (parsed.value("command", "") == "shutdown") {
            std::cout << json({
                {"ok", true},
                {"event", "shutdown"},
            }).dump() << std::endl;
            break;
        }

        common_agent_runtime_daemon_turn_request request;
        request.prompt = parsed.value("prompt", "");
        request.session_id = parsed.value("session_id", "default-session");
        request.namespace_id = parsed.value("namespace_id", "default-namespace");
        request.project_id = parsed.value("project_id", "");
        request.turn_id = parsed.value("turn_id", "");
        request.n_predict = parsed.value("n_predict", 0);
        request.mode = default_mode;
        request.memory_scope = common_memory_scope::session;
        request.plan_scope = common_plan_scope::turn;

        const std::string mode_value = parsed.value("mode", options.default_mode);
        if (!parse_mode(mode_value, request.mode)) {
            std::cout << make_error_response("unsupported mode: " + mode_value).dump() << std::endl;
            continue;
        }

        const std::string memory_scope_value = parsed.value("memory_scope", "session");
        if (!parse_memory_scope(memory_scope_value, request.memory_scope)) {
            std::cout << make_error_response("unsupported memory_scope: " + memory_scope_value).dump() << std::endl;
            continue;
        }

        const std::string plan_scope_value = parsed.value("plan_scope", request.memory_scope == common_memory_scope::turn    ? "turn" :
                                                                         request.memory_scope == common_memory_scope::session ? "session" :
                                                                         request.memory_scope == common_memory_scope::project ? "project" : "global");
        if (!parse_plan_scope(plan_scope_value, request.plan_scope)) {
            std::cout << make_error_response("unsupported plan_scope: " + plan_scope_value).dump() << std::endl;
            continue;
        }

        common_agent_runtime_daemon_turn_result result;
        error.clear();
        daemon.run_turn(request, result, error);
        if (!error.empty() && result.error.empty()) {
            result.error = error;
        }
        std::cout << make_turn_response(result).dump() << std::endl;
    }

    return 0;
}
