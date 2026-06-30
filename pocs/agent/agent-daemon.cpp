#include "agent-runtime-host.h"

#include "log.h"
#include "memory/memory-in-memory.h"
#include "plan/plan-in-memory.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>

using json = nlohmann::ordered_json;

namespace {

struct daemon_options {
    std::string model;
    std::string default_mode = "chat";
    int n_predict = 64;
    int n_gpu_layers = 0;
    std::string planning_mode = "off";
    std::string reflection_mode = "off";
    std::string memory_learn = "off";
    std::string agent_plan = "off";
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
        } else if (std::strcmp(argv[i], "--agent-plan") == 0) {
            const char * value = need_value(argv[i]); if (!value) return false; options.agent_plan = value;
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

    return true;
}

void usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s --model MODEL [--default-mode chat|mini] [--planning-mode off|mini] [--reflection-mode off|always]\n"
        "         [--memory-learn off|post-turn] [--agent-plan off|auto] [--n-predict N] [-ngl N]\n",
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
        {"response", result.response},
        {"total_decoded_tokens", result.total_decoded_tokens},
    };
    if (!result.plan_id.empty()) {
        response["plan_id"] = result.plan_id;
    }
    if (!result.error.empty()) {
        response["error"] = result.error;
    }
    return response;
}

} // namespace

int main(int argc, char ** argv) {
    daemon_options options;
    if (!parse_args(argc, argv, options)) {
        usage(argv[0]);
        return 2;
    }
    common_log_set_verbosity_thold(LOG_LEVEL_WARN);

    common_memory_in_memory_store memory_store;
    common_plan_in_memory_store plan_store;
    std::string error;
    if (!memory_store.open("", error)) {
        std::fprintf(stderr, "failed to open in-memory memory store: %s\n", error.c_str());
        return 1;
    }
    if (!plan_store.open("", error)) {
        std::fprintf(stderr, "failed to open in-memory plan store: %s\n", error.c_str());
        return 1;
    }

    args runtime_args;
    runtime_args.prompt = "";
    runtime_args.planning_mode = options.planning_mode;
    runtime_args.agent_plan = options.agent_plan;
    runtime_args.agent_blueprint = "off";
    runtime_args.reflection_mode = options.reflection_mode;
    runtime_args.memory_learn = options.memory_learn;

    common_agent_runtime_host_mode default_mode = common_agent_runtime_host_mode::chat;
    if (!parse_mode(options.default_mode, default_mode)) {
        std::fprintf(stderr, "unsupported default mode: %s\n", options.default_mode.c_str());
        return 2;
    }

    common_agent_runtime_daemon_host daemon({
        memory_store,
        plan_store,
        {
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
        },
        make_agent_runtime_policy(runtime_args),
        make_agent_runtime_config(runtime_args),
        make_agent_orchestration_config(runtime_args),
        common_memory_scope::session,
        true,
        {},
        {},
        false,
        nullptr,
    });

    std::cout << json({
        {"ok", true},
        {"event", "ready"},
        {"default_mode", options.default_mode},
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
