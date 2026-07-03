#include "agent-daemon-adapter.h"

#include <cstdio>
#include <cstring>

using json = nlohmann::ordered_json;

namespace {

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

std::string default_plan_scope_for_memory_scope(common_memory_scope memory_scope) {
    switch (memory_scope) {
        case common_memory_scope::turn:    return "turn";
        case common_memory_scope::session: return "session";
        case common_memory_scope::project: return "project";
        case common_memory_scope::global:  return "global";
    }
    return "session";
}

} // namespace

bool parse_agent_daemon_args(int argc, char ** argv, daemon_options & options) {
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

void print_agent_daemon_usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s --model MODEL [--default-mode chat|mini] [--planning-mode off|mini] [--reflection-mode off|always]\n"
        "         [--embedding-model MODEL] [--backend auto|in-memory|cozo] [--memory-db PATH]\n"
        "         [--plan-backend auto|in-memory|cozo] [--plan-db PATH] [--memory-learn off|post-turn] [--memory-learn-min-confidence F] [--memory-learn-min-reuse F]\n"
        "         [--memory-learn-show-candidate] [--agent-plan off|auto] [--agent-trace] [--plan-show-summary] [--n-predict N] [-ngl N]\n",
        argv0);
}

bool initialize_agent_daemon_environment(
        const daemon_options & options,
        common_agent_daemon_runtime & runtime,
        std::string & error) {
    if (!open_daemon_memory_store(options, runtime.memory_store, error)) {
        return false;
    }
    if (!open_daemon_plan_store(options, runtime.plan_store, error)) {
        return false;
    }
    if (!parse_mode(options.default_mode, runtime.default_mode)) {
        error = "unsupported default mode: " + options.default_mode;
        return false;
    }

    runtime.host = std::make_unique<common_agent_runtime_daemon_host>(
        make_agent_runtime_daemon_config(
            make_session_host_build_config(*runtime.memory_store, *runtime.plan_store, options)));
    error.clear();
    return true;
}

bool parse_agent_daemon_command(
        const json & parsed,
        const daemon_options & options,
        common_agent_runtime_host_mode default_mode,
        common_agent_daemon_command & command,
        std::string & error) {
    command = {};
    command.request_id = parsed.value("request_id", "");

    const std::string command_name = parsed.value("command", "");
    if (command_name == "shutdown") {
        command.type = common_agent_daemon_command_type::shutdown;
        error.clear();
        return true;
    }
    if (command_name == "status") {
        command.type = common_agent_daemon_command_type::get_status;
        error.clear();
        return true;
    }
    if (command_name == "reset_session" || command_name == "close_session") {
        command.type = command_name == "reset_session"
            ? common_agent_daemon_command_type::reset_session
            : common_agent_daemon_command_type::close_session;
        command.session = common_agent_runtime_session_key{
            parsed.value("namespace_id", "default-namespace"),
            parsed.value("session_id", "default-session"),
            parsed.value("project_id", ""),
        };
        error.clear();
        return true;
    }
    if (!command_name.empty() && command_name != "run_turn") {
        error = "unsupported command: " + command_name;
        return false;
    }

    command.type = common_agent_daemon_command_type::run_turn;
    command.turn.emplace();
    auto & request = *command.turn;
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
        error = "unsupported mode: " + mode_value;
        return false;
    }

    const std::string memory_scope_value = parsed.value("memory_scope", "session");
    if (!parse_memory_scope(memory_scope_value, request.memory_scope)) {
        error = "unsupported memory_scope: " + memory_scope_value;
        return false;
    }

    const std::string plan_scope_value = parsed.value(
        "plan_scope",
        default_plan_scope_for_memory_scope(request.memory_scope));
    if (!parse_plan_scope(plan_scope_value, request.plan_scope)) {
        error = "unsupported plan_scope: " + plan_scope_value;
        return false;
    }

    error.clear();
    return true;
}

json make_agent_daemon_ready_response(const daemon_options & options) {
    return {
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
    };
}

json make_agent_daemon_error_response(const std::string & error) {
    return {
        {"ok", false},
        {"error", error},
    };
}

json make_agent_daemon_command_response(const common_agent_daemon_command_result & result) {
    json response = {
        {"ok", result.ok},
    };
    if (!result.request_id.empty()) {
        response["request_id"] = result.request_id;
    }
    if (!result.event.empty()) {
        response["event"] = result.event;
    }

    if (result.event == "status") {
        response["state"] = result.state;
        response["live"] = result.live;
        response["ready"] = result.ready;
        response["sessions"] = result.session_count;
        json session_array = json::array();
        for (const auto & session : result.sessions) {
            session_array.push_back({
                {"namespace_id", session.namespace_id},
                {"session_id", session.session_id},
                {"project_id", session.project_id},
            });
        }
        response["session_keys"] = std::move(session_array);
        if (!result.error.empty()) {
            response["error"] = result.error;
        }
        return response;
    }

    if (!result.event.empty()) {
        if (!result.error.empty()) {
            response["error"] = result.error;
        }
        return response;
    }

    const auto & turn = result.turn_result;
    response["runtime_reused"] = turn.runtime_reused;
    response["limit_reached"] = turn.limit_reached;
    response["reflected"] = turn.reflected;
    response["revised"] = turn.revised;
    response["response"] = turn.response;
    response["total_decoded_tokens"] = turn.total_decoded_tokens;
    response["event_count"] = turn.event_count;
    response["memory_learning_related_count"] = turn.memory_learning_related_count;
    response["memory_learning_summary"] = turn.memory_learning_summary;
    if (!turn.plan_id.empty()) {
        response["plan_id"] = turn.plan_id;
    }
    if (!result.error.empty()) {
        response["error"] = result.error;
    }

    return response;
}
