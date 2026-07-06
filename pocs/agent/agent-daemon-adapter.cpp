#include "agent-daemon-adapter.h"

#include "agent-cli-selection.h"
#include "agent-resource-store.h"

#include <cstdio>
#include <cstring>

using json = nlohmann::ordered_json;

namespace {

} // namespace

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
        } else if (std::strcmp(argv[i], "--tool-profile") == 0) {
            const char * value = need_value(argv[i]); if (!value) return false; options.tool_profile = value;
        } else if (std::strcmp(argv[i], "--repository-root") == 0) {
            const char * value = need_value(argv[i]); if (!value) return false; options.repository_root = value;
        } else if (std::strcmp(argv[i], "--mcp-tool-command") == 0) {
            const char * value = need_value(argv[i]); if (!value) return false; options.mcp_tool_command = value;
        } else if (std::strcmp(argv[i], "--mcp-tool-arg") == 0) {
            const char * value = need_value(argv[i]); if (!value) return false; options.mcp_tool_args.push_back(value);
        } else if (std::strcmp(argv[i], "--mcp-tool-server-name") == 0) {
            const char * value = need_value(argv[i]); if (!value) return false; options.mcp_tool_server_name = value;
        } else if (std::strcmp(argv[i], "--mcp-tool-prefix") == 0) {
            const char * value = need_value(argv[i]); if (!value) return false; options.mcp_tool_prefix = value;
        } else if (std::strcmp(argv[i], "--resource-blob-backend") == 0) {
            const char * value = need_value(argv[i]); if (!value) return false; options.resource_blob_backend = value;
        } else if (std::strcmp(argv[i], "--resource-blob-root") == 0) {
            const char * value = need_value(argv[i]); if (!value) return false; options.resource_blob_root = value;
        } else if (std::strcmp(argv[i], "--resource-metadata-backend") == 0) {
            const char * value = need_value(argv[i]); if (!value) return false; options.resource_metadata_backend = value;
        } else if (std::strcmp(argv[i], "--resource-metadata-db") == 0) {
            const char * value = need_value(argv[i]); if (!value) return false; options.resource_metadata_db = value;
        } else if (std::strcmp(argv[i], "--max-tool-rounds") == 0) {
            const char * value = need_value(argv[i]); if (!value) return false; options.max_tool_rounds = (size_t) std::stoul(value);
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
    if (options.max_tool_rounds > 4) {
        std::fprintf(stderr, "--max-tool-rounds must be between 0 and 4\n");
        return false;
    }
    std::string resource_error;
    if (!validate_agent_resource_store_config({
            options.resource_blob_backend,
            options.resource_blob_root,
            options.resource_metadata_backend,
            options.resource_metadata_db,
        }, resource_error)) {
        std::fprintf(stderr, "%s\n", resource_error.c_str());
        return false;
    }
    if (options.mcp_tool_command.empty() && !options.mcp_tool_args.empty()) {
        std::fprintf(stderr, "--mcp-tool-arg requires --mcp-tool-command\n");
        return false;
    }
    if (options.mcp_tool_server_name.empty()) {
        std::fprintf(stderr, "--mcp-tool-server-name must not be empty\n");
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
        "         [--resource-blob-backend auto|in-memory|fs|s3] [--resource-blob-root PATH]\n"
        "         [--resource-metadata-backend auto|in-memory|cozo] [--resource-metadata-db PATH]\n"
        "         [--memory-learn-show-candidate] [--agent-plan off|auto] [--agent-trace] [--plan-show-summary] [--max-tool-rounds N]\n"
        "         [--tool-profile ID] [--repository-root PATH] [--mcp-tool-command PATH] [--mcp-tool-arg VALUE ...]\n"
        "         [--mcp-tool-server-name NAME] [--mcp-tool-prefix PREFIX] [--n-predict N] [-ngl N]\n",
        argv0);
}
