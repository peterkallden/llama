#include "agent-cli-config.h"

#include "memory/memory-context.h"

#include <cstdio>
#include <cstring>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

void print_agent_usage(const char * argv0, const char * command_name) {
    fprintf(stderr,
        "usage:\n"
        "  %s %s --model MODEL --prompt TEXT [--backend auto|in-memory|cozo] [--memory-db PATH] [--config PATH] [--embedding-model MODEL] [--agent-profile default|learning|research|safe|static]\n"
        "         [--tool-profile NAME] [--thinking-mode auto|reflective|deliberate|research]\n"
        "         [--max-reflection-rounds N] [--max-plan-revisions N] [--max-research-iterations N]\n"
        "         [--max-tool-rounds N] [--n-predict N] [--context-size N] [--threads N] [-ngl N]\n"
        "         [--inference-step-timeout-ms N] [--generation-trace]\n"
        "         [--agent-inference-backend cli|server-context] [--mmproj PATH]\n"
        "         [--mcp-tool-command PATH] [--mcp-tool-arg VALUE ...] [--mcp-tool-server-name NAME] [--mcp-tool-prefix PREFIX]\n"
        "         [--resource-blob-backend auto|in-memory|fs|s3] [--resource-blob-root PATH]\n"
        "         [--resource-metadata-backend auto|in-memory|cozo] [--resource-metadata-db PATH]\n"
        "         [--resource PATH ...] [--resource-mime-type MIME]\n"
        "         [--memory-scope turn|session|project|global] [--memory-namespace ID] [--memory-session ID] [--memory-project ID] [--memory-turn ID]\n"
        "         [--plan-backend in-memory|cozo] [--plan-db PATH] [--plan-id ID] [--agent-plan off|auto]\n"
        "         [--agent-bootstrap none|default|--agent-import PATH|--agent-export PATH] [--agent-blueprint ID] [--repository-root PATH]\n"
        "         [--agent-trace] [--plan-show-summary] [--include-summary]\n"
        "  %s daemon-chat --model MODEL --prompt TEXT [--embedding-model MODEL] [--thinking-mode auto|reflective|deliberate|research]\n"
        "         [--memory-learn off|post-turn] [--memory-learn-min-confidence F] [--memory-learn-min-reuse F] [--memory-learn-show-candidate]\n"
        "         [--resource-blob-backend auto|in-memory|fs|s3] [--resource-blob-root PATH]\n"
        "         [--resource-metadata-backend auto|in-memory|cozo] [--resource-metadata-db PATH]\n"
        "         [--agent-plan off|auto] [--agent-trace] [--plan-show-summary] [--include-summary] [--memory-scope turn|session|project|global]\n"
        "         [--memory-namespace ID] [--memory-session ID] [--memory-project ID] [--memory-turn ID] [--plan-scope turn|session|project|global]\n"
        "         [--n-predict N] [--context-size N] [--threads N] [-ngl N] [--agent-inference-backend server-context]\n"
        "  %s daemon-session --model MODEL [--prompt TEXT] [--embedding-model MODEL] [--thinking-mode auto|reflective|deliberate|research]\n"
        "         [--memory-learn off|post-turn] [--memory-learn-min-confidence F] [--memory-learn-min-reuse F] [--memory-learn-show-candidate]\n"
        "         [--resource-blob-backend auto|in-memory|fs|s3] [--resource-blob-root PATH]\n"
        "         [--resource-metadata-backend auto|in-memory|cozo] [--resource-metadata-db PATH]\n"
        "         [--agent-plan off|auto] [--agent-trace] [--plan-show-summary] [--include-summary] [--memory-scope turn|session|project|global]\n"
        "         [--memory-namespace ID] [--memory-session ID] [--memory-project ID] [--memory-turn ID] [--plan-scope turn|session|project|global]\n"
        "         [--n-predict N] [--context-size N] [-ngl N] [--agent-inference-backend server-context]\n",
        argv0, command_name, argv0, argv0);
}

static bool parse_embedding(const std::string & value, std::vector<float> & out) {
    out.clear();
    std::stringstream ss(value);
    std::string item;
    while (std::getline(ss, item, ',')) {
        try {
            out.push_back(std::stof(item));
        } catch (...) {
            return false;
        }
    }
    return true;
}

bool parse_agent_run_args(int argc, char ** argv, args & out) {
    if (argc < 2) {
        return false;
    }
    out.command = argv[1];
    if (out.command != "run" && out.command != "chat" && out.command != "daemon-chat" && out.command != "daemon-session") {
        fprintf(stderr, "unsupported agent command: %s\n", out.command.c_str());
        return false;
    }

    for (int i = 2; i < argc; ++i) {
        auto need_value = [&](const char * name) -> const char * {
            if (i + 1 >= argc) {
                fprintf(stderr, "missing value for %s\n", name);
                return nullptr;
            }
            return argv[++i];
        };

        if (strcmp(argv[i], "--config") == 0) {
            const char * v = need_value(argv[i]); if (!v) return false;
        } else if (strcmp(argv[i], "--backend") == 0) {
            const char * v = need_value(argv[i]); if (!v) return false; out.backend = v;
        } else if (strcmp(argv[i], "--memory-db") == 0) {
            const char * v = need_value(argv[i]); if (!v) return false; out.memory_db = v;
        } else if (strcmp(argv[i], "--model") == 0 || strcmp(argv[i], "-m") == 0) {
            const char * v = need_value(argv[i]); if (!v) return false; out.model = v;
        } else if (strcmp(argv[i], "--embedding-model") == 0) {
            const char * v = need_value(argv[i]); if (!v) return false; out.embedding_model = v;
        } else if (strcmp(argv[i], "--prompt") == 0 || strcmp(argv[i], "-p") == 0) {
            const char * v = need_value(argv[i]); if (!v) return false; out.prompt = v;
        } else if (strcmp(argv[i], "--memory-scope") == 0) {
            const char * v = need_value(argv[i]); if (!v) return false; out.memory_scope = v;
        } else if (strcmp(argv[i], "--memory-namespace") == 0) {
            const char * v = need_value(argv[i]); if (!v) return false; out.memory_namespace = v;
        } else if (strcmp(argv[i], "--memory-session") == 0) {
            const char * v = need_value(argv[i]); if (!v) return false; out.memory_session = v;
        } else if (strcmp(argv[i], "--memory-project") == 0) {
            const char * v = need_value(argv[i]); if (!v) return false; out.memory_project = v;
        } else if (strcmp(argv[i], "--memory-turn") == 0) {
            const char * v = need_value(argv[i]); if (!v) return false; out.memory_turn = v;
        } else if (strcmp(argv[i], "--limit") == 0 || strcmp(argv[i], "--memory-top-k") == 0) {
            const char * v = need_value(argv[i]); if (!v) return false; out.limit = (size_t) std::stoul(v);
        } else if (strcmp(argv[i], "--memory-token-budget") == 0) {
            const char * v = need_value(argv[i]); if (!v) return false; out.memory_token_budget = (size_t) std::stoul(v);
        } else if (strcmp(argv[i], "--max-tool-rounds") == 0) {
            const char * v = need_value(argv[i]); if (!v) return false; out.max_tool_rounds = (size_t) std::stoul(v);
        } else if (strcmp(argv[i], "--thinking-mode") == 0) {
            const char * v = need_value(argv[i]); if (!v) return false; out.thinking_mode = v; out.thinking_mode_explicit = true;
        } else if (strcmp(argv[i], "--max-reflection-rounds") == 0) {
            const char * v = need_value(argv[i]); if (!v) return false; out.max_reflection_rounds = std::stoi(v);
        } else if (strcmp(argv[i], "--max-plan-revisions") == 0) {
            const char * v = need_value(argv[i]); if (!v) return false; out.max_plan_revisions = std::stoi(v);
        } else if (strcmp(argv[i], "--max-research-iterations") == 0) {
            const char * v = need_value(argv[i]); if (!v) return false; out.max_research_iterations = (size_t) std::stoul(v);
        } else if (strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "--n-predict") == 0) {
            const char * v = need_value(argv[i]); if (!v) return false; out.n_predict = std::stoi(v);
        } else if (strcmp(argv[i], "--context-size") == 0) {
            const char * v = need_value(argv[i]); if (!v) return false; out.context_size = std::stoi(v);
            if (out.context_size < 0) { fprintf(stderr, "--context-size must not be negative\n"); return false; }
        } else if (strcmp(argv[i], "--inference-step-timeout-ms") == 0) {
            const char * v = need_value(argv[i]); if (!v) return false;
            const long long timeout_ms = std::stoll(v);
            if (timeout_ms < 0 || timeout_ms > std::numeric_limits<uint32_t>::max()) {
                fprintf(stderr, "--inference-step-timeout-ms is outside the supported range\n");
                return false;
            }
            out.inference_step_timeout_ms = static_cast<uint32_t>(timeout_ms);
        } else if (strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--threads") == 0) {
            const char * v = need_value(argv[i]); if (!v) return false; out.n_threads = std::stoi(v);
            if (out.n_threads < 1) {
                fprintf(stderr, "--threads must be greater than zero\n");
                return false;
            }
        } else if (strcmp(argv[i], "-ngl") == 0) {
            const char * v = need_value(argv[i]); if (!v) return false; out.n_gpu_layers = std::stoi(v);
        } else if (strcmp(argv[i], "--embedding") == 0) {
            const char * v = need_value(argv[i]); if (!v || !parse_embedding(v, out.embedding)) return false;
        } else if (strcmp(argv[i], "--memory-record-episode") == 0) {
            out.record_episode = true;
        } else if (strcmp(argv[i], "--memory-learn") == 0) {
            const char * v = need_value(argv[i]); if (!v) return false; out.memory_learn = v; out.memory_learn_explicit = true;
        } else if (strcmp(argv[i], "--memory-learn-show-candidate") == 0) {
            out.memory_learn_show_candidate = true;
        } else if (strcmp(argv[i], "--memory-learn-min-confidence") == 0) {
            const char * v = need_value(argv[i]); if (!v) return false; out.memory_learn_min_confidence = std::stof(v);
        } else if (strcmp(argv[i], "--memory-learn-min-reuse") == 0) {
            const char * v = need_value(argv[i]); if (!v) return false; out.memory_learn_min_reuse = std::stof(v);
        } else if (strcmp(argv[i], "--agent-inference-backend") == 0) {
            const char * v = need_value(argv[i]); if (!v) return false; out.agent_inference_backend = v;
        } else if (strcmp(argv[i], "--mmproj") == 0) {
            const char * v = need_value(argv[i]); if (!v) return false; out.mmproj = v;
        } else if (strcmp(argv[i], "--tool-profile") == 0) {
            const char * v = need_value(argv[i]); if (!v) return false; out.tool_profile = v; out.tool_profile_explicit = true;
        } else if (strcmp(argv[i], "--mcp-tool-command") == 0) {
            const char * v = need_value(argv[i]); if (!v) return false; out.mcp_tool_command = v;
        } else if (strcmp(argv[i], "--mcp-tool-arg") == 0) {
            const char * v = need_value(argv[i]); if (!v) return false; out.mcp_tool_args.push_back(v);
        } else if (strcmp(argv[i], "--mcp-tool-server-name") == 0) {
            const char * v = need_value(argv[i]); if (!v) return false; out.mcp_tool_server_name = v;
        } else if (strcmp(argv[i], "--mcp-tool-prefix") == 0) {
            const char * v = need_value(argv[i]); if (!v) return false; out.mcp_tool_prefix = v;
        } else if (strcmp(argv[i], "--resource-blob-backend") == 0) {
            const char * v = need_value(argv[i]); if (!v) return false; out.resource_blob_backend = v;
        } else if (strcmp(argv[i], "--resource-blob-root") == 0) {
            const char * v = need_value(argv[i]); if (!v) return false; out.resource_blob_root = v;
        } else if (strcmp(argv[i], "--resource-metadata-backend") == 0) {
            const char * v = need_value(argv[i]); if (!v) return false; out.resource_metadata_backend = v;
        } else if (strcmp(argv[i], "--resource-metadata-db") == 0) {
            const char * v = need_value(argv[i]); if (!v) return false; out.resource_metadata_db = v;
        } else if (strcmp(argv[i], "--resource") == 0) {
            const char * v = need_value(argv[i]); if (!v) return false; out.resource_paths.emplace_back(v);
        } else if (strcmp(argv[i], "--resource-mime-type") == 0) {
            const char * v = need_value(argv[i]); if (!v) return false; out.resource_mime_type = v;
        } else if (strcmp(argv[i], "--agent-profile") == 0) {
            const char * v = need_value(argv[i]); if (!v) return false; out.agent_profile = v; out.agent_profile_explicit = true;
        } else if (strcmp(argv[i], "--plan-scope") == 0) {
            const char * v = need_value(argv[i]); if (!v) return false; out.plan_scope = v;
        } else if (strcmp(argv[i], "--plan-backend") == 0) {
            const char * v = need_value(argv[i]); if (!v) return false; out.plan_backend = v;
        } else if (strcmp(argv[i], "--plan-db") == 0) {
            const char * v = need_value(argv[i]); if (!v) return false; out.plan_db = v;
        } else if (strcmp(argv[i], "--data-backend") == 0) {
            const char * v = need_value(argv[i]); if (!v) return false; out.data_backend = v;
        } else if (strcmp(argv[i], "--data-db") == 0) {
            const char * v = need_value(argv[i]); if (!v) return false; out.data_db = v;
        } else if (strcmp(argv[i], "--plan-id") == 0) {
            const char * v = need_value(argv[i]); if (!v) return false; out.plan_id = v;
        } else if (strcmp(argv[i], "--agent-plan") == 0) {
            const char * v = need_value(argv[i]); if (!v) return false; out.agent_plan = v;
        } else if (strcmp(argv[i], "--repository-root") == 0) {
            const char * v = need_value(argv[i]); if (!v) return false; out.repository_root = v;
        } else if (strcmp(argv[i], "--agent-bootstrap") == 0) {
            const char * v = need_value(argv[i]); if (!v) return false; out.agent_bootstrap = v;
        } else if (strcmp(argv[i], "--agent-import") == 0) {
            const char * v = need_value(argv[i]); if (!v) return false; out.agent_import = v;
        } else if (strcmp(argv[i], "--agent-export") == 0) {
            const char * v = need_value(argv[i]); if (!v) return false; out.agent_export = v;
        } else if (strcmp(argv[i], "--agent-blueprint") == 0) {
            const char * v = need_value(argv[i]); if (!v) return false; out.agent_blueprint = v;
        } else if (strcmp(argv[i], "--plan-show-summary") == 0) {
            out.plan_show_summary = true;
        } else if (strcmp(argv[i], "--include-summary") == 0) {
            out.include_summary = true;
        } else if (strcmp(argv[i], "--agent-trace") == 0) {
            out.agent_trace = true;
        } else if (strcmp(argv[i], "--generation-trace") == 0) {
            out.generation_trace = true;
        } else if (strcmp(argv[i], "--memory-global-opt-in") == 0) {
            out.memory_global_opt_in = true;
        } else {
            fprintf(stderr, "unknown argument: %s\n", argv[i]);
            return false;
        }
    }

    return true;
}

bool validate_agent_memory_scope(const args & a, std::string & error) {
    common_memory_scope scope;
    if (!common_memory_scope_parse(a.memory_scope, scope)) {
        error = "unsupported memory scope: " + a.memory_scope;
        return false;
    }
    if (a.memory_namespace.empty()) {
        error = "memory namespace must not be empty";
        return false;
    }
    if (scope == common_memory_scope::turn && a.memory_turn.empty()) {
        error = "turn-scoped memory requires --memory-turn";
        return false;
    }
    if (scope == common_memory_scope::session && a.memory_session.empty()) {
        error = "session-scoped memory requires --memory-session";
        return false;
    }
    if (scope == common_memory_scope::project && a.memory_project.empty()) {
        error = "project-scoped memory requires --memory-project";
        return false;
    }
    if (scope == common_memory_scope::global && !a.memory_global_opt_in) {
        error = "global memory requires --memory-global-opt-in (local single-user/test environments only)";
        return false;
    }
    error.clear();
    return true;
}
