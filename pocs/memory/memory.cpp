#include "memory/memory-context.h"
#include "memory/memory-in-memory.h"

#include "tools/agent/cli/agent-cli-options.h"
#include "tools/agent/cli/agent-cli-memory-tools.h"
#include "tools/agent/cli/agent-cli-command.h"
#include "tools/agent/cli/agent-cli-config.h"

#ifdef LLAMA_MEMORY_POC_USE_AGENT_TOOLS
#include "plan/plan-in-memory.h"
#ifdef LLAMA_PLAN_USE_COZO
#include "plan/cozo/plan-cozo.h"
#endif
#endif

#ifdef LLAMA_MEMORY_USE_COZO
#include "memory/cozo/memory-cozo.h"
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <algorithm>
#include <atomic>
#include <memory>
#include <sstream>
static void usage(const char * argv0) {
    fprintf(stderr,
        "usage:\n"
        "  %s add --memory-db PATH --id ID --kind KIND --content TEXT [--memory-scope turn|session|project|global] [--memory-namespace ID] [--memory-session ID] [--memory-project ID] [--memory-turn ID] [--memory-global-opt-in] [--embedding VALUE|--embedding-model MODEL] [--backend cozo]\n"
        "  %s search --memory-db PATH --query TEXT [--memory-scope turn|session|project|global] [--memory-namespace ID] [--memory-session ID] [--memory-project ID] [--memory-turn ID] [--memory-global-opt-in] [--limit N] [--embedding VALUE|--embedding-model MODEL] [--backend cozo]\n"
        "  %s relate --memory-db PATH --from ID --relation REL --to ID [--weight W] [--backend cozo]\n"
        "  %s chat --memory-db PATH --model MODEL --prompt TEXT [--embedding-model MODEL] [--agent-profile default|learning|research|safe|static] [--agent-bootstrap none|default|--agent-import PATH|--agent-export PATH] [--agent-plan off|auto] [--agent-blueprint ID --plan-id ID] [--plan-backend in-memory|cozo] [--plan-db PATH]\n",
        argv0, argv0, argv0, argv0);
}

static bool parse_args(int argc, char ** argv, args & out) {
    if (argc < 2) {
        return false;
    }
    out.command = argv[1];
    if (out.command == "chat") {
        return parse_agent_run_args(argc, argv, out);
    }
    for (int i = 2; i < argc; ++i) {
        auto need_value = [&](const char * name) -> const char * {
            if (i + 1 >= argc) {
                fprintf(stderr, "missing value for %s\n", name);
                return nullptr;
            }
            return argv[++i];
        };

        if (strcmp(argv[i], "--backend") == 0) {
            const char * v = need_value(argv[i]); if (!v) return false; out.backend = v;
        } else if (strcmp(argv[i], "--memory-db") == 0) {
            const char * v = need_value(argv[i]); if (!v) return false; out.memory_db = v;
        } else if (strcmp(argv[i], "--id") == 0) {
            const char * v = need_value(argv[i]); if (!v) return false; out.id = v;
        } else if (strcmp(argv[i], "--kind") == 0) {
            const char * v = need_value(argv[i]); if (!v) return false; out.kind = v;
        } else if (strcmp(argv[i], "--content") == 0) {
            const char * v = need_value(argv[i]); if (!v) return false; out.content = v;
        } else if (strcmp(argv[i], "--query") == 0) {
            const char * v = need_value(argv[i]); if (!v) return false; out.query = v;
        } else if (strcmp(argv[i], "--from") == 0) {
            const char * v = need_value(argv[i]); if (!v) return false; out.from = v;
        } else if (strcmp(argv[i], "--relation") == 0) {
            const char * v = need_value(argv[i]); if (!v) return false; out.relation = v;
        } else if (strcmp(argv[i], "--to") == 0) {
            const char * v = need_value(argv[i]); if (!v) return false; out.to = v;
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
        } else if (strcmp(argv[i], "--importance") == 0) {
            const char * v = need_value(argv[i]); if (!v) return false; out.importance = std::stof(v);
        } else if (strcmp(argv[i], "--confidence") == 0) {
            const char * v = need_value(argv[i]); if (!v) return false; out.confidence = std::stof(v);
        } else if (strcmp(argv[i], "--weight") == 0) {
            const char * v = need_value(argv[i]); if (!v) return false; out.weight = std::stof(v);
        } else if (strcmp(argv[i], "--limit") == 0 || strcmp(argv[i], "--memory-top-k") == 0) {
            const char * v = need_value(argv[i]); if (!v) return false; out.limit = (size_t) std::stoul(v);
        } else if (strcmp(argv[i], "--embedding-model") == 0) {
            const char * v = need_value(argv[i]); if (!v) return false; out.embedding_model = v;
        } else if (strcmp(argv[i], "--embedding") == 0) {
            const char * v = need_value(argv[i]); if (!v) return false;
            std::stringstream ss(v);
            std::string item;
            out.embedding.clear();
            while (std::getline(ss, item, ',')) {
                try {
                    out.embedding.push_back(std::stof(item));
                } catch (...) {
                    return false;
                }
            }
        } else if (strcmp(argv[i], "--memory-search-tool") == 0) {
            out.enable_memory_search_tool = true;
        } else if (strcmp(argv[i], "--memory-remember-tool") == 0) {
            out.enable_memory_remember_tool = true;
        } else if (strcmp(argv[i], "--memory-global-opt-in") == 0) {
            out.memory_global_opt_in = true;
        } else {
            fprintf(stderr, "unknown argument: %s\n", argv[i]);
            return false;
        }
    }
    return true;
}

static bool memory_scope_from_args(const args & a, common_memory_scope & scope, std::string & error) {
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

int main(int argc, char ** argv) {
    args a;
    if (!parse_args(argc, argv, a)) {
        usage(argv[0]);
        return 1;
    }

    std::string error;
    common_memory_scope parsed_scope;
    if (!memory_scope_from_args(a, parsed_scope, error)) {
        fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    auto store = make_memory_store(a, error);
    if (!store) {
        fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    if (!open_memory_store(*store, a, error)) {
        fprintf(stderr, "failed to open memory store: %s\n", error.c_str());
        return 1;
    }

    if (a.command == "add") {
        common_memory_kind kind;
        if (!common_memory_kind_parse(a.kind, kind)) {
            fprintf(stderr, "unsupported memory kind: %s\n", a.kind.c_str());
            return 1;
        }
        common_memory_record record;
        record.id = a.id;
        record.kind = kind;
        record.content = a.content;
        record.embedding = a.embedding;
        if (!ensure_memory_cli_embedding(a, record.content, record.embedding, "memory", error)) {
            fprintf(stderr, "failed to generate memory embedding: %s\n", error.c_str());
            return 1;
        }
        record.created_at = std::time(nullptr);
        record.accessed_at = record.created_at;
        record.importance = a.importance;
        record.confidence = a.confidence;
        apply_memory_scope(a, record);
        if (!store->put(record, error)) {
            fprintf(stderr, "failed to add memory: %s\n", error.c_str());
            return 1;
        }
        fprintf(stderr, "added memory %s\n", record.id.c_str());
    } else if (a.command == "search") {
        common_memory_query query;
        query.text = a.query;
        query.embedding = a.embedding;
        if (!ensure_memory_cli_embedding(a, query.text, query.embedding, "query", error)) {
            fprintf(stderr, "failed to generate query embedding: %s\n", error.c_str());
            return 1;
        }
        query.limit = a.limit;
        apply_memory_scope(a, query);
        auto hits = store->search(query, error);
        if (!error.empty()) {
            fprintf(stderr, "search failed: %s\n", error.c_str());
            return 1;
        }
        for (const auto & hit : hits) {
            printf("%s\t%s\t%.4f\t%s\n", hit.memory.id.c_str(), common_memory_kind_name(hit.memory.kind), hit.final_score, hit.memory.content.c_str());
        }
    } else if (a.command == "relate") {
        if (!store->relate(a.from, a.relation, a.to, a.weight, error)) {
            fprintf(stderr, "failed to relate memories: %s\n", error.c_str());
            return 1;
        }
        fprintf(stderr, "related %s --%s--> %s\n", a.from.c_str(), a.relation.c_str(), a.to.c_str());
    } else if (a.command == "chat") {
        return run_memory_chat_command(argv[0], *store, a);
    } else {
        usage(argv[0]);
        return 1;
    }

    store->close();
    return 0;
}
