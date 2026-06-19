#include "memory/memory-context.h"
#include "memory/memory-in-memory.h"
#include "memory/memory-retrieval.h"

#include "llama.h"

#ifdef LLAMA_MEMORY_USE_COZO
#include "memory/cozo/memory-cozo.h"
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <memory>
#include <sstream>

struct args {
    std::string command;
    std::string backend = "in-memory";
    std::string memory_db;
    std::string id;
    std::string kind = "episode";
    std::string content;
    std::string query;
    std::string from;
    std::string relation;
    std::string to;
    std::string model;
    std::string prompt;
    std::vector<float> embedding;
    float importance = 0.5f;
    float confidence = 0.5f;
    float weight = 1.0f;
    size_t limit = 8;
    size_t memory_token_budget = 768;
    int n_predict = 128;
    int n_gpu_layers = 99;
    bool record_episode = false;
};

static void usage(const char * argv0) {
    fprintf(stderr,
        "usage:\n"
        "  %s add --memory-db PATH --id ID --kind KIND --content TEXT [--backend cozo]\n"
        "  %s search --memory-db PATH --query TEXT [--limit N] [--backend cozo]\n"
        "  %s relate --memory-db PATH --from ID --relation REL --to ID [--weight W] [--backend cozo]\n"
        "  %s chat --memory-db PATH --model MODEL --prompt TEXT [--memory-top-k N] [--memory-record-episode]\n",
        argv0, argv0, argv0, argv0);
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

static bool parse_args(int argc, char ** argv, args & out) {
    if (argc < 2) {
        return false;
    }
    out.command = argv[1];
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
        } else if (strcmp(argv[i], "--model") == 0 || strcmp(argv[i], "-m") == 0) {
            const char * v = need_value(argv[i]); if (!v) return false; out.model = v;
        } else if (strcmp(argv[i], "--prompt") == 0 || strcmp(argv[i], "-p") == 0) {
            const char * v = need_value(argv[i]); if (!v) return false; out.prompt = v;
        } else if (strcmp(argv[i], "--importance") == 0) {
            const char * v = need_value(argv[i]); if (!v) return false; out.importance = std::stof(v);
        } else if (strcmp(argv[i], "--confidence") == 0) {
            const char * v = need_value(argv[i]); if (!v) return false; out.confidence = std::stof(v);
        } else if (strcmp(argv[i], "--weight") == 0) {
            const char * v = need_value(argv[i]); if (!v) return false; out.weight = std::stof(v);
        } else if (strcmp(argv[i], "--limit") == 0 || strcmp(argv[i], "--memory-top-k") == 0) {
            const char * v = need_value(argv[i]); if (!v) return false; out.limit = (size_t) std::stoul(v);
        } else if (strcmp(argv[i], "--memory-token-budget") == 0) {
            const char * v = need_value(argv[i]); if (!v) return false; out.memory_token_budget = (size_t) std::stoul(v);
        } else if (strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "--n-predict") == 0) {
            const char * v = need_value(argv[i]); if (!v) return false; out.n_predict = std::stoi(v);
        } else if (strcmp(argv[i], "-ngl") == 0) {
            const char * v = need_value(argv[i]); if (!v) return false; out.n_gpu_layers = std::stoi(v);
        } else if (strcmp(argv[i], "--embedding") == 0) {
            const char * v = need_value(argv[i]); if (!v || !parse_embedding(v, out.embedding)) return false;
        } else if (strcmp(argv[i], "--memory-record-episode") == 0) {
            out.record_episode = true;
        } else {
            fprintf(stderr, "unknown argument: %s\n", argv[i]);
            return false;
        }
    }
    return true;
}

static std::unique_ptr<common_memory_store> make_store(const args & a, std::string & error) {
    if (a.backend == "in-memory") {
        return std::unique_ptr<common_memory_store>(new common_memory_in_memory_store());
    }
    if (a.backend == "cozo") {
#ifdef LLAMA_MEMORY_USE_COZO
        return std::unique_ptr<common_memory_store>(new common_memory_cozo_store());
#else
        error = "this binary was built without LLAMA_MEMORY_COZO";
        return nullptr;
#endif
    }
    error = "unknown memory backend: " + a.backend;
    return nullptr;
}

static bool open_store(common_memory_store & store, const args & a, std::string & error) {
    return store.open(a.memory_db, error);
}

static int run_chat(common_memory_store & store, const args & a) {
    std::string error;
    common_memory_query query;
    query.text = a.prompt;
    query.limit = a.limit;
    query.token_budget = a.memory_token_budget;

    common_memory_retrieval retrieval(store);
    auto hits = retrieval.retrieve(query, error);
    if (!error.empty()) {
        fprintf(stderr, "memory retrieval failed: %s\n", error.c_str());
        return 1;
    }
    for (const auto & hit : hits) {
        fprintf(stderr, "memory: id=%s score=%.4f provenance=%s\n", hit.memory.id.c_str(), hit.final_score, hit.provenance.c_str());
    }

    common_memory_context_config ctx_cfg;
    ctx_cfg.char_budget = a.memory_token_budget * 4;
    const std::string memory_context = common_memory_render_context(hits, ctx_cfg);
    const std::string final_prompt = memory_context.empty() ? a.prompt : memory_context + "\n" + a.prompt;

    ggml_backend_load_all();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = a.n_gpu_layers;
    llama_model * model = llama_model_load_from_file(a.model.c_str(), model_params);
    if (model == nullptr) {
        fprintf(stderr, "failed to load model: %s\n", a.model.c_str());
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_prompt = -llama_tokenize(vocab, final_prompt.c_str(), final_prompt.size(), nullptr, 0, true, true);
    std::vector<llama_token> prompt_tokens(n_prompt);
    if (llama_tokenize(vocab, final_prompt.c_str(), final_prompt.size(), prompt_tokens.data(), prompt_tokens.size(), true, true) < 0) {
        fprintf(stderr, "failed to tokenize prompt\n");
        llama_model_free(model);
        return 1;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_prompt + a.n_predict;
    ctx_params.n_batch = n_prompt;
    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (ctx == nullptr) {
        fprintf(stderr, "failed to create llama context\n");
        llama_model_free(model);
        return 1;
    }

    auto sparams = llama_sampler_chain_default_params();
    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
    int n_decode = 0;
    for (int n_pos = 0; n_pos + batch.n_tokens < n_prompt + a.n_predict; ) {
        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "failed to decode\n");
            break;
        }
        n_pos += batch.n_tokens;
        llama_token token = llama_sampler_sample(smpl, ctx, -1);
        if (llama_vocab_is_eog(vocab, token)) {
            break;
        }
        char buf[256];
        const int n = llama_token_to_piece(vocab, token, buf, sizeof(buf), 0, true);
        if (n > 0) {
            printf("%.*s", n, buf);
            fflush(stdout);
        }
        batch = llama_batch_get_one(&token, 1);
        n_decode++;
    }
    printf("\n");

    if (a.record_episode) {
        common_memory_record episode;
        episode.id = "episode-" + std::to_string(std::time(nullptr));
        episode.kind = common_memory_kind::episode;
        episode.content = a.prompt;
        episode.importance = 0.5f;
        episode.confidence = 0.5f;
        if (!store.put(episode, error)) {
            fprintf(stderr, "failed to record memory episode: %s\n", error.c_str());
        }
    }

    fprintf(stderr, "decoded %d tokens\n", n_decode);
    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}

int main(int argc, char ** argv) {
    args a;
    if (!parse_args(argc, argv, a)) {
        usage(argv[0]);
        return 1;
    }

    std::string error;
    auto store = make_store(a, error);
    if (!store) {
        fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    if (!open_store(*store, a, error)) {
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
        record.importance = a.importance;
        record.confidence = a.confidence;
        if (!store->put(record, error)) {
            fprintf(stderr, "failed to add memory: %s\n", error.c_str());
            return 1;
        }
        fprintf(stderr, "added memory %s\n", record.id.c_str());
    } else if (a.command == "search") {
        common_memory_query query;
        query.text = a.query;
        query.embedding = a.embedding;
        query.limit = a.limit;
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
        if (a.model.empty() || a.prompt.empty()) {
            usage(argv[0]);
            return 1;
        }
        return run_chat(*store, a);
    } else {
        usage(argv[0]);
        return 1;
    }

    store->close();
    return 0;
}
