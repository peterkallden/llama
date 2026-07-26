#include "agent-cli-memory-tools.h"

#include "agent/agent-scope.h"
#include "tools/agent/cli/agent-cli-scope.h"
#include "common.h"
#include "llama.h"
#include "memory/memory-context.h"
#include "memory/memory-policy.h"
#include "memory/memory-retrieval.h"
#include "memory/memory-tool-service.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <memory>

#include <nlohmann/json.hpp>

using json = nlohmann::ordered_json;

namespace {

constexpr size_t k_memory_search_max_limit = 8;
constexpr size_t k_memory_search_max_query_chars = 1024;
constexpr size_t k_memory_remember_max_content_chars = 512;
constexpr size_t k_memory_remember_max_rationale_chars = 240;

common_agent_scope make_memory_cli_scope(const args & a) {
    return common_cli_make_agent_scope(a, common_plan_scope::turn);
}

std::string embedding_model_path(const args & a) {
    // A chat model is not an embedding model.  Falling back to a.model here
    // makes a normal agent run load and evaluate the chat model a second time
    // for retrieval, and may produce vectors with an incompatible meaning.
    return a.embedding_model;
}

struct embedding_model_cache {
    ~embedding_model_cache() {
        if (model != nullptr) {
            llama_model_free(model);
        }
    }

    bool load(const std::string & requested_path, int requested_n_gpu_layers, std::string & error) {
        if (model != nullptr && path == requested_path && n_gpu_layers == requested_n_gpu_layers) {
            return true;
        }

        if (model != nullptr) {
            llama_model_free(model);
            model = nullptr;
            path.clear();
        }

        ggml_backend_load_all();

        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = requested_n_gpu_layers;
        model = llama_model_load_from_file(requested_path.c_str(), model_params);
        if (model == nullptr) {
            error = "failed to load embedding model: " + requested_path;
            return false;
        }

        path = requested_path;
        n_gpu_layers = requested_n_gpu_layers;
        return true;
    }

    std::string path;
    int n_gpu_layers = -1;
    llama_model * model = nullptr;
};

embedding_model_cache & local_embedding_model_cache() {
    static embedding_model_cache cache;
    return cache;
}

bool compute_text_embedding(
        const std::string & model_path,
        const std::string & text,
        int n_gpu_layers,
        std::vector<float> & out,
        std::string & error) {
    if (model_path.empty()) {
        error = "embedding generation requested without a model path";
        return false;
    }
    if (text.empty()) {
        error = "cannot generate an embedding from empty text";
        return false;
    }

    embedding_model_cache & cache = local_embedding_model_cache();
    if (!cache.load(model_path, n_gpu_layers, error)) {
        return false;
    }
    llama_model * model = cache.model;

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_tokens = -llama_tokenize(vocab, text.c_str(), text.size(), nullptr, 0, true, true);
    if (n_tokens <= 0) {
        error = "failed to tokenize text for embedding generation";
        return false;
    }

    std::vector<llama_token> tokens(n_tokens);
    if (llama_tokenize(vocab, text.c_str(), text.size(), tokens.data(), tokens.size(), true, true) < 0) {
        error = "failed to tokenize text for embedding generation";
        return false;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_tokens;
    ctx_params.n_batch = n_tokens;
    ctx_params.embeddings = true;
    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (ctx == nullptr) {
        error = "failed to create embedding context";
        return false;
    }

    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
    llama_memory_clear(llama_get_memory(ctx), true);
    if (llama_decode(ctx, batch) != 0) {
        llama_free(ctx);
        error = "embedding decode failed";
        return false;
    }

    const int n_embd = llama_model_n_embd_out(model);
    out.assign(n_embd, 0.0f);

    const enum llama_pooling_type pooling_type = llama_pooling_type(ctx);
    if (pooling_type == LLAMA_POOLING_TYPE_NONE) {
        std::vector<float> pooled(n_embd, 0.0f);
        int used = 0;
        for (int i = 0; i < n_tokens; ++i) {
            const float * embd = llama_get_embeddings_ith(ctx, i);
            if (embd == nullptr) {
                continue;
            }
            for (int j = 0; j < n_embd; ++j) {
                pooled[j] += embd[j];
            }
            used++;
        }
        if (used == 0) {
            llama_free(ctx);
            error = "embedding model did not expose token embeddings";
            return false;
        }
        for (float & value : pooled) {
            value /= used;
        }
        common_embd_normalize(pooled.data(), out.data(), n_embd, 2);
    } else {
        const float * embd = llama_get_embeddings_seq(ctx, 0);
        if (embd == nullptr) {
            llama_free(ctx);
            error = "embedding model did not expose a pooled sequence embedding";
            return false;
        }
        common_embd_normalize(embd, out.data(), n_embd, 2);
    }

    llama_free(ctx);
    error.clear();
    return true;
}

void log_memory_remember_audit(
        common_memory_remember_decision decision,
        const common_memory_remember_request & request,
        const common_memory_remember_result & result) {
    fprintf(stderr,
        "audit: memory_remember decision=%s kind=%s scope=%s namespace=%s reason=%s related=%zu content=\"%s\"\n",
        common_memory_remember_decision_name(decision),
        common_memory_kind_name(request.kind),
        common_memory_scope_name(request.scope),
        request.namespace_id.c_str(),
        result.reason.c_str(),
        result.related_hits.size(),
        request.content.c_str());
}

} // namespace

void apply_memory_scope(const args & a, common_memory_record & record) {
    common_agent_scope_apply(make_memory_cli_scope(a), record);
}

void apply_memory_scope(const args & a, common_memory_query & query) {
    common_agent_scope_apply(make_memory_cli_scope(a), query);
}

bool ensure_memory_cli_embedding(
        const args & a,
        const std::string & text,
        std::vector<float> & embedding,
        const char * label,
        std::string & error) {
    return ensure_memory_cli_embedding_from_model(
        embedding_model_path(a),
        a.n_gpu_layers,
        text,
        embedding,
        label,
        error);
}

bool ensure_memory_cli_embedding_from_model(
        const std::string & model_path,
        int n_gpu_layers,
        const std::string & text,
        std::vector<float> & embedding,
        const char * label,
        std::string & error) {
    if (!embedding.empty()) {
        return true;
    }

    if (model_path.empty()) {
        return true;
    }

    if (!compute_text_embedding(model_path, text, n_gpu_layers, embedding, error)) {
        return false;
    }

    fprintf(stderr, "generated %s embedding with %zu dimensions using %s\n", label, embedding.size(), model_path.c_str());
    return true;
}

common_chat_tool memory_search_tool_definition() {
    common_chat_tool tool;
    tool.name = "memory_search";
    tool.description = "Search the user's stored memory for relevant factual context. This is read-only. Use it only when stored memory could answer the user's question.";
    tool.parameters = R"({"type":"object","properties":{"query":{"type":"string","description":"Focused natural-language memory query.","maxLength":1024},"limit":{"type":"integer","description":"Maximum number of memories to return.","minimum":1,"maximum":8}},"required":["query"],"additionalProperties":false})";
    return tool;
}

common_chat_tool memory_remember_tool_definition() {
    common_chat_tool tool;
    tool.name = "memory_remember";
    tool.description = "Propose a low-risk durable memory based on what the user explicitly stated. Native policy may reject, deduplicate, or store it.";
    tool.parameters = R"({"type":"object","properties":{"kind":{"type":"string","description":"Memory kind to propose.","enum":["fact","preference","procedure","constraint","decision","goal","observation","reflection","episode"]},"content":{"type":"string","description":"Single durable memory candidate stated as a concise sentence.","maxLength":512},"importance":{"type":"number","description":"Estimated importance from 0 to 1.","minimum":0.0,"maximum":1.0},"confidence":{"type":"number","description":"Estimated confidence from 0 to 1.","minimum":0.0,"maximum":1.0},"rationale":{"type":"string","description":"Short reason this might be worth remembering.","maxLength":240}},"required":["kind","content"],"additionalProperties":false})";
    return tool;
}

std::string memory_search_tool_result(
        common_memory_store & store,
        const args & a,
        const std::string & arguments) {
    json result = json::object();
    result["ok"] = false;

    try {
        const json request = json::parse(arguments);
        if (!request.is_object()) {
            result["error"] = "arguments must be a JSON object";
            return result.dump();
        }
        for (const auto & item : request.items()) {
            if (item.key() != "query" && item.key() != "limit") {
                result["error"] = "unsupported argument: " + item.key();
                return result.dump();
            }
        }
        if (!request.contains("query") || !request.at("query").is_string()) {
            result["error"] = "query must be a string";
            return result.dump();
        }

        common_memory_tool_context context;
        context.max_search_limit = k_memory_search_max_limit;
        context.max_query_chars = k_memory_search_max_query_chars;
        context.query_defaults.limit = std::clamp(a.limit, size_t(1), k_memory_search_max_limit);
        context.query_defaults.token_budget = a.memory_token_budget;
        apply_memory_scope(a, context.query_defaults);
        context.embed = [&a](const std::string & text, std::vector<float> & embedding, std::string & tool_error) {
            if (!ensure_memory_cli_embedding(a, text, embedding, "tool query", tool_error)) {
                tool_error = "unable to generate a memory query embedding: " + tool_error;
                return false;
            }
            return true;
        };

        common_memory_tool_search_result search_result;
        common_memory_tool_service service(store);
        std::string error;
        if (!service.search(context, arguments, search_result, error)) {
            result["error"] = "memory search failed: " + error;
            return result.dump();
        }

        result["ok"] = true;
        result["query"] = search_result.query;
        result["count"] = search_result.hits.size();
        result["context"] = search_result.context;
        fprintf(stderr, "debug: memory_search returned %zu result(s)\n", search_result.hits.size());
    } catch (const std::exception & e) {
        result["error"] = std::string("invalid tool arguments: ") + e.what();
    }

    return result.dump();
}

std::string memory_remember_tool_result(
        common_memory_store & store,
        const args & a,
        const std::string & arguments) {
    json result = json::object();
    result["ok"] = false;

    try {
        const json request = json::parse(arguments);
        if (!request.is_object()) {
            result["error"] = "arguments must be a JSON object";
            return result.dump();
        }
        for (const auto & item : request.items()) {
            if (item.key() != "kind" && item.key() != "content" && item.key() != "importance" &&
                    item.key() != "confidence" && item.key() != "rationale") {
                result["error"] = "unsupported argument: " + item.key();
                return result.dump();
            }
        }
        if (!request.contains("kind") || !request.at("kind").is_string()) {
            result["error"] = "kind must be a string";
            return result.dump();
        }
        if (!request.contains("content") || !request.at("content").is_string()) {
            result["error"] = "content must be a string";
            return result.dump();
        }

        common_memory_tool_context context;
        context.max_content_chars = k_memory_remember_max_content_chars;
        context.max_rationale_chars = k_memory_remember_max_rationale_chars;
        context.allow_write_proposals = true;
        apply_memory_scope(a, context.query_defaults);
        context.now = std::time(nullptr);
        context.embed = [&a](const std::string & text, std::vector<float> & embedding, std::string & tool_error) {
            if (!ensure_memory_cli_embedding(a, text, embedding, "tool memory", tool_error)) {
                tool_error = "unable to generate a memory embedding: " + tool_error;
                return false;
            }
            return true;
        };

        common_memory_tool_remember_result remember_result;
        common_memory_tool_service service(store);
        std::string error;
        if (!service.remember_proposal(context, arguments, remember_result, error)) {
            result["error"] = "memory policy evaluation failed: " + error;
            return result.dump();
        }

        const auto & proposal = remember_result.proposal;
        const auto & decision = remember_result.decision;
        result["ok"] = true;
        result["decision"] = common_memory_remember_decision_name(decision.decision);
        result["reason"] = decision.reason;
        result["kind"] = common_memory_kind_name(proposal.kind);
        result["scope"] = common_memory_scope_name(proposal.scope);
        result["content"] = proposal.content;
        result["related_count"] = decision.related_hits.size();
        if (!decision.related_hits.empty()) {
            json related = json::array();
            for (const auto & hit : decision.related_hits) {
                related.push_back({
                    {"id", hit.memory.id},
                    {"kind", common_memory_kind_name(hit.memory.kind)},
                    {"score", hit.final_score},
                    {"content", hit.memory.content},
                });
            }
            result["related"] = std::move(related);
        }

        if (decision.record.has_value()) {
            if (!store.put(*decision.record, error)) {
                result["ok"] = false;
                result["decision"] = "reject";
                result["error"] = "failed to persist accepted memory: " + error;
            } else {
                result["id"] = decision.record->id;
                fprintf(stderr, "debug: memory_remember stored %s\n", decision.record->id.c_str());
            }
        }

        log_memory_remember_audit(decision.decision, proposal, decision);
    } catch (const std::exception & e) {
        result["error"] = std::string("invalid tool arguments: ") + e.what();
    }

    return result.dump();
}
