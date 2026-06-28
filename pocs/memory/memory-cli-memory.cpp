#include "memory-cli-memory.h"

#include "agent/agent-scope.h"
#include "common.h"
#include "llama.h"
#include "memory/memory-context.h"
#include "memory/memory-policy.h"
#include "memory/memory-retrieval.h"

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
    common_agent_scope scope;
    common_memory_scope_parse(a.memory_scope, scope.memory_scope);
    scope.namespace_id = a.memory_namespace;
    scope.session_id = a.memory_session;
    scope.project_id = a.memory_project;
    scope.turn_id = a.memory_turn;
    scope.memory_global_opt_in = a.memory_global_opt_in;
    return scope;
}

std::string embedding_model_path(const args & a) {
    if (!a.embedding_model.empty()) {
        return a.embedding_model;
    }
    return a.model;
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
    if (!embedding.empty()) {
        return true;
    }

    const std::string model_path = embedding_model_path(a);
    if (model_path.empty()) {
        return true;
    }

    if (!compute_text_embedding(model_path, text, a.n_gpu_layers, embedding, error)) {
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
    tool.parameters = R"({"type":"object","properties":{"kind":{"type":"string","description":"Memory kind to propose.","enum":["fact","preference","procedure","goal","observation","reflection","episode"]},"content":{"type":"string","description":"Single durable memory candidate stated as a concise sentence.","maxLength":512},"importance":{"type":"number","description":"Estimated importance from 0 to 1.","minimum":0.0,"maximum":1.0},"confidence":{"type":"number","description":"Estimated confidence from 0 to 1.","minimum":0.0,"maximum":1.0},"rationale":{"type":"string","description":"Short reason this might be worth remembering.","maxLength":240}},"required":["kind","content"],"additionalProperties":false})";
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

        const std::string query_text = request.at("query").get<std::string>();
        if (query_text.empty() || query_text.size() > k_memory_search_max_query_chars) {
            result["error"] = "query must contain between 1 and 1024 characters";
            return result.dump();
        }

        size_t limit = std::clamp(a.limit, size_t(1), k_memory_search_max_limit);
        if (request.contains("limit")) {
            if (!request.at("limit").is_number_unsigned() && !request.at("limit").is_number_integer()) {
                result["error"] = "limit must be an integer";
                return result.dump();
            }
            const int requested_limit = request.at("limit").get<int>();
            if (requested_limit < 1 || requested_limit > (int) k_memory_search_max_limit) {
                result["error"] = "limit must be between 1 and 8";
                return result.dump();
            }
            limit = (size_t) requested_limit;
        }

        common_memory_query query;
        query.text = query_text;
        query.limit = limit;
        query.token_budget = a.memory_token_budget;
        apply_memory_scope(a, query);

        std::string error;
        if (!ensure_memory_cli_embedding(a, query.text, query.embedding, "tool query", error)) {
            result["error"] = "unable to generate a memory query embedding: " + error;
            return result.dump();
        }

        common_memory_retrieval retrieval(store);
        const auto hits = retrieval.retrieve(query, error);
        if (!error.empty()) {
            result["error"] = "memory search failed: " + error;
            return result.dump();
        }

        common_memory_context_config context_config;
        context_config.char_budget = a.memory_token_budget * 4;
        result["ok"] = true;
        result["query"] = query_text;
        result["count"] = hits.size();
        result["context"] = common_memory_render_context(hits, context_config);
        fprintf(stderr, "debug: memory_search returned %zu result(s)\n", hits.size());
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

        common_memory_remember_request proposal;
        const auto scope = make_memory_cli_scope(a);
        if (!common_memory_kind_parse(request.at("kind").get<std::string>(), proposal.kind)) {
            result["error"] = "unsupported memory kind";
            return result.dump();
        }
        proposal.content = request.at("content").get<std::string>();
        proposal.scope = scope.memory_scope;
        proposal.namespace_id = scope.namespace_id;
        proposal.session_id = scope.session_id;
        proposal.project_id = scope.project_id;
        proposal.turn_id = scope.turn_id;
        proposal.global_opt_in = scope.memory_global_opt_in;
        if (proposal.content.empty() || proposal.content.size() > k_memory_remember_max_content_chars) {
            result["error"] = "content must contain between 1 and 512 characters";
            return result.dump();
        }

        if (request.contains("importance")) {
            if (!request.at("importance").is_number()) {
                result["error"] = "importance must be a number";
                return result.dump();
            }
            proposal.importance = request.at("importance").get<float>();
        }
        if (request.contains("confidence")) {
            if (!request.at("confidence").is_number()) {
                result["error"] = "confidence must be a number";
                return result.dump();
            }
            proposal.confidence = request.at("confidence").get<float>();
        }
        if (request.contains("rationale")) {
            if (!request.at("rationale").is_string()) {
                result["error"] = "rationale must be a string";
                return result.dump();
            }
            proposal.rationale = request.at("rationale").get<std::string>();
            if (proposal.rationale.size() > k_memory_remember_max_rationale_chars) {
                result["error"] = "rationale must contain at most 240 characters";
                return result.dump();
            }
        }

        std::vector<float> embedding;
        std::string error;
        if (!ensure_memory_cli_embedding(a, proposal.content, embedding, "tool memory", error)) {
            result["error"] = "unable to generate a memory embedding: " + error;
            return result.dump();
        }

        const auto decision = common_memory_evaluate_remember_request(
            store, proposal, embedding, std::time(nullptr), error);
        if (!error.empty()) {
            result["error"] = "memory policy evaluation failed: " + error;
            return result.dump();
        }

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
