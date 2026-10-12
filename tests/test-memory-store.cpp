#include "memory/memory-in-memory.h"

#include <cassert>
#include <cmath>

static common_memory_record make_record(const std::string & id, const std::string & content, std::vector<float> embedding = {}) {
    common_memory_record record;
    record.id = id;
    record.kind = common_memory_kind::fact;
    record.content = content;
    record.embedding = std::move(embedding);
    record.importance = 0.6f;
    record.confidence = 0.7f;
    return record;
}

int main() {
    std::string error;
    common_memory_in_memory_store store;
    assert(store.open("", error));

    assert(store.put(make_record("fact-1", "zero budget package search", {1.0f, 0.0f}), error));
    auto got = store.get("fact-1", error);
    assert(got);
    assert(got->content == "zero budget package search");

    assert(store.put(make_record("fact-1", "updated memory", {0.0f, 1.0f}), error));
    got = store.get("fact-1", error);
    assert(got);
    assert(got->content == "updated memory");

    assert(store.put(make_record("episode-1", "the earlier pass produced the fact", {0.0f, 1.0f}), error));
    assert(store.relate("episode-1", "produced", "fact-1", 1.0f, error));

    common_memory_query graph_query;
    graph_query.text = "earlier pass";
    graph_query.limit = 4;
    graph_query.graph_depth = 1;
    auto graph_hits = store.search(graph_query, error);
    assert(graph_hits.size() == 2);

    common_memory_query emb_query;
    emb_query.embedding = {0.0f, 1.0f};
    emb_query.limit = 1;
    auto emb_hits = store.search(emb_query, error);
    assert(emb_hits.size() == 1);
    assert(emb_hits[0].memory.id == "episode-1" || emb_hits[0].memory.id == "fact-1");

    auto session_a = make_record("session-a", "scope isolated memory", {0.5f, 0.5f});
    session_a.session_id = "session-a";
    auto session_b = make_record("session-b", "scope isolated memory", {0.5f, 0.5f});
    session_b.session_id = "session-b";
    auto global = make_record("global", "globally opt-in memory", {0.5f, 0.5f});
    global.scope = common_memory_scope::global;
    assert(store.put(session_a, error));
    assert(store.put(session_b, error));
    assert(store.put(global, error));

    common_memory_query scoped_query;
    scoped_query.text = "scope isolated";
    scoped_query.session_id = "session-a";
    scoped_query.limit = 8;
    auto scoped_hits = store.search(scoped_query, error);
    assert(scoped_hits.size() == 1);
    assert(scoped_hits[0].memory.id == "session-a");

    scoped_query.scope = common_memory_scope::global;
    scoped_query.text = "globally";
    scoped_query.global_opt_in = false;
    assert(store.search(scoped_query, error).empty());
    scoped_query.global_opt_in = true;
    scoped_hits = store.search(scoped_query, error);
    assert(scoped_hits.size() == 1);
    assert(scoped_hits[0].memory.id == "global");

    assert(std::fabs(common_memory_cosine_similarity({1.0f, 0.0f}, {1.0f, 0.0f}) - 1.0f) < 0.0001f);
    assert(common_memory_cosine_similarity({}, {1.0f}) == 0.0f);
    assert(common_memory_cosine_similarity({1.0f}, {1.0f, 2.0f}) == 0.0f);
    assert(common_memory_cosine_similarity({0.0f, 0.0f}, {1.0f, 2.0f}) == 0.0f);
    assert(common_memory_cosine_similarity({NAN}, {1.0f}) == 0.0f);

    assert(store.erase("fact-1", error));
    assert(!store.get("fact-1", error));

    common_memory_query empty_query;
    empty_query.text = "missing";
    empty_query.minimum_score = 0.01f;
    assert(store.search(empty_query, error).empty());

    store.close();
    return 0;
}
