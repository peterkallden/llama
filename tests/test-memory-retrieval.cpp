#ifdef NDEBUG
#undef NDEBUG
#endif

#include "memory/memory-in-memory.h"
#include "memory/memory-retrieval.h"

#include <cassert>
#include <ctime>

static common_memory_record make_record(const std::string & id, const std::string & content, float importance) {
    common_memory_record record;
    record.id = id;
    record.kind = common_memory_kind::fact;
    record.content = content;
    record.importance = importance;
    record.confidence = 1.0f;
    return record;
}

int main() {
    std::string error;
    common_memory_in_memory_store store;
    assert(store.open("", error));
    assert(store.put(make_record("b", "same query term", 0.5f), error));
    assert(store.put(make_record("a", "same query term", 0.5f), error));
    assert(store.put(make_record("c", "different", 1.0f), error));

    common_memory_retrieval retrieval(store);
    common_memory_query query;
    query.text = "same query";
    query.limit = 2;
    query.minimum_score = 0.01f;

    auto hits = retrieval.retrieve(query, error);
    assert(hits.size() == 2);
    assert(hits[0].memory.id == "a");
    assert(hits[1].memory.id == "b");
    assert(hits[0].final_score >= hits[1].final_score);

    query.limit = 1;
    hits = retrieval.retrieve(query, error);
    assert(hits.size() == 1);

    const int64_t now = static_cast<int64_t>(std::time(nullptr));
    auto prior = make_record("temporal-old", "temporal project preference", 0.8f);
    prior.metadata["agent.temporal.v1.observed_at"] = std::to_string(now - 200);
    prior.metadata["agent.temporal.v1.valid_from"] = std::to_string(now - 200);
    prior.metadata["agent.temporal.v1.valid_to"] = std::to_string(now - 1);
    auto current = make_record("temporal-current", "temporal project preference", 0.8f);
    current.metadata["agent.temporal.v1.observed_at"] = std::to_string(now);
    current.metadata["agent.temporal.v1.valid_from"] = std::to_string(now - 1);
    auto future = make_record("temporal-future", "temporal project preference", 0.8f);
    future.metadata["agent.temporal.v1.valid_from"] = std::to_string(now + 3600);
    assert(store.put(prior, error));
    assert(store.put(current, error));
    assert(store.put(future, error));
    assert(store.relate("temporal-current", "supersedes", "temporal-old", 1.0f, error));

    common_memory_query temporal_query;
    temporal_query.text = "temporal project preference";
    temporal_query.limit = 8;
    temporal_query.minimum_score = 0.01f;
    temporal_query.as_of = now + 2;
    auto current_hits = retrieval.retrieve(temporal_query, error);
    assert(error.empty());
    assert(current_hits.size() == 1 && current_hits[0].memory.id == "temporal-current");
    temporal_query.include_history = true;
    auto history_hits = retrieval.retrieve(temporal_query, error);
    assert(error.empty());
    bool saw_old = false, saw_current = false, saw_future = false;
    for (const auto & hit : history_hits) {
        saw_old |= hit.memory.id == "temporal-old";
        saw_current |= hit.memory.id == "temporal-current";
        saw_future |= hit.memory.id == "temporal-future";
    }
    assert(saw_old && saw_current && !saw_future);

    auto invalid = make_record("invalid-interval", "invalid temporal interval", 0.5f);
    invalid.metadata["agent.temporal.v1.valid_from"] = "20";
    invalid.metadata["agent.temporal.v1.valid_to"] = "10";
    assert(!store.put(invalid, error));

    return 0;
}
