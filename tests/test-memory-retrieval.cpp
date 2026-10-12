#include "memory/memory-in-memory.h"
#include "memory/memory-retrieval.h"

#include <cassert>

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

    return 0;
}
