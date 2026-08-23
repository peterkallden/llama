#include "memory/sqlite/memory-sqlite.h"

#include <cassert>
#include <filesystem>
#include <iostream>

int main() {
    const auto path = std::filesystem::temp_directory_path() / "llama-agent-memory-sqlite-smoke.sqlite";
    std::error_code ignored;
    std::filesystem::remove(path, ignored);

    common_memory_record record;
    record.id = "memory-1";
    record.kind = common_memory_kind::fact;
    record.content = "SQLite persistence keeps the agent memory available";
    record.summary = "sqlite persistence";
    record.embedding = {1.0f, 0.0f};
    record.scope = common_memory_scope::session;
    record.namespace_id = "smoke";
    record.session_id = "session-1";
    record.metadata["source"] = "ctest";

    std::string error;
    {
        common_memory_sqlite_store store;
        assert(store.open(path.string(), error));
        assert(store.put(record, error));
        const auto loaded = store.get(record.id, error);
        assert(loaded.has_value());
        assert(loaded->content == record.content);
        assert(loaded->metadata.at("source") == "ctest");

        common_memory_query query;
        query.text = "persistence";
        query.scope = common_memory_scope::session;
        query.namespace_id = "smoke";
        query.session_id = "session-1";
        query.limit = 4;
        const auto hits = store.search(query, error);
        assert(hits.size() == 1);
        assert(hits.front().memory.id == record.id);

        common_memory_record other = record;
        other.id = "memory-2";
        other.content = "A second record";
        assert(store.put(other, error));
        assert(store.relate(record.id, "supports", other.id, 0.75f, error));
    }

    {
        common_memory_sqlite_store store;
        assert(store.open(path.string(), error));
        const auto loaded = store.get(record.id, error);
        assert(loaded.has_value());
        assert(loaded->embedding == record.embedding);
        assert(store.erase(record.id, error));
        assert(!store.get(record.id, error).has_value());
    }

    std::filesystem::remove(path, ignored);
    std::cout << "SQLite memory persistence smoke passed\n";
    return 0;
}
