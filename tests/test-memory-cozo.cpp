#include "memory/cozo/memory-cozo.h"

#include <cstdio>
#include <ctime>
#include <filesystem>
#include <string>

static common_memory_record make_record(
        const std::string & id,
        const std::string & content,
        std::vector<float> embedding,
        common_memory_kind kind = common_memory_kind::fact) {
    common_memory_record record;
    record.id = id;
    record.kind = kind;
    record.content = content;
    record.embedding = std::move(embedding);
    record.importance = 0.8f;
    record.confidence = 0.9f;
    record.created_at = 1710000000;
    record.accessed_at = 1710000100;
    record.access_count = 2;
    record.metadata["source"] = "cozo-test";
    return record;
}

#define CHECK(condition) do { \
    if (!(condition)) { \
        std::fprintf(stderr, "%s:%d: check failed: %s\\n", __FILE__, __LINE__, #condition); \
        return 1; \
    } \
} while (false)

int main() {
    namespace fs = std::filesystem;

    const fs::path db_dir = fs::temp_directory_path() / "llama-memory-cozo-test";
    const fs::path db_path = db_dir / "memory.db";

    std::error_code ec;
    fs::remove_all(db_dir, ec);
    fs::create_directories(db_dir, ec);

    std::string error;
    {
        common_memory_cozo_store store;
        CHECK(store.open(db_path.string(), error));
        CHECK(error.empty());

        const auto fact = make_record("fact-1", "zero budget package search", {1.0f, 0.0f});
        const auto episode = make_record("episode-1", "the earlier pass produced the fact", {0.0f, 1.0f}, common_memory_kind::episode);
        common_memory_record cli_like;
        cli_like.id = "cli-1";
        cli_like.kind = common_memory_kind::fact;
        cli_like.content = "cli style insertion";
        cli_like.embedding = {1.0f, 0.0f};
        cli_like.importance = 0.8f;
        cli_like.confidence = 0.9f;
        cli_like.created_at = std::time(nullptr);
        cli_like.accessed_at = cli_like.created_at;

        CHECK(store.put(fact, error));
        CHECK(store.put(episode, error));
        CHECK(store.put(cli_like, error));
        CHECK(store.relate("episode-1", "produced", "fact-1", 1.0f, error));

        auto got = store.get("fact-1", error);
        CHECK(got);
        CHECK(got->content == fact.content);
        CHECK(got->metadata["source"] == "cozo-test");

        common_memory_query text_query;
        text_query.text = "zero budget";
        text_query.limit = 4;
        auto text_hits = store.search(text_query, error);
        CHECK(error.empty());
        CHECK(!text_hits.empty());
        CHECK(text_hits[0].memory.id == "fact-1");

        store.close();
    }

    {
        common_memory_cozo_store reopened;
        CHECK(reopened.open(db_path.string(), error));
        CHECK(error.empty());

        auto got = reopened.get("episode-1", error);
        CHECK(got);
        CHECK(got->content == "the earlier pass produced the fact");
        CHECK(got->kind == common_memory_kind::episode);

        common_memory_query emb_query;
        emb_query.embedding = {0.0f, 1.0f};
        emb_query.limit = 1;
        auto emb_hits = reopened.search(emb_query, error);
        CHECK(error.empty());
        CHECK(emb_hits.size() == 1);
        CHECK(emb_hits[0].memory.id == "episode-1");
        CHECK(emb_hits[0].provenance == "CozoDB candidate scan with C++ scoring");

        CHECK(reopened.erase("fact-1", error));
        CHECK(!reopened.get("fact-1", error));

        reopened.close();
    }

    fs::remove_all(db_dir, ec);
    return 0;
}
