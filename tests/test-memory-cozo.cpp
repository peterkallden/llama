#include "memory/cozo/memory-cozo.h"

#include <cstdio>
#include <ctime>
#include <filesystem>
#include <string>

extern "C" {
#include <cozo_c.h>
}

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

static bool run_legacy_query(int32_t db_id, const char * script) {
    char * result = cozo_run_query(db_id, script, "{}", false);
    if (result == nullptr) {
        return false;
    }
    cozo_free_str(result);
    return true;
}

static bool create_legacy_database(const std::string & path) {
    int32_t db_id = -1;
    char * open_error = cozo_open_db("sqlite", path.c_str(), "{}", &db_id);
    if (open_error != nullptr) {
        cozo_free_str(open_error);
        return false;
    }
    const bool ok =
        run_legacy_query(db_id, R"COZO(
            ?[id, kind, content, summary, embedding, importance, confidence, created_at, accessed_at, access_count, metadata_json] <-
                [['__schema_probe__', 'fact', '', '', [], 0.0, 0.0, 0, 0, 0, '{}']]
            :create memory {
                id: String => kind: String, content: String, summary: String, embedding: [Float], importance: Float, confidence: Float,
                created_at: Int, accessed_at: Int, access_count: Int, metadata_json: String
            }
        )COZO") &&
        run_legacy_query(db_id, R"COZO(
            ?[from, relation, to, weight, created_at] <- [['__schema_probe__', 'related', '__schema_probe__', 0.0, 0]]
            :create memory_edge { from: String, relation: String, to: String => weight: Float, created_at: Int }
        )COZO") &&
        run_legacy_query(db_id, R"COZO(
            ?[id, kind, content, summary, embedding, importance, confidence, created_at, accessed_at, access_count, metadata_json] <-
                [['legacy-1', 'fact', 'legacy scoped migration record', '', [1.0, 0.0], 0.8, 0.9, 1710000000, 1710000000, 0, '{}']]
            :put memory { id => kind, content, summary, embedding, importance, confidence, created_at, accessed_at, access_count, metadata_json }
        )COZO");
    cozo_close_db(db_id);
    return ok;
}

int main() {
    namespace fs = std::filesystem;

    const fs::path db_dir = fs::temp_directory_path() / "llama-memory-cozo-test";
    const fs::path db_path = db_dir / "memory.db";

    std::error_code ec;
    fs::remove_all(db_dir, ec);
    fs::create_directories(db_dir, ec);
    CHECK(create_legacy_database(db_path.string()));

    std::string error;
    {
        common_memory_cozo_store store;
        CHECK(store.open(db_path.string(), error));
        CHECK(error.empty());

        auto migrated = store.get("legacy-1", error);
        CHECK(migrated);
        CHECK(migrated->scope == common_memory_scope::session);
        CHECK(migrated->namespace_id == "local");
        CHECK(migrated->session_id == "default");
        common_memory_query migrated_query;
        migrated_query.text = "legacy scoped";
        migrated_query.limit = 1;
        auto migrated_hits = store.search(migrated_query, error);
        CHECK(error.empty());
        CHECK(migrated_hits.size() == 1);
        CHECK(migrated_hits[0].memory.id == "legacy-1");

        const auto fact = make_record("fact-1", "zero budget package search", {1.0f, 0.0f});
        const auto episode = make_record("episode-1", "the earlier pass produced the fact", {0.0f, 1.0f}, common_memory_kind::episode);
        auto other_session = make_record("session-b", "zero budget from another session", {1.0f, 0.0f});
        other_session.session_id = "session-b";
        auto global = make_record("global-1", "globally opt-in record", {0.0f, 1.0f});
        global.scope = common_memory_scope::global;
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
        CHECK(store.put(other_session, error));
        CHECK(store.put(global, error));
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

        common_memory_query global_query;
        global_query.text = "globally";
        global_query.scope = common_memory_scope::global;
        global_query.limit = 4;
        CHECK(store.search(global_query, error).empty());
        global_query.global_opt_in = true;
        auto global_hits = store.search(global_query, error);
        CHECK(error.empty());
        CHECK(global_hits.size() == 1);
        CHECK(global_hits[0].memory.id == "global-1");

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

        auto scoped = reopened.get("session-b", error);
        CHECK(scoped);
        CHECK(scoped->scope == common_memory_scope::session);
        CHECK(scoped->session_id == "session-b");
        auto reopened_legacy = reopened.get("legacy-1", error);
        CHECK(reopened_legacy);
        CHECK(reopened_legacy->scope == common_memory_scope::session);

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
