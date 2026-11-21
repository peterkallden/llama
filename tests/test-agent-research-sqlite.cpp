#include "tools/agent/data/agent-data-store-sqlite.h"
#include <filesystem>
#include <iostream>

#define REQUIRE(expr) do { if (!(expr)) { std::cerr << __LINE__ << ": " << error << '\n'; return 1; } } while (0)

int main() {
    const auto path = std::filesystem::temp_directory_path() / "llama-agent-research-sqlite-contract.db";
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
    std::string error;
    common_agent_scope scope; scope.namespace_id = "ns"; scope.session_id = "session"; scope.project_id = "project";
    common_agent_research_draft_ref draft;
    draft.research_id = "research"; draft.draft_id = "draft"; draft.owner_session_id = scope.session_id;
    draft.draft_revision = 1; draft.payload_uri = "agent-resource://resource/1"; draft.payload_fingerprint = "hash";
    common_agent_sqlite_data_store store;
    REQUIRE(store.open(path.string(), error));
    common_agent_research_draft_ref saved; bool already = false;
    REQUIRE(store.save_research_draft(scope, draft, 0, saved, already, error) && !already);
    REQUIRE(store.save_research_draft(scope, draft, 0, saved, already, error) && already);
    auto conflicting = draft; conflicting.payload_fingerprint = "other";
    REQUIRE(!store.save_research_draft(scope, conflicting, 0, saved, already, error));
    error.clear(); store.close();
    REQUIRE(store.open(path.string(), error));
    REQUIRE(store.get_research_draft("draft", scope, saved, error) && saved.payload_uri == draft.payload_uri);
    auto other = scope; other.session_id = "other";
    REQUIRE(!store.get_research_draft("draft", other, saved, error));
    error.clear();
    common_agent_research_revision_ref revision;
    revision.research_id = "research"; revision.revision = 1; revision.payload_uri = draft.payload_uri;
    revision.payload_fingerprint = "hash"; revision.publication_id = "pub";
    common_agent_research_revision_ref published;
    REQUIRE(store.publish_research_revision(scope, revision, 0, published, already, error) && !already);
    REQUIRE(store.publish_research_revision(scope, revision, 0, published, already, error) && already);
    store.close(); REQUIRE(store.open(path.string(), error));
    REQUIRE(store.get_research_revision("research", 0, scope, published, error) && published.revision == 1);
    other.project_id = "other-project"; error.clear();
    REQUIRE(!store.get_research_revision("research", 0, other, published, error));
    store.close(); std::filesystem::remove(path, ignored);
    return 0;
}
