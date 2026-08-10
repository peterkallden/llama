#include "tools/agent/resource/agent-resource-store.h"
#include "plan/plan-types.h"

#include <cstdio>
#include <filesystem>
#include <memory>
#include <string>

int main() {
    std::string error;
    auto blob_store = std::make_shared<agent_in_memory_blob_store>();
    agent_catalogued_resource_store store(
        blob_store,
        std::make_unique<agent_in_memory_resource_catalog>());

    agent_resource_descriptor first;
    if (!store.put_text({
            "search-results.json",
            "Full repository search payload",
            "application/json",
            R"({"matches":[1,2,3]})",
            common_runtime_resource_scope::turn,
            "tenant-a",
            "session-1",
            "project-x",
            "turn-1",
            "tool-1",
            "native",
            "repository.search",
            0,
            0,
            {
                "Preserve the full repository search output for later plan/tool reuse.",
                "Three repository search matches in JSON form.",
                "Read the resource when a later step needs the full match list instead of an inline summary.",
                "Search previews are partial lines only.",
                {"repository.search", "matches"},
                {"tool-adapters.cpp"},
            },
            {
                "agent-resource://resource/original",
                1,
                3,
                8,
                24,
                2,
                "host.chunk.text.v1",
            },
        }, first, error)) {
        std::fprintf(stderr, "put_text failed: %s\n", error.c_str());
        return 1;
    }

    if (!blob_store->exists_sha256(first.sha256)) {
        std::fprintf(stderr, "blob store did not retain sha256 %s\n", first.sha256.c_str());
        return 1;
    }

    agent_resource_descriptor second;
    if (!store.put_text({
            "search-results-copy.json",
            "Duplicate content",
            "application/json",
            R"({"matches":[1,2,3]})",
            common_runtime_resource_scope::session,
            "tenant-a",
            "session-1",
            "project-x",
            "turn-2",
            "tool-2",
            "native",
            "repository.search",
        }, second, error)) {
        std::fprintf(stderr, "second put_text failed: %s\n", error.c_str());
        return 1;
    }

    if (first.sha256 != second.sha256) {
        std::fprintf(stderr, "duplicate payload did not deduplicate by sha256\n");
        return 1;
    }

    agent_resource_read_authority turn_authority;
    turn_authority.namespace_id = "tenant-a";
    turn_authority.session_id = "session-1";
    turn_authority.project_id = "project-x";
    turn_authority.turn_id = "turn-1";
    turn_authority.now = first.created_at;

    std::string content;
    if (!store.read_text(first.uri, turn_authority, 1024, content, error)) {
        std::fprintf(stderr, "read_text failed: %s\n", error.c_str());
        return 1;
    }
    if (content.find("\"matches\"") == std::string::npos) {
        std::fprintf(stderr, "read_text returned unexpected payload: %s\n", content.c_str());
        return 1;
    }

    agent_resource_descriptor statted;
    if (!store.stat(first.uri, turn_authority, statted, error)) {
        std::fprintf(stderr, "stat failed: %s\n", error.c_str());
        return 1;
    }
    if (statted.source_tool != "repository.search") {
        std::fprintf(stderr, "stat returned unexpected source tool: %s\n", statted.source_tool.c_str());
        return 1;
    }
    if (statted.metadata.content_summary != "Three repository search matches in JSON form.") {
        std::fprintf(stderr, "stat returned unexpected metadata content summary: %s\n", statted.metadata.content_summary.c_str());
        return 1;
    }
    if (statted.lineage.parent_uri != "agent-resource://resource/original" ||
            statted.lineage.chunk_index != 1 || statted.lineage.byte_offset != 8) {
        std::fprintf(stderr, "stat returned unexpected resource lineage\n");
        return 1;
    }

    if (!store.read_text_range(first.uri, turn_authority, 2, 9, content, error) ||
            content != "matches\":") {
        std::fprintf(stderr, "read_text_range returned unexpected payload: %s (%s)\n", content.c_str(), error.c_str());
        return 1;
    }
    if (store.read_text_range(first.uri, turn_authority, first.size_bytes + 1, 9, content, error)) {
        std::fprintf(stderr, "out-of-bounds resource range unexpectedly succeeded\n");
        return 1;
    }

    common_plan_observation chunk_one{
        "chunk-1", "resource_chunk", "First bounded observation", 1.0f, {}, {first}, 0};
    chunk_one.resource_refs.front().lineage.chunk_index = 0;
    chunk_one.resource_refs.front().lineage.chunk_count = 2;
    common_plan_observation chunk_two = chunk_one;
    chunk_two.id = "chunk-2";
    chunk_two.resource_refs.front().uri = "agent-resource://resource/chunk-2";
    chunk_two.resource_refs.front().lineage.chunk_index = 1;
    std::vector<common_plan_observation> chunk_observations{chunk_one, chunk_two};
    if (!common_plan_chunk_observations_valid(chunk_observations, error)) {
        std::fprintf(stderr, "valid chunk observations were rejected: %s\n", error.c_str());
        return 1;
    }
    common_plan_chunk_synthesis_input synthesis;
    if (!common_plan_chunk_synthesis_from_observations(
                chunk_observations, first.lineage.parent_uri, synthesis, error) ||
            synthesis.status != common_plan_chunk_synthesis_status::complete ||
            synthesis.summaries.size() != 2 || synthesis.completed_chunk_indexes[0] != 0) {
        std::fprintf(stderr, "complete chunk synthesis projection failed: %s\n", error.c_str());
        return 1;
    }
    auto incomplete_observations = chunk_observations;
    incomplete_observations.pop_back();
    if (!common_plan_chunk_synthesis_from_observations(
                incomplete_observations, first.lineage.parent_uri, synthesis, error) ||
            synthesis.status != common_plan_chunk_synthesis_status::incomplete ||
            synthesis.missing_chunk_indexes.empty()) {
        std::fprintf(stderr, "incomplete chunk synthesis projection failed: %s\n", error.c_str());
        return 1;
    }
    chunk_observations.push_back(chunk_two);
    if (common_plan_chunk_observations_valid(chunk_observations, error) ||
            error != "resource_chunk observations must not contain duplicate chunk indexes") {
        std::fprintf(stderr, "duplicate chunk observation was not rejected\n");
        return 1;
    }

    agent_resource_read_authority wrong_turn_authority = turn_authority;
    wrong_turn_authority.turn_id = "turn-9";
    if (store.read_text(first.uri, wrong_turn_authority, 1024, content, error)) {
        std::fprintf(stderr, "turn-scoped resource unexpectedly allowed mismatched turn authority\n");
        return 1;
    }

    agent_resource_read_authority session_authority = turn_authority;
    session_authority.turn_id = "turn-99";
    if (!store.read_text(second.uri, session_authority, 1024, content, error)) {
        std::fprintf(stderr, "session-scoped resource should allow same-session authority: %s\n", error.c_str());
        return 1;
    }

    if (store.read_text(second.uri, session_authority, 4, content, error)) {
        std::fprintf(stderr, "resource read limit was not enforced\n");
        return 1;
    }

    agent_resource_put_request binary_request;
    binary_request.name = "image.bin";
    binary_request.description = "Opaque binary resource";
    binary_request.mime_type = "application/octet-stream";
    binary_request.scope = common_runtime_resource_scope::session;
    binary_request.namespace_id = "tenant-a";
    binary_request.session_id = "session-1";
    binary_request.project_id = "project-x";
    binary_request.turn_id = "turn-2";
    binary_request.bytes = "A";
    binary_request.bytes.push_back('\0');
    binary_request.bytes.push_back('B');
    binary_request.bytes.push_back(static_cast<char>(0x7f));
    binary_request.bytes.push_back('C');
    agent_resource_descriptor binary_descriptor;
    if (!store.put_bytes(binary_request, binary_descriptor, error)) {
        std::fprintf(stderr, "put_bytes failed: %s\n", error.c_str());
        return 1;
    }
    std::string binary_content;
    if (!store.read_bytes(binary_descriptor.uri, session_authority, 16, binary_content, error) ||
            binary_content != binary_request.bytes ||
            binary_descriptor.size_bytes != binary_request.bytes.size()) {
        std::fprintf(stderr, "read_bytes returned unexpected binary payload: %s\n", error.c_str());
        return 1;
    }
    if (!store.read_bytes_range(binary_descriptor.uri, session_authority, 1, 3, binary_content, error) ||
            binary_content != std::string("\0B", 2) + static_cast<char>(0x7f)) {
        std::fprintf(stderr, "read_bytes_range returned unexpected binary payload\n");
        return 1;
    }

    const std::filesystem::path fs_root =
        std::filesystem::temp_directory_path() / "llama-agent-resource-store-smoke";
    std::filesystem::create_directories(fs_root);
    agent_resource_store_config fs_config;
    fs_config.blob_backend = "auto";
    fs_config.blob_root = fs_root.string();
    fs_config.metadata_backend = "in-memory";

    std::shared_ptr<agent_blob_store> fs_blob_store = make_agent_blob_store(fs_config, error);
    if (!fs_blob_store) {
        std::fprintf(stderr, "filesystem blob store factory failed: %s\n", error.c_str());
        return 1;
    }

    agent_blob_descriptor fs_blob;
    std::string filesystem_bytes("filesystem", 10);
    filesystem_bytes.push_back('\0');
    filesystem_bytes.append("blob", 4);
    filesystem_bytes.push_back(static_cast<char>(0xff));
    if (!fs_blob_store->put_bytes(filesystem_bytes, fs_blob, error)) {
        std::fprintf(stderr, "filesystem blob write failed: %s\n", error.c_str());
        return 1;
    }
    if (!fs_blob_store->exists_sha256(fs_blob.sha256)) {
        std::fprintf(stderr, "filesystem blob store did not retain expected blob\n");
        return 1;
    }
    std::string filesystem_read;
    if (!fs_blob_store->get_bytes(fs_blob.sha256, 32, filesystem_read, error) ||
            filesystem_read != filesystem_bytes) {
        std::fprintf(stderr, "filesystem blob store did not preserve opaque bytes\n");
        return 1;
    }

#ifdef LLAMA_MEMORY_USE_COZO
    const std::filesystem::path cozo_root =
        std::filesystem::temp_directory_path() / "llama-agent-resource-store-cozo-smoke";
    std::filesystem::remove_all(cozo_root);
    std::filesystem::create_directories(cozo_root);
    const std::filesystem::path cozo_db = cozo_root / "resource-meta.cozo";
    agent_resource_store_config cozo_config;
    cozo_config.blob_backend = "auto";
    cozo_config.metadata_backend = "cozo";
    cozo_config.metadata_db = cozo_db.string();

    std::unique_ptr<agent_resource_store> cozo_store = make_agent_resource_store(cozo_config, error);
    if (!cozo_store) {
        std::fprintf(stderr, "cozo resource store factory failed: %s\n", error.c_str());
        return 1;
    }

    agent_resource_descriptor cozo_descriptor;
    agent_resource_put_request cozo_request;
    cozo_request.name = "cozo-results.bin";
    cozo_request.description = "Cozo-backed metadata entry";
    cozo_request.mime_type = "application/octet-stream";
    cozo_request.scope = common_runtime_resource_scope::project;
    cozo_request.namespace_id = "tenant-a";
    cozo_request.session_id = "session-1";
    cozo_request.project_id = "project-x";
    cozo_request.turn_id = "turn-3";
    cozo_request.tool_call_id = "tool-3";
    cozo_request.source_provider = "native";
    cozo_request.source_tool = "repository.search";
    cozo_request.metadata.processing_cache_key = "resource-processing-cache-v1;smoke";
    cozo_request.metadata.declared_language = "sv";
    cozo_request.metadata.resolved_language = "swe";
    cozo_request.metadata.language_confidence = 0.97;
    cozo_request.metadata.language_source = "user";
    cozo_request.bytes = std::string("cozo\0bytes", 10);
    if (!cozo_store->put_bytes(cozo_request, cozo_descriptor, error)) {
        std::fprintf(stderr, "cozo put_bytes failed: %s\n", error.c_str());
        return 1;
    }

    agent_resource_read_authority project_authority = turn_authority;
    project_authority.project_id = "project-x";
    if (!cozo_store->read_bytes(cozo_descriptor.uri, project_authority, 1024, content, error) ||
            content != cozo_request.bytes) {
        std::fprintf(stderr, "cozo read_bytes failed: %s\n", error.c_str());
        return 1;
    }

    cozo_store.reset();
    cozo_store = make_agent_resource_store(cozo_config, error);
    if (!cozo_store) {
        std::fprintf(stderr, "cozo resource store reopen failed: %s\n", error.c_str());
        return 1;
    }
    agent_resource_descriptor reopened_cozo_descriptor;
    if (!cozo_store->stat(cozo_descriptor.uri, project_authority, reopened_cozo_descriptor, error) ||
            reopened_cozo_descriptor.metadata.processing_cache_key != cozo_request.metadata.processing_cache_key ||
            reopened_cozo_descriptor.metadata.declared_language != "sv" ||
            reopened_cozo_descriptor.metadata.resolved_language != "swe" ||
            reopened_cozo_descriptor.metadata.language_confidence != 0.97 ||
            reopened_cozo_descriptor.metadata.language_source != "user") {
        std::fprintf(stderr, "cozo resource metadata did not preserve processing/language metadata: %s\n", error.c_str());
        return 1;
    }
#else
    if (validate_agent_resource_store_config({
            "auto",
            "",
            "cozo",
            "resource-meta.cozo",
        }, error)) {
        std::fprintf(stderr, "cozo metadata backend unexpectedly validated without Cozo support\n");
        return 1;
    }
#endif

    std::printf("resource_uri=%s\n", first.uri.c_str());
    std::printf("resource_sha256=%s\n", first.sha256.c_str());
    std::printf("resource_scope=%s\n", common_runtime_resource_scope_name(first.scope));
    std::printf("fs_blob_sha256=%s\n", fs_blob.sha256.c_str());
    return 0;
}
