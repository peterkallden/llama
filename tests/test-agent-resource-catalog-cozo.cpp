#include "tools/agent/resource/agent-resource-catalog.h"

#include <chrono>
#include <filesystem>
#include <iostream>
#include <stdexcept>

#ifdef LLAMA_MEMORY_USE_COZO
#include <nlohmann/json.hpp>
extern "C" {
#include <cozo_c.h>
}

namespace {
void require(bool condition, const std::string & message) {
    if (!condition) throw std::runtime_error(message);
}
void catalog_script(const std::string & path, const char * script) {
    int32_t db = -1;
    char * opened = cozo_open_db("sqlite", path.c_str(), "{}", &db);
    if (opened) { const std::string message(opened); cozo_free_str(opened); throw std::runtime_error(message); }
    char * changed = cozo_run_query(db, script, "{}", false);
    const auto result = nlohmann::ordered_json::parse(changed, nullptr, false);
    cozo_free_str(changed);
    cozo_close_db(db);
    require(result.is_object() && result.value("ok", false), "catalog fixture setup failed");
}
void check_lineage(const agent_resource_descriptor & value) {
    const auto & lineage = value.lineage;
    require(lineage.parent_uri == "agent-resource://resource/resource-1", "parent reference was lost");
    require(lineage.chunk_index == 2 && lineage.chunk_count == 4, "chunk identity was lost");
    require(lineage.byte_offset == 2048 && lineage.byte_length == 1024 && lineage.overlap_bytes == 32,
            "source positions were lost");
    require(lineage.derivation == "contract.derived.v1", "derivation was lost");
    require(value.metadata.source_request_json == R"({"revision":7})", "provenance was lost");
}
}
#endif

int main() {
#ifndef LLAMA_MEMORY_USE_COZO
    return 77;
#else
    namespace fs = std::filesystem;
    const auto directory = fs::temp_directory_path() / ("agent-resource-lineage-" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(directory);
    const auto path = (directory / "resources.cozo").string();
    try {
        std::string error;
        agent_resource_descriptor legacy;
        legacy.uri = "agent-resource://resource/resource-1";
        legacy.resource_id = "resource-1";
        legacy.mime_type = "text/plain";
        legacy.namespace_id = "contract";
        legacy.session_id = "session";
        legacy.metadata.source_operation = "contract.original";
        {
            agent_cozo_resource_catalog catalog;
            require(catalog.open(path, error), error);
            require(catalog.put_descriptor(legacy, error), error);
        }
        // Reproduce a catalog created before lineage had durable storage.
        catalog_script(path, "::remove agent_resource_lineage");

        auto derived = legacy;
        derived.uri = "agent-resource://resource/resource-2";
        derived.resource_id = "resource-2";
        derived.metadata.source_request_json = R"({"revision":7})";
        derived.lineage = {legacy.uri, 2, 4, 2048, 1024, 32, "contract.derived.v1"};
        {
            agent_cozo_resource_catalog catalog;
            require(catalog.open(path, error), error);
            agent_resource_descriptor restored;
            require(catalog.find_descriptor(legacy.uri, restored, error), error);
            require(restored.lineage.parent_uri.empty(), "migration invented legacy lineage");
            require(restored.metadata.source_operation == "contract.original", "migration changed provenance");
            require(catalog.put_descriptor(derived, error), error);
            require(catalog.find_descriptor(derived.uri, restored, error), error);
            check_lineage(restored);
        }
        {
            agent_cozo_resource_catalog catalog;
            require(catalog.open(path, error), error);
            agent_resource_descriptor restored;
            require(catalog.find_descriptor(derived.uri, restored, error), error);
            check_lineage(restored);
            std::vector<agent_resource_descriptor> listed;
            require(catalog.list_descriptors(listed, error), error);
            bool found = false;
            for (const auto & item : listed) if (item.uri == derived.uri) { check_lineage(item); found = true; }
            require(found, "listed descriptor was lost");
            auto invalid = derived;
            invalid.lineage.chunk_count = 0;
            require(!catalog.put_descriptor(invalid, error), "invalid lineage was accepted");
            require(catalog.find_descriptor(derived.uri, restored, error), error);
            check_lineage(restored);
        }
        // Fail the last metadata write and ensure no descriptor is partly published.
        catalog_script(path, "::remove agent_resource_lineage");
        catalog_script(path, "?[uri, lineage_json] <- [['fixture', 0]] "
            ":create agent_resource_lineage { uri: String => lineage_json: Int }");
        {
            agent_cozo_resource_catalog catalog;
            require(catalog.open(path, error), error);
            auto rejected = derived;
            rejected.uri = "agent-resource://resource/resource-3";
            rejected.resource_id = "resource-3";
            require(!catalog.put_descriptor(rejected, error), "metadata storage failure was ignored");
            agent_resource_descriptor unexpected;
            require(!catalog.find_descriptor(rejected.uri, unexpected, error), "partial descriptor was published");
        }
        fs::remove_all(directory);
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        fs::remove_all(directory);
        return 1;
    }
#endif
}
