#include "examples/llama.android/lib/src/main/cpp/agent_android_resource_catalog.h"
#include "tools/agent/resource/agent-resource-store.h"
#include <filesystem>
#include <iostream>

#define REQUIRE(expr) do { if (!(expr)) { std::cerr << __LINE__ << ": " << error << '\n'; return 1; } } while (0)

int main() {
    const auto root = std::filesystem::temp_directory_path() / "llama-agent-android-catalog-contract";
    std::error_code ec; std::filesystem::remove_all(root, ec); std::filesystem::create_directories(root, ec);
    std::string error, uri;
    {
        auto catalog = std::make_unique<android_resource_catalog>();
        REQUIRE(catalog->open((root / "resources.sqlite").string(), error));
        auto blobs = std::make_shared<agent_filesystem_blob_store>((root / "blobs").string());
        agent_catalogued_resource_store store(blobs, std::move(catalog));
        agent_resource_put_request request; request.name = "test.txt"; request.mime_type = "text/plain";
        request.text = "A saved resource"; request.scope = common_runtime_resource_scope::session;
        request.namespace_id = "ns"; request.session_id = "session"; request.metadata.purpose = "research";
        agent_resource_descriptor result; REQUIRE(store.put_text(request, result, error)); uri = result.uri;
    }
    {
        auto catalog = std::make_unique<android_resource_catalog>();
        REQUIRE(catalog->open((root / "resources.sqlite").string(), error));
        auto blobs = std::make_shared<agent_filesystem_blob_store>((root / "blobs").string());
        agent_catalogued_resource_store store(blobs, std::move(catalog));
        agent_resource_read_authority authority; authority.namespace_id = "ns"; authority.session_id = "session";
        std::string text; REQUIRE(store.read_text(uri, authority, 1024, text, error) && text == "A saved resource");
        agent_resource_descriptor descriptor; REQUIRE(store.stat(uri, authority, descriptor, error));
        REQUIRE(descriptor.metadata.purpose == "research");
        authority.session_id = "other"; error.clear(); REQUIRE(!store.read_text(uri, authority, 1024, text, error));
    }
    std::filesystem::remove_all(root, ec);
    return 0;
}
