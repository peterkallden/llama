#include "agent-data-store-sqlite.h"

#include <filesystem>
#include <iostream>
#include <nlohmann/json.hpp>

using json = nlohmann::ordered_json;

#define TEST_ASSERT(expr) do { \
    const bool test_result = static_cast<bool>(expr); \
    if (!test_result) { std::cerr << "failed: " << #expr << " error=" << error << '\\n'; return 1; } \
} while (false)

int main() {
    const auto path = std::filesystem::temp_directory_path() / "llama-agent-data-sqlite-smoke.sqlite";
    std::error_code ignored;
    std::filesystem::remove(path, ignored);

    common_agent_dataset_descriptor descriptor;
    descriptor.ref.uri = "dataset://orders";
    descriptor.ref.name = "orders";
    descriptor.ref.source_resource_uri = "resource://turn/t/orders.json";
    descriptor.ref.source_representation = "openapi:json-array";
    descriptor.ref.source_provider = "sales-api";
    descriptor.ref.source_operation = "listOrders";
    descriptor.ref.source_request_json = R"({"limit":2})";
    descriptor.ref.retrieved_at = 123;
    descriptor.ref.content_hash = "sha256:abc";
    descriptor.origin.header_mode = common_agent_table_header_mode::explicit_;
    descriptor.import_processor_id = "openapi-json-array";
    descriptor.columns.push_back({"amount", common_agent_dataset_column_type::integer, false});
    descriptor.ref.column_count = 1;

    std::string error;
    common_agent_sqlite_data_store store;
    TEST_ASSERT(store.open(path.string(), error));
    TEST_ASSERT(store.put_dataset_descriptor(descriptor, error));
    TEST_ASSERT(store.put_row(descriptor.ref.uri, "1", json({{"amount", 10}}).dump(), error));
    TEST_ASSERT(store.put_row(descriptor.ref.uri, "2", json({{"amount", 20}}).dump(), error));

    common_agent_dataset_descriptor found;
    TEST_ASSERT(store.find_dataset_by_name(" orders ", found, error));
    TEST_ASSERT(found.ref.uri == descriptor.ref.uri);
    TEST_ASSERT(found.ref.source_provider == descriptor.ref.source_provider);
    TEST_ASSERT(found.ref.source_operation == descriptor.ref.source_operation);
    TEST_ASSERT(found.ref.source_request_json == descriptor.ref.source_request_json);
    TEST_ASSERT(found.ref.retrieved_at == descriptor.ref.retrieved_at);
    TEST_ASSERT(found.ref.content_hash == descriptor.ref.content_hash);
    TEST_ASSERT(found.origin.header_mode == descriptor.origin.header_mode);

    std::string result;
    TEST_ASSERT(store.execute("data.query", json({
        {"dataset", descriptor.ref.uri},
        {"where", json::array({json({{"field", "amount"}, {"operator", ">"}, {"value", 15}})})},
        {"max_scan_rows", 10}, {"max_result_rows", 10}
    }).dump(), result, error));
    const auto parsed = json::parse(result, nullptr, false);
    TEST_ASSERT(parsed.is_object());
    TEST_ASSERT(parsed["rows"].size() == 1);
    TEST_ASSERT(parsed["rows"][0]["amount"] == 20);

    store.close();
    std::filesystem::remove(path, ignored);
    return 0;
}
