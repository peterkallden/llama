#include "agent-data-store-cozo.h"

#include <nlohmann/json.hpp>

#include <cassert>
#include <cstdio>
#include <filesystem>

using json = nlohmann::ordered_json;

#define TEST_ASSERT(expr) do { \
    const bool test_result = static_cast<bool>(expr); \
    assert(test_result); \
    if (!test_result) return 1; \
} while (false)

int main() {
    namespace fs = std::filesystem;
    const auto root = fs::temp_directory_path() / "llama-agent-data-store-cozo-test";
    std::error_code ec;
    fs::remove_all(root, ec);
    fs::create_directories(root, ec);

    common_agent_cozo_data_store store;
    std::string error;
    if (!store.open((root / "data.cozo").string(), error)) {
        std::fprintf(stderr, "Cozo open failed: %s\n", error.c_str());
        return 1;
    }
    TEST_ASSERT(store.put_row("orders", "1", R"({"id":1,"customer_id":10,"region":"north","value":12})", error));
    TEST_ASSERT(store.put_row("orders", "2", R"({"id":2,"customer_id":11,"region":"south","value":8})", error));
    TEST_ASSERT(store.put_row("orders", "3", R"({"id":3,"customer_id":10,"region":"north","value":4})", error));
    TEST_ASSERT(store.put_row("customers", "10", R"({"customer_id":10,"name":"Ada"})", error));
    common_agent_dataset_descriptor orders_descriptor;
    orders_descriptor.ref = {"orders", "Orders", 3, 4, "resource://uploads/orders.csv", "tabular-dataset"};
    orders_descriptor.columns = {{"id", common_agent_dataset_column_type::integer, false},
        {"customer_id", common_agent_dataset_column_type::integer, false},
        {"region", common_agent_dataset_column_type::string, true},
        {"value", common_agent_dataset_column_type::integer, true}};
    orders_descriptor.import_processor_id = "test-importer";
    TEST_ASSERT(store.put_dataset_descriptor(orders_descriptor, error));
    common_agent_dataset_descriptor customers_descriptor = orders_descriptor;
    customers_descriptor.ref = {"customers", "Customers", 1, 2, "resource://uploads/customers.csv", "tabular-dataset"};
    customers_descriptor.columns = {{"customer_id", common_agent_dataset_column_type::integer, false},
        {"name", common_agent_dataset_column_type::string, true}};
    TEST_ASSERT(store.put_dataset_descriptor(customers_descriptor, error));

    std::string output;
    TEST_ASSERT(store.execute("data.query", R"({"dataset":"orders","order_by":[{"field":"value","direction":"desc"}],"max_scan_rows":2,"max_result_rows":1})", output, error));
    auto query = json::parse(output);
    TEST_ASSERT(query["scanned_rows"] == 2 && query["scan_truncated"] == true && query["row_count"] == 1 && query["result_truncated"] == true);
    TEST_ASSERT(query["rows"][0]["value"] == 12);

    TEST_ASSERT(store.execute("data.filter", R"({"dataset":"orders","conditions":[{"field":"region","operator":"=","value":"north"}],"max_scan_rows":10})", output, error));
    auto filtered = json::parse(output);
    TEST_ASSERT(filtered["row_count"] == 2 && filtered["rows"][0]["region"] == "north");

    TEST_ASSERT(store.execute("data.aggregate", R"({"dataset":"orders","group_by":["region"],"measures":[{"function":"count","column":"*","as":"count"},{"function":"sum","column":"value","as":"total"}]})", output, error));
    auto aggregate = json::parse(output);
    TEST_ASSERT(aggregate["row_count"] == 2 && aggregate["scan_mode"] == "native_bounded" && aggregate["scanned_rows"] == 3 && aggregate["scan_truncated"] == false);

    TEST_ASSERT(store.execute("data.aggregate", R"({"dataset":"orders","group_by":["region"],"measures":[{"function":"count","column":"*","as":"count"}],"max_scan_rows":2})", output, error));
    auto bounded_aggregate = json::parse(output);
    TEST_ASSERT(bounded_aggregate["scan_mode"] == "native_bounded" && bounded_aggregate["scanned_rows"] == 2 && bounded_aggregate["scan_truncated"] == true);

    TEST_ASSERT(store.execute("data.join", R"({"left":"orders","right":"customers","type":"inner","on":[{"left":"customer_id","right":"customer_id"}]})", output, error));
    auto inner_join = json::parse(output);
    TEST_ASSERT(inner_join["row_count"] == 2 && inner_join["scan_mode"] == "native_bounded" && inner_join["scanned_rows"] == 4 && inner_join["scan_truncated"] == false && inner_join["rows"][0].contains("name"));

    TEST_ASSERT(store.execute("data.join", R"({"left":"orders","right":"customers","type":"left","on":[{"left":"customer_id","right":"customer_id"}]})", output, error));
    auto join = json::parse(output);
    TEST_ASSERT(join["row_count"] == 3 && join["scan_mode"] == "native_bounded" && join["scanned_rows"] == 4 && join["scan_truncated"] == false && join["rows"][0].contains("name"));
    bool found_unmatched = false;
    for (const auto & row : join["rows"]) if (row.value("customer_id", 0) == 11) found_unmatched = !row.contains("name");
    TEST_ASSERT(found_unmatched);

    TEST_ASSERT(store.execute("data.join", R"({"left":"orders","right":"customers","type":"left","on":[{"left":"customer_id","right":"customer_id"}],"max_scan_rows":2})", output, error));
    auto bounded_join = json::parse(output);
    TEST_ASSERT(bounded_join["row_count"] == 2 && bounded_join["scanned_rows"] == 3 && bounded_join["scan_truncated"] == true);

    TEST_ASSERT(!store.execute("data.join", R"({"dataset":"orders","right":"customers","on":[]})", output, error));
    TEST_ASSERT(error.find("left") != std::string::npos);

    TEST_ASSERT(store.execute("data.filter", R"({"dataset":"orders","conditions":[{"field":"region","operator":"=","value":"north"}],"materialize":true,"result_dataset":"dataset://derived/north-orders"})", output, error));
    const auto materialized = json::parse(output);
    TEST_ASSERT(materialized["materialized"] == true && materialized["dataset"] == "dataset://derived/north-orders" && materialized["rows"] == 2);
    common_agent_dataset_descriptor derived;
    TEST_ASSERT(store.get_dataset_descriptor("dataset://derived/north-orders", derived, error));
    TEST_ASSERT(derived.lineage.parent_dataset_uris.size() == 1 && derived.lineage.parent_dataset_uris[0] == "orders" &&
        derived.lineage.operation == "data.filter" && derived.ref.source_resource_uri == "resource://uploads/orders.csv");
    TEST_ASSERT(!store.execute("data.filter", R"({"dataset":"orders","conditions":[],"max_scan_rows":1,"materialize":true,"result_dataset":"dataset://derived/truncated"})", output, error));
    TEST_ASSERT(error.find("truncated") != std::string::npos);

    store.close();
    fs::remove_all(root, ec);
    return 0;
}

#undef TEST_ASSERT
