#include "agent-data-store-cozo.h"

#include <nlohmann/json.hpp>

#include <cassert>
#include <cmath>
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
    TEST_ASSERT(store.put_row("dataset://local/orders", "1", R"({"id":1,"customer_id":10,"region":"north","value":12})", error));
    TEST_ASSERT(store.put_row("dataset://local/orders", "2", R"({"id":2,"customer_id":11,"region":"south","value":8})", error));
    TEST_ASSERT(store.put_row("dataset://local/orders", "3", R"({"id":3,"customer_id":10,"region":"north","value":4})", error));
    TEST_ASSERT(store.put_row("dataset://local/metrics", "1", R"({"region":"all","value":10})", error));
    TEST_ASSERT(store.put_row("dataset://local/metrics", "2", R"({"region":"all","value":11})", error));
    TEST_ASSERT(store.put_row("dataset://local/metrics", "3", R"({"region":"all","value":12})", error));
    TEST_ASSERT(store.put_row("dataset://local/metrics", "4", R"({"region":"all","value":13})", error));
    TEST_ASSERT(store.put_row("dataset://local/metrics", "5", R"({"region":"all","value":100})", error));
    TEST_ASSERT(store.put_row("dataset://local/customers", "10", R"({"customer_id":10,"name":"Ada"})", error));
    common_agent_dataset_descriptor orders_descriptor;
    orders_descriptor.ref = {"dataset://local/orders", "Orders", 3, 4, "resource://uploads/orders.csv", "tabular-dataset"};
    orders_descriptor.columns = {{"id", common_agent_dataset_column_type::integer, false},
        {"customer_id", common_agent_dataset_column_type::integer, false},
        {"region", common_agent_dataset_column_type::string, true},
        {"value", common_agent_dataset_column_type::integer, true}};
    orders_descriptor.import_processor_id = "test-importer";
    orders_descriptor.origin.kind = "document_table";
    orders_descriptor.origin.source_representation_uri = "resource://uploads/orders.semantic";
    orders_descriptor.origin.source_node_id = "document-node://table/2";
    orders_descriptor.origin.table_index = 2;
    orders_descriptor.origin.caption = "Orders by customer";
    orders_descriptor.origin.header_mode = common_agent_table_header_mode::first_row;
    orders_descriptor.origin.header_confidence = 0.9;
    TEST_ASSERT(store.put_dataset_descriptor(orders_descriptor, error));
    common_agent_dataset_descriptor reloaded_orders;
    TEST_ASSERT(store.get_dataset_descriptor("dataset://local/orders", reloaded_orders, error));
    TEST_ASSERT(reloaded_orders.origin.kind == "document_table" &&
        reloaded_orders.origin.source_node_id == "document-node://table/2" &&
        reloaded_orders.origin.table_index == 2 &&
        reloaded_orders.origin.caption == "Orders by customer" &&
        reloaded_orders.origin.header_mode == common_agent_table_header_mode::first_row);
    common_agent_dataset_descriptor customers_descriptor = orders_descriptor;
    customers_descriptor.ref = {"dataset://local/customers", "Customers", 1, 2, "resource://uploads/customers.csv", "tabular-dataset"};
    customers_descriptor.columns = {{"customer_id", common_agent_dataset_column_type::integer, false},
        {"name", common_agent_dataset_column_type::string, true}};
    TEST_ASSERT(store.put_dataset_descriptor(customers_descriptor, error));
    common_agent_dataset_descriptor duplicate_customers = customers_descriptor;
    duplicate_customers.ref.uri = "dataset://local/customers-duplicate";
    TEST_ASSERT(store.put_dataset_descriptor(duplicate_customers, error));
    common_agent_dataset_descriptor ambiguous;
    TEST_ASSERT(!store.find_dataset_by_name("customers", ambiguous, error));
    TEST_ASSERT(error.find("dataset name is ambiguous") != std::string::npos &&
        error.find("customers-duplicate") != std::string::npos &&
        error.find("customers") != std::string::npos);

    std::string output;
    TEST_ASSERT(store.execute("data.query", R"({"dataset":"dataset://local/orders","order_by":[{"field":"value","direction":"desc"}],"max_scan_rows":2,"max_result_rows":1})", output, error));
    auto query = json::parse(output);
    TEST_ASSERT(query["scanned_rows"] == 2 && query["scan_truncated"] == true && query["row_count"] == 1 && query["result_truncated"] == true);
    TEST_ASSERT(query["rows"][0]["value"] == 12);

    TEST_ASSERT(store.execute("data.filter", R"({"dataset":"dataset://local/orders","conditions":[{"field":"region","operator":"=","value":"north"}],"max_scan_rows":10})", output, error));
    auto filtered = json::parse(output);
    TEST_ASSERT(filtered["row_count"] == 2 && filtered["rows"][0]["region"] == "north");

    TEST_ASSERT(store.execute("data.aggregate", R"({"dataset":"dataset://local/orders","group_by":["region"],"measures":[{"function":"count","column":"*","as":"count"},{"function":"sum","column":"value","as":"total"}]})", output, error));
    auto aggregate = json::parse(output);
    TEST_ASSERT(aggregate["row_count"] == 2 && aggregate["scan_mode"] == "native_bounded" && aggregate["scanned_rows"] == 3 && aggregate["scan_truncated"] == false);

    TEST_ASSERT(store.execute("data.aggregate", R"({"dataset":"dataset://local/orders","group_by":["region"],"measures":[{"function":"count","column":"*","as":"count"}],"max_scan_rows":2})", output, error));
    auto bounded_aggregate = json::parse(output);
    TEST_ASSERT(bounded_aggregate["scan_mode"] == "native_bounded" && bounded_aggregate["scanned_rows"] == 2 && bounded_aggregate["scan_truncated"] == true);

    TEST_ASSERT(store.execute("statistics.describe", R"({"dataset":"dataset://local/orders","columns":["value"]})", output, error));
    auto statistics = json::parse(output);
    TEST_ASSERT(statistics["columns"].size() == 1 && statistics["columns"][0]["count"] == 3 &&
        statistics["columns"][0]["null_count"] == 0 && statistics["columns"][0]["min"] == 4.0 &&
        statistics["columns"][0]["max"] == 12.0 && statistics["columns"][0]["mean"] == 8.0 &&
        std::abs(statistics["columns"][0]["stddev"].get<double>() - 3.265986323710904) < 1e-9);
    TEST_ASSERT(store.execute("statistics.describe", R"({"dataset":"dataset://local/orders"})", output, error));
    statistics = json::parse(output);
    TEST_ASSERT(statistics["columns"].size() == 3);
    TEST_ASSERT(store.execute("statistics.describe", R"({"dataset":"dataset://local/orders","columns":["value"],"group_by":["region"]})", output, error));
    auto grouped_statistics = json::parse(output);
    TEST_ASSERT(grouped_statistics["group_by"].size() == 1 && grouped_statistics["groups"].size() == 2);
    for (const auto & group : grouped_statistics["groups"]) {
        if (group["region"] == "north") {
            TEST_ASSERT(group["columns"][0]["count"] == 2 && group["columns"][0]["mean"] == 8.0);
        }
    }
    TEST_ASSERT(store.execute("statistics.outliers", R"({"dataset":"dataset://local/metrics","column":"value","group_by":["region"]})", output, error));
    auto outliers = json::parse(output);
    TEST_ASSERT(outliers["method"] == "iqr" && outliers["columns"].size() == 1 &&
        outliers["columns"][0]["groups"].size() == 1 &&
        outliers["columns"][0]["groups"][0]["outliers"].size() == 1 &&
        outliers["columns"][0]["groups"][0]["outliers"][0]["value"] == 100.0);

    TEST_ASSERT(store.put_row("dataset://local/orders", "4", R"({"id":4,"customer_id":12,"region":"north","value":null})", error));
    TEST_ASSERT(store.execute("statistics.value_counts", R"({"dataset":"dataset://local/orders","column":"region","limit":2})", output, error));
    auto value_counts = json::parse(output);
    TEST_ASSERT(value_counts["column"] == "region" && value_counts["distinct_count"] == 2 &&
        value_counts["null_count"] == 0 && value_counts["values"].size() == 2 &&
        value_counts["values"][0]["value"] == "north" && value_counts["values"][0]["count"] == 3 &&
        value_counts["result_truncated"] == false);
    TEST_ASSERT(store.execute("statistics.value_counts", R"({"dataset":"dataset://local/orders","column":"value"})", output, error));
    value_counts = json::parse(output);
    TEST_ASSERT(value_counts["null_count"] == 1 && value_counts["distinct_count"] == 3);

    TEST_ASSERT(store.execute("data.join", R"({"left":"dataset://local/orders","right":"dataset://local/customers","type":"inner","on":[{"left":"customer_id","right":"customer_id"}]})", output, error));
    auto inner_join = json::parse(output);
    TEST_ASSERT(inner_join["row_count"] == 2 && inner_join["scan_mode"] == "native_bounded" && inner_join["scanned_rows"] == 5 && inner_join["scan_truncated"] == false && inner_join["rows"][0].contains("name"));

    TEST_ASSERT(store.execute("data.join", R"({"left":"dataset://local/orders","right":"dataset://local/customers","type":"left","on":[{"left":"customer_id","right":"customer_id"}]})", output, error));
    auto join = json::parse(output);
    TEST_ASSERT(join["row_count"] == 4 && join["scan_mode"] == "native_bounded" && join["scanned_rows"] == 5 && join["scan_truncated"] == false && join["rows"][0].contains("name"));
    bool found_unmatched = false;
    for (const auto & row : join["rows"]) if (row.value("customer_id", 0) == 11) found_unmatched = !row.contains("name");
    TEST_ASSERT(found_unmatched);

    TEST_ASSERT(store.execute("data.join", R"({"left":"dataset://local/orders","right":"dataset://local/customers","type":"left","on":[{"left":"customer_id","right":"customer_id"}],"max_scan_rows":2})", output, error));
    auto bounded_join = json::parse(output);
    TEST_ASSERT(bounded_join["row_count"] == 2 && bounded_join["scanned_rows"] == 3 && bounded_join["scan_truncated"] == true);

    TEST_ASSERT(!store.execute("data.join", R"({"dataset":"dataset://local/orders","right":"dataset://local/customers","on":[]})", output, error));
    TEST_ASSERT(error.find("left") != std::string::npos);

    TEST_ASSERT(store.execute("data.filter", R"({"dataset":"dataset://local/orders","conditions":[{"field":"region","operator":"=","value":"north"}],"materialize":true,"result_dataset":"dataset://derived/north-orders"})", output, error));
    const auto materialized = json::parse(output);
    TEST_ASSERT(materialized["materialized"] == true && materialized["dataset"] == "dataset://derived/north-orders" && materialized["rows"] == 3);
    common_agent_dataset_descriptor derived;
    TEST_ASSERT(store.get_dataset_descriptor("dataset://derived/north-orders", derived, error));
    TEST_ASSERT(derived.lineage.parent_dataset_uris.size() == 1 && derived.lineage.parent_dataset_uris[0] == "dataset://local/orders" &&
        derived.lineage.operation == "data.filter" && derived.ref.source_resource_uri == "resource://uploads/orders.csv");
    TEST_ASSERT(!store.execute("data.filter", R"({"dataset":"dataset://local/orders","conditions":[],"max_scan_rows":1,"materialize":true,"result_dataset":"dataset://derived/truncated"})", output, error));
    TEST_ASSERT(error.find("truncated") != std::string::npos);

    store.close();
    fs::remove_all(root, ec);
    return 0;
}

#undef TEST_ASSERT
