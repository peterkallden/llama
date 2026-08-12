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
    const auto root = fs::temp_directory_path() / "llama-agent-data-manipulation-smoke";
    std::error_code ec;
    fs::remove_all(root, ec);
    fs::create_directories(root, ec);

    common_agent_cozo_data_store store;
    std::string error;
    if (!store.open((root / "data.cozo").string(), error)) {
        std::fprintf(stderr, "Cozo open failed: %s\n", error.c_str());
        return 1;
    }

    TEST_ASSERT(store.put_row("sales", "1", R"({"id":1,"customer_id":10,"region":"north","amount":12})", error));
    TEST_ASSERT(store.put_row("sales", "2", R"({"id":2,"customer_id":11,"region":"south","amount":8})", error));
    TEST_ASSERT(store.put_row("sales", "3", R"({"id":3,"customer_id":10,"region":"north","amount":4})", error));
    TEST_ASSERT(store.put_row("customers", "10", R"({"customer_id":10,"name":"Ada"})", error));
    TEST_ASSERT(store.put_row("customers", "11", R"({"customer_id":11,"name":"Linus"})", error));

    common_agent_dataset_descriptor sales;
    sales.ref = {"sales", "Sales", 3, 4, "resource://uploads/sales.xlsx", "xlsx:worksheet"};
    sales.columns = {{"id", common_agent_dataset_column_type::integer, false},
        {"customer_id", common_agent_dataset_column_type::integer, false},
        {"region", common_agent_dataset_column_type::string, false},
        {"amount", common_agent_dataset_column_type::integer, false}};
    sales.source_workbook_name = "sales.xlsx";
    sales.source_sheet_name = "Sales";
    sales.source_sheet_index = 0;
    sales.import_processor_id = "test-xlsx-importer";
    TEST_ASSERT(store.put_dataset_descriptor(sales, error));

    common_agent_dataset_descriptor customers = sales;
    customers.ref = {"customers", "Customers", 2, 2, "resource://uploads/sales.xlsx", "xlsx:worksheet"};
    customers.columns = {{"customer_id", common_agent_dataset_column_type::integer, false},
        {"name", common_agent_dataset_column_type::string, false}};
    customers.source_sheet_name = "Customers";
    customers.source_sheet_index = 1;
    TEST_ASSERT(store.put_dataset_descriptor(customers, error));

    std::string output;
    TEST_ASSERT(store.execute("data.filter", R"({"dataset":"sales","conditions":[{"field":"region","operator":"=","value":"north"}]})", output, error));
    auto filtered = json::parse(output);
    TEST_ASSERT(filtered["row_count"] == 2 && filtered["rows"].size() == 2);

    TEST_ASSERT(store.execute("data.join", R"({"left":"sales","right":"customers","type":"inner","on":[{"left":"customer_id","right":"customer_id"}]})", output, error));
    auto joined = json::parse(output);
    TEST_ASSERT(joined["row_count"] == 3 && joined["rows"][0].contains("name"));

    TEST_ASSERT(store.execute("data.aggregate", R"({"dataset":"sales","group_by":["region"],"measures":[{"function":"count","column":"*","as":"count"},{"function":"sum","column":"amount","as":"total"}],"materialize":true,"result_dataset":"dataset://analysis/sales-by-region"})", output, error));
    auto aggregate = json::parse(output);
    TEST_ASSERT(aggregate["materialized"] == true && aggregate["dataset"] == "dataset://analysis/sales-by-region");
    common_agent_dataset_descriptor derived;
    TEST_ASSERT(store.get_dataset_descriptor("dataset://analysis/sales-by-region", derived, error));
    TEST_ASSERT(derived.lineage.parent_dataset_uris.size() == 1 &&
        derived.lineage.parent_dataset_uris[0] == "sales" &&
        derived.lineage.operation == "data.aggregate" &&
        derived.ref.source_resource_uri == "resource://uploads/sales.xlsx");

    store.close();
    fs::remove_all(root, ec);
    return 0;
}

#undef TEST_ASSERT
