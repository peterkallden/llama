#include "agent-data-store-sqlite.h"

#include <cassert>
#include <filesystem>
#include <nlohmann/json.hpp>

using json = nlohmann::ordered_json;

int main() {
    const auto path = std::filesystem::temp_directory_path() / "llama-agent-data-sqlite-smoke.sqlite";
    std::error_code ignored;
    std::filesystem::remove(path, ignored);

    common_agent_dataset_descriptor descriptor;
    descriptor.ref.uri = "dataset://orders";
    descriptor.ref.name = "orders";
    descriptor.columns.push_back({"amount", common_agent_dataset_column_type::integer, false});
    descriptor.ref.column_count = 1;

    std::string error;
    common_agent_sqlite_data_store store;
    assert(store.open(path.string(), error));
    assert(store.put_dataset_descriptor(descriptor, error));
    assert(store.put_row(descriptor.ref.uri, "1", json({{"amount", 10}}).dump(), error));
    assert(store.put_row(descriptor.ref.uri, "2", json({{"amount", 20}}).dump(), error));

    common_agent_dataset_descriptor found;
    assert(store.find_dataset_by_name(" orders ", found, error));
    assert(found.ref.uri == descriptor.ref.uri);

    std::string result;
    assert(store.execute("data.query", json({
        {"dataset", descriptor.ref.uri},
        {"where", json::array({json({{"field", "amount"}, {"operator", ">"}, {"value", 15}})})},
        {"max_scan_rows", 10}, {"max_result_rows", 10}
    }).dump(), result, error));
    const auto parsed = json::parse(result, nullptr, false);
    assert(parsed.is_object());
    assert(parsed["rows"].size() == 1);
    assert(parsed["rows"][0]["amount"] == 20);

    store.close();
    std::filesystem::remove(path, ignored);
    return 0;
}
