#include "agent-data-store-cozo.h"

#include <nlohmann/json.hpp>

#include <cassert>
#include <cstdio>
#include <filesystem>

using json = nlohmann::ordered_json;

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
    assert(store.put_row("orders", "1", R"({"id":1,"customer_id":10,"region":"north","value":12})", error));
    assert(store.put_row("orders", "2", R"({"id":2,"customer_id":11,"region":"south","value":8})", error));
    assert(store.put_row("orders", "3", R"({"id":3,"customer_id":10,"region":"north","value":4})", error));
    assert(store.put_row("customers", "10", R"({"customer_id":10,"name":"Ada"})", error));

    std::string output;
    assert(store.execute("data.query", R"({"dataset":"orders","order_by":[{"field":"value","direction":"desc"}],"max_scan_rows":2,"max_result_rows":1})", output, error));
    auto query = json::parse(output);
    assert(query["scanned_rows"] == 2 && query["scan_truncated"] == true && query["row_count"] == 1 && query["result_truncated"] == true);
    assert(query["rows"][0]["value"] == 12);

    assert(store.execute("data.aggregate", R"({"dataset":"orders","group_by":["region"],"measures":[{"function":"count","column":"*","as":"count"},{"function":"sum","column":"value","as":"total"}]})", output, error));
    auto aggregate = json::parse(output);
    assert(aggregate["row_count"] == 2);

    assert(store.execute("data.join", R"({"left":"orders","right":"customers","type":"left","on":[{"left":"customer_id","right":"customer_id"}]})", output, error));
    auto join = json::parse(output);
    assert(join["row_count"] == 3 && join["rows"][0].contains("name"));

    assert(!store.execute("data.join", R"({"dataset":"orders","right":"customers","on":[]})", output, error));
    assert(error.find("left") != std::string::npos);

    store.close();
    fs::remove_all(root, ec);
    return 0;
}
