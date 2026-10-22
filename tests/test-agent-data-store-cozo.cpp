#include "agent-data-store-cozo.h"

#include <nlohmann/json.hpp>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <filesystem>

#ifdef NDEBUG
#undef NDEBUG
#endif

using json = nlohmann::ordered_json;

class virtual_dataset_fixture final : public common_agent_virtual_dataset_source {
public:
    size_t calls = 0;
    common_agent_scope last_scope;
    bool last_allow_network = false;
    bool authorized = true;
    bool authorize_source(const std::string & source_id, const common_agent_scope &, std::string & error) const override {
        if (!authorized || source_id != "elastic-a") { error = "source authorization denied"; return false; }
        error.clear(); return true;
    }
    bool execute(const common_agent_dataset_descriptor &, const std::string &,
            const std::string &, const common_agent_scope & scope, bool allow_network,
            std::string & result, std::string & error) override {
        ++calls;
        last_scope = scope;
        last_allow_network = allow_network;
        if (!allow_network) {
            error = "network permission is required for the virtual source";
            return false;
        }
        result = json{{"rows", json::array()}, {"row_count", 0}, {"complete", false},
            {"scan_truncated", true}, {"result_truncated", false}}.dump();
        error.clear();
        return true;
    }
};

#define TEST_ASSERT(expr) do { \
    const bool test_result = static_cast<bool>(expr); \
    assert(test_result); \
    if (!test_result) { std::fprintf(stderr, "test assertion failed at line %d: %s (error=%s)\n", __LINE__, #expr, error.c_str()); return 1; } \
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
    virtual_dataset_fixture virtual_source;
    TEST_ASSERT(store.set_virtual_dataset_source(&virtual_source));
    common_investigation_case_revision investigation;
    investigation.case_id = "case-persist-1";
    investigation.title = "Persistence contract";
    investigation.question = "Can this case be reopened?";
    investigation.scope.namespace_id = "tenant-a";
    investigation.scope.project_id = "project-a";
    investigation.scope.session_id = "session-a";
    investigation.source_id = "elastic-a";
    investigation.window_start_ms = 100;
    investigation.window_end_ms = 200;
    common_investigation_case_revision published;
    TEST_ASSERT(store.publish_investigation_case(investigation, 0,
        "resource://investigation/case-persist-1/revision-1", published, error));
    TEST_ASSERT(published.revision == 1 && published.fingerprint.rfind("sha256:", 0) == 0);
    const auto first_case_fingerprint = published.fingerprint;
    auto second_case = investigation;
    second_case.case_id = "case-persist-2";
    second_case.title = "Second case";
    TEST_ASSERT(store.publish_investigation_case(second_case, 0,
        "resource://investigation/case-persist-2/revision-1", published, error));
    std::vector<common_investigation_case_summary> listed_cases;
    std::string case_cursor;
    TEST_ASSERT(store.list_investigation_cases(investigation.scope, 1, "",
        listed_cases, case_cursor, error));
    TEST_ASSERT(listed_cases.size() == 1 && listed_cases[0].case_id == "case-persist-1" &&
        listed_cases[0].title == "Persistence contract" && case_cursor == "case-persist-1");
    const auto next_case_after = case_cursor;
    TEST_ASSERT(store.list_investigation_cases(investigation.scope, 1, next_case_after,
        listed_cases, case_cursor, error));
    TEST_ASSERT(listed_cases.size() == 1 && listed_cases[0].case_id == "case-persist-2" && case_cursor.empty());
    common_investigation_case_revision reloaded_case;
    TEST_ASSERT(store.get_investigation_case("case-persist-1", investigation.scope, reloaded_case, error));
    TEST_ASSERT(reloaded_case.revision == 1 && reloaded_case.fingerprint == first_case_fingerprint);
    virtual_source.authorized = false;
    TEST_ASSERT(!store.get_investigation_case("case-persist-1", investigation.scope, reloaded_case, error));
    virtual_source.authorized = true;
    TEST_ASSERT(store.get_investigation_case("case-persist-1", investigation.scope, reloaded_case, error));
    TEST_ASSERT(!store.publish_investigation_case(investigation, 0,
        "resource://investigation/case-persist-1/conflict", published, error));
    TEST_ASSERT(error.find("revision conflict") != std::string::npos);
    auto next_revision = reloaded_case;
    next_revision.revision = 2;
    next_revision.title = "Persistence contract updated";
    TEST_ASSERT(store.publish_investigation_case(next_revision, 1,
        "resource://investigation/case-persist-1/revision-2", published, error));
    TEST_ASSERT(published.revision == 2 && published.previous_revision_uri ==
        "resource://investigation/case-persist-1/revision-1");
    TEST_ASSERT(store.list_investigation_cases(investigation.scope, 1, "",
        listed_cases, case_cursor, error));
    TEST_ASSERT(listed_cases.size() == 1 && listed_cases[0].revision == 2 &&
        listed_cases[0].title == "Persistence contract updated");
    auto other_scope = investigation.scope;
    other_scope.project_id = "project-b";
    TEST_ASSERT(!store.get_investigation_case("case-persist-1", other_scope, reloaded_case, error));
    TEST_ASSERT(store.list_investigation_cases(other_scope, 10, "",
        listed_cases, case_cursor, error) && listed_cases.empty());

    common_agent_research_revision_ref research_revision;
    research_revision.research_id = "research-shared";
    research_revision.revision = 1;
    research_revision.parent_revision = 0;
    research_revision.payload_uri = "resource://project/research/revision-1";
    research_revision.payload_fingerprint = "sha256:research-v1";
    research_revision.publication_id = "publish-1";
    common_agent_research_revision_ref published_research;
    bool already_published = false;
    TEST_ASSERT(store.publish_research_revision(investigation.scope, research_revision, 0,
        published_research, already_published, error));
    TEST_ASSERT(!already_published && published_research.revision == 1);
    TEST_ASSERT(store.publish_research_revision(investigation.scope, research_revision, 0,
        published_research, already_published, error));
    TEST_ASSERT(already_published && published_research.publication_id == "publish-1");
    auto retry_research = research_revision;
    retry_research.payload_uri = "resource://project/research/orphaned-retry-copy";
    TEST_ASSERT(store.publish_research_revision(investigation.scope, retry_research, 0,
        published_research, already_published, error));
    TEST_ASSERT(already_published && published_research.payload_uri == research_revision.payload_uri);
    common_agent_research_revision_ref reloaded_research;
    TEST_ASSERT(store.get_research_revision("research-shared", 0, investigation.scope, reloaded_research, error));
    TEST_ASSERT(reloaded_research.revision == 1 && reloaded_research.payload_uri == research_revision.payload_uri);
    auto conflicting_research = research_revision;
    conflicting_research.revision = 2;
    conflicting_research.parent_revision = 1;
    conflicting_research.payload_uri = "resource://project/research/revision-2a";
    conflicting_research.payload_fingerprint = "sha256:research-v2a";
    conflicting_research.publication_id = "publish-2a";
    TEST_ASSERT(store.publish_research_revision(investigation.scope, conflicting_research, 1,
        published_research, already_published, error));
    auto stale_research = conflicting_research;
    stale_research.payload_uri = "resource://project/research/revision-2b";
    stale_research.payload_fingerprint = "sha256:research-v2b";
    stale_research.publication_id = "publish-2b";
    TEST_ASSERT(!store.publish_research_revision(investigation.scope, stale_research, 1,
        published_research, already_published, error));
    TEST_ASSERT(error.find("revision conflict") != std::string::npos);
    auto other_research_scope = investigation.scope;
    other_research_scope.project_id = "project-b";
    TEST_ASSERT(!store.get_research_revision("research-shared", 0, other_research_scope, reloaded_research, error));
    common_agent_research_draft_ref draft_ref{"research-shared", "draft-session-a", "session-a", 2, 1,
        "resource://session/draft-1", "sha256:draft-1", 111};
    common_agent_research_draft_ref saved_draft;
    bool already_saved = false;
    TEST_ASSERT(store.save_research_draft(investigation.scope, draft_ref, 0, saved_draft, already_saved, error));
    TEST_ASSERT(!already_saved && saved_draft.draft_revision == 1);
    common_agent_research_draft_ref loaded_draft;
    TEST_ASSERT(store.get_research_draft(draft_ref.draft_id, investigation.scope, loaded_draft, error));
    TEST_ASSERT(loaded_draft.owner_session_id == "session-a" && loaded_draft.base_revision == 2);

    TEST_ASSERT(store.put_row("dataset://local/orders", "1", R"({"id":1,"customer_id":10,"region":"north","value":12})", error));
    TEST_ASSERT(store.put_row("dataset://local/orders", "2", R"({"id":2,"customer_id":11,"region":"south","value":8})", error));
    TEST_ASSERT(store.put_row("dataset://local/orders", "3", R"({"id":3,"customer_id":10,"region":"north","value":4})", error));
    TEST_ASSERT(store.put_row("dataset://local/metrics", "1", R"({"region":"all","value":10})", error));
    TEST_ASSERT(store.put_row("dataset://local/metrics", "2", R"({"region":"all","value":11})", error));
    TEST_ASSERT(store.put_row("dataset://local/metrics", "3", R"({"region":"all","value":12})", error));
    TEST_ASSERT(store.put_row("dataset://local/metrics", "4", R"({"region":"all","value":13})", error));
    TEST_ASSERT(store.put_row("dataset://local/metrics", "5", R"({"region":"all","value":100})", error));
    TEST_ASSERT(store.put_row("dataset://local/customers", "10", R"({"customer_id":10,"name":"Ada"})", error));
    TEST_ASSERT(store.put_rows("dataset://local/batch", {
        {"batch-0", R"({"value":10})"}, {"batch-1", R"({"value":20})"}}, error));
    common_agent_dataset_descriptor batch_descriptor;
    batch_descriptor.ref = {"dataset://local/batch", "Batch", 2, 1, {}, "tabular-dataset"};
    batch_descriptor.ref.source_resource_uri = "resource://test/batch.json";
    batch_descriptor.ref.content_hash = "sha256:batch-content";
    batch_descriptor.import_processor_id = "test-importer";
    batch_descriptor.import_processor_version = "v1";
    batch_descriptor.columns = {{"value", common_agent_dataset_column_type::integer, false}};
    TEST_ASSERT(store.put_dataset_descriptor(batch_descriptor, error));
    common_agent_dataset_descriptor cached_batch;
    TEST_ASSERT(store.find_dataset_materialization("resource://test/batch.json", "sha256:batch-content", "v1", cached_batch, error));
    TEST_ASSERT(cached_batch.ref.uri == batch_descriptor.ref.uri);
    TEST_ASSERT(!store.find_dataset_materialization("resource://test/batch.json", "sha256:changed", "v1", cached_batch, error));
    TEST_ASSERT(error.empty());
    std::string batch_output;
    TEST_ASSERT(store.execute("data.query", R"({"dataset":"dataset://local/batch","max_result_rows":10})", batch_output, error));
    const auto batch_result = json::parse(batch_output);
    TEST_ASSERT(batch_result["rows"].size() == 2 &&
        batch_result["rows"][0]["value"] == 10 && batch_result["rows"][1]["value"] == 20);
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
    common_agent_dataset_descriptor virtual_dataset;
    virtual_dataset.ref.uri = "dataset://investigation/elastic-a/logs-2026.10";
    virtual_dataset.ref.name = "Security events";
    virtual_dataset.ref.column_count = 1;
    virtual_dataset.ref.source_representation = "elasticsearch:search";
    virtual_dataset.ref.source_provider = "elastic-a";
    virtual_dataset.ref.source_operation = "search";
    virtual_dataset.ref.source_request_json = R"({"time_field":"@timestamp"})";
    virtual_dataset.ref.storage_mode = common_agent_dataset_storage_mode::virtual_;
    virtual_dataset.ref.count_kind = common_agent_dataset_count_kind::unknown;
    virtual_dataset.columns = {{"event.id", common_agent_dataset_column_type::string, false}};
    virtual_dataset.capabilities = {"filter", "project", "sort", "page"};
    virtual_dataset.read_context_id = "read-context-1";
    TEST_ASSERT(store.put_dataset_descriptor(virtual_dataset, error));
    common_agent_scope virtual_scope;
    virtual_scope.namespace_id = "tenant-a";
    virtual_scope.project_id = "project-a";
    TEST_ASSERT(store.execute_scoped("data.query", R"({"dataset":"dataset://investigation/elastic-a/logs-2026.10","max_result_rows":5})", virtual_scope, true, output, error));
    TEST_ASSERT(virtual_source.calls == 1 && json::parse(output)["complete"] == false);
    TEST_ASSERT(virtual_source.last_scope.namespace_id == "tenant-a" &&
        virtual_source.last_scope.project_id == "project-a" && virtual_source.last_allow_network);
    TEST_ASSERT(!store.execute_scoped("data.query", R"({"dataset":"dataset://investigation/elastic-a/logs-2026.10","max_result_rows":5})", virtual_scope, false, output, error));
    TEST_ASSERT(error.find("network permission") != std::string::npos && virtual_source.calls == 2);
    TEST_ASSERT(!store.execute_scoped("data.query", R"({"dataset":"dataset://investigation/elastic-a/logs-2026.10","materialize":true,"result_dataset":"dataset://derived/all"})", virtual_scope, true, output, error));
    TEST_ASSERT(error.find("incomplete virtual") != std::string::npos && virtual_source.calls == 3);
    TEST_ASSERT(!store.execute("data.join", R"({"left":"dataset://investigation/elastic-a/logs-2026.10","right":"dataset://local/orders","type":"inner","on":[{"left":"event.id","right":"id"}]})", output, error));
    TEST_ASSERT(error.find("virtual datasets") != std::string::npos);

    TEST_ASSERT(store.execute("data.query", R"({"dataset":"dataset://local/orders","order_by":[{"field":"value","direction":"desc"}],"max_scan_rows":2,"max_result_rows":1})", output, error));
    auto query = json::parse(output);
    TEST_ASSERT(query["scanned_rows"] == 2 && query["scan_truncated"] == true && query["row_count"] == 1 && query["result_truncated"] == true);
    TEST_ASSERT(query["rows"][0]["value"] == 12);

    TEST_ASSERT(store.execute("data.filter", R"({"dataset":"dataset://local/orders","conditions":[{"field":"region","operator":"=","value":"north"}],"max_scan_rows":10})", output, error));
    auto filtered = json::parse(output);
    TEST_ASSERT(filtered["row_count"] == 2 && filtered["rows"][0]["region"] == "north");
    TEST_ASSERT(store.put_row("dataset://local/catalogue", "1", R"({"code":"DEMO_R_D3DENS","title":"Population density by NUTS 3 region"})", error));
    TEST_ASSERT(store.put_row("dataset://local/catalogue", "2", R"({"code":"GDP","title":"Gross domestic product"})", error));
    TEST_ASSERT(store.execute("data.query", R"({"dataset":"dataset://local/catalogue","where":[{"field":"title","operator":"contains","value":"POPULATION DENSITY"}],"max_scan_rows":10})", output, error));
    const auto catalogue_matches = json::parse(output);
    TEST_ASSERT(catalogue_matches["row_count"] == 1 && catalogue_matches["rows"][0]["code"] == "DEMO_R_D3DENS");
    for (const auto * operation : {"data.query", "data.filter"}) {
        const auto key = std::string(operation) == "data.query" ? "where" : "conditions";
        const json membership = {{"dataset", "dataset://local/orders"},
            {key, json::array({{{"field", "region"}, {"operator", "in"},
                {"value", json::array({"north", "east"})}}})}};
        TEST_ASSERT(store.execute(operation, membership.dump(), output, error));
        const auto matching = json::parse(output);
        TEST_ASSERT(matching["rows"].size() == 2);
        TEST_ASSERT(matching["rows"][0]["region"] == "north");
        json typed_membership = membership;
        typed_membership[key][0] = {{"field", "value"}, {"operator", "in"},
            {"value", json::array({12, "4"})}};
        TEST_ASSERT(store.execute(operation, typed_membership.dump(), output, error));
        TEST_ASSERT(json::parse(output)["rows"].size() == 1);
    }

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
    TEST_ASSERT(store.execute("data.aggregate", R"({"dataset":"dataset://derived/north-orders","measures":[{"function":"count","column":"*","as":"count"}]})", output, error));
    const auto derived_aggregate = json::parse(output);
    TEST_ASSERT(derived_aggregate["row_count"] == 1 && derived_aggregate["rows"][0]["count"] == 3);
    TEST_ASSERT(!store.execute("data.filter", R"({"dataset":"dataset://local/orders","conditions":[],"max_scan_rows":1,"materialize":true,"result_dataset":"dataset://derived/truncated"})", output, error));
    TEST_ASSERT(error.find("truncated") != std::string::npos);

    common_investigation_report_index_entry report_entry;
    report_entry.case_id = "case-persist-1";
    report_entry.case_revision = 1; // Historical reports remain independently indexed.
    report_entry.source_id = "elastic-a";
    report_entry.case_fingerprint = first_case_fingerprint;
    report_entry.content_fingerprint = "sha256:report-fixture";
    report_entry.report_uri = "resource://reports/case-persist-1-r1.json";
    report_entry.markdown_uri = "resource://reports/case-persist-1-r1.md";
    report_entry.source_manifest_uri = "resource://reports/case-persist-1-r1-sources.json";
    report_entry.visualization_uris = {"resource://snapshots/one.png"};
    report_entry.complete = false;
    report_entry.limitations = {"Coverage is partial."};
    report_entry.created_at_ms = 1234;
    common_investigation_report_index_entry indexed_report;
    TEST_ASSERT(store.publish_investigation_report(investigation.scope, report_entry, indexed_report, error));
    TEST_ASSERT(indexed_report.report_uri == report_entry.report_uri);
    // Publishing the same content fingerprint is idempotent and returns the original frozen references.
    auto retry_entry = report_entry;
    retry_entry.report_uri = "resource://reports/retry-must-not-replace.json";
    TEST_ASSERT(store.publish_investigation_report(investigation.scope, retry_entry, indexed_report, error));
    TEST_ASSERT(indexed_report.report_uri == report_entry.report_uri);
    std::vector<common_investigation_report_index_entry> indexed_reports;
    TEST_ASSERT(store.list_investigation_reports(investigation.scope, report_entry.case_id, 1, indexed_reports, error));
    TEST_ASSERT(indexed_reports.size() == 1 && indexed_reports[0].content_fingerprint == report_entry.content_fingerprint);
    auto report_other_scope = investigation.scope;
    report_other_scope.namespace_id = "tenant-b";
    TEST_ASSERT(store.list_investigation_reports(report_other_scope, report_entry.case_id, 1, indexed_reports, error));
    TEST_ASSERT(indexed_reports.empty());

    store.close();
    TEST_ASSERT(store.open((root / "data.cozo").string(), error));
    TEST_ASSERT(store.get_research_revision("research-shared", 0, investigation.scope,
        reloaded_research, error));
    TEST_ASSERT(reloaded_research.revision == 2 &&
        reloaded_research.payload_uri == conflicting_research.payload_uri);
    TEST_ASSERT(store.get_research_draft(draft_ref.draft_id, investigation.scope, loaded_draft, error));
    TEST_ASSERT(loaded_draft.draft_revision == 1 && loaded_draft.base_revision == 2 &&
        loaded_draft.payload_uri == draft_ref.payload_uri);
    TEST_ASSERT(store.list_investigation_cases(investigation.scope, 10, "",
        listed_cases, case_cursor, error));
    TEST_ASSERT(listed_cases.size() == 2 && listed_cases[0].case_id == "case-persist-1" &&
        listed_cases[0].revision == 2 && listed_cases[1].case_id == "case-persist-2");
    TEST_ASSERT(store.list_investigation_reports(investigation.scope, report_entry.case_id, 1, indexed_reports, error));
    TEST_ASSERT(indexed_reports.size() == 1 && indexed_reports[0].report_uri == report_entry.report_uri &&
        indexed_reports[0].case_fingerprint == first_case_fingerprint);
    store.close();
    fs::remove_all(root, ec);
    return 0;
}

#undef TEST_ASSERT
