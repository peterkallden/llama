#include "common/agent/dataset-contracts.h"

#include <cassert>

int main() {
    assert(std::string(common_agent_dataset_column_type_name(
        common_agent_dataset_column_type::datetime)) == "datetime");

    common_agent_dataset_descriptor descriptor;
    descriptor.ref.uri = "dataset://analysis/sales";
    descriptor.ref.name = "Sales";
    descriptor.ref.row_count = 2;
    descriptor.ref.column_count = 2;
    descriptor.ref.source_resource_uri = "agent-resource://uploads/sales.xlsx";
    descriptor.ref.source_representation = "xlsx:worksheet";
    descriptor.ref.source_provider = "sales";
    descriptor.ref.source_operation = "listSales";
    descriptor.ref.source_request_json = R"({"query":{"limit":20}})";
    descriptor.ref.retrieved_at = 1787748689;
    descriptor.ref.content_hash = "sha256:collection-result";
    descriptor.columns = {
        {"customer_id", common_agent_dataset_column_type::integer, false},
        {"amount", common_agent_dataset_column_type::decimal, true},
    };
    descriptor.source_workbook_name = "sales.xlsx";
    descriptor.source_sheet_name = "Sales";
    descriptor.source_sheet_index = 0;
    descriptor.source_range = "A1:B3";
    descriptor.import_processor_id = "xlsx-workbook-json-v1";
    descriptor.import_processor_version = "pandoc-3.10.1";
    descriptor.origin.kind = "spreadsheet";
    descriptor.origin.source_node_id = "document-node://table/0";
    descriptor.origin.header_mode = common_agent_table_header_mode::explicit_;

    std::string error;
    assert(validate_common_agent_dataset_descriptor(
        descriptor, common_agent_dataset_limits{}, error));
    assert(descriptor.ref.source_provider == "sales");
    assert(descriptor.ref.source_operation == "listSales");
    assert(descriptor.ref.source_request_json == R"({"query":{"limit":20}})");
    assert(descriptor.ref.retrieved_at > 0);
    assert(descriptor.ref.content_hash == "sha256:collection-result");

    double confidence = 0.0;
    std::string reason;
    assert(classify_common_agent_table_headers(
        {{"City", "Population"}, {"Stockholm", "1000000"}, {"Uppsala", "170000"}},
        confidence, reason) == common_agent_table_header_mode::first_row);
    assert(confidence >= 0.8 && !reason.empty());
    assert(classify_common_agent_table_headers(
        {{"", "Q1", "Q2"}, {"Population", "10", "12"}, {"Revenue", "4", "5"}},
        confidence, reason) == common_agent_table_header_mode::both);
    assert(classify_common_agent_table_headers(
        {{"Population by city"}, {"Stockholm"}}, confidence, reason) == common_agent_table_header_mode::ambiguous);

    common_agent_document_table_catalog catalog;
    catalog.tables = {
        {0, "Population", "Population", "document-node://table/0", "dataset://report/table/0"},
        {1, "Budget summary", "Budget summary", "document-node://table/1", "dataset://report/table/1"},
    };
    common_agent_document_table_entry resolved;
    common_agent_document_table_locator by_name;
    by_name.name = "  budget   SUMMARY ";
    assert(resolve_common_agent_document_table(catalog, by_name, resolved, error));
    assert(resolved.table_index == 1 && resolved.dataset_uri == "dataset://report/table/1");
    common_agent_document_table_locator by_index;
    by_index.table_index = 0;
    assert(resolve_common_agent_document_table(catalog, by_index, resolved, error));
    assert(resolved.name == "Population");
    common_agent_document_table_locator by_node;
    by_node.node_id = "document-node://table/1";
    assert(resolve_common_agent_document_table(catalog, by_node, resolved, error));
    assert(resolved.table_index == 1);
    catalog.tables.push_back({2, "Budget summary", "", "document-node://table/2", "dataset://report/table/2"});
    assert(!resolve_common_agent_document_table(catalog, by_name, resolved, error));
    assert(error == "document table name is ambiguous");

    descriptor.ref.column_count = 1;
    assert(!validate_common_agent_dataset_descriptor(
        descriptor, common_agent_dataset_limits{}, error));
    assert(error == "dataset column count does not match schema");

    descriptor.ref.column_count = 2;
    descriptor.ref.source_resource_uri.clear();
    assert(!validate_common_agent_dataset_ref(descriptor.ref, error));
    assert(error == "dataset reference requires source resource provenance");
    return 0;
}
