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
    descriptor.columns = {
        {"customer_id", common_agent_dataset_column_type::integer, false},
        {"amount", common_agent_dataset_column_type::decimal, true},
    };
    descriptor.source_workbook_name = "sales.xlsx";
    descriptor.source_sheet_name = "Sales";
    descriptor.source_sheet_index = 0;
    descriptor.source_range = "A1:B3";
    descriptor.import_processor_id = "pandoc-xlsx-workbook-json-v1";
    descriptor.import_processor_version = "pandoc-3.10.1";
    descriptor.origin.kind = "spreadsheet";
    descriptor.origin.source_node_id = "document-node://table/0";
    descriptor.origin.header_mode = common_agent_table_header_mode::explicit_;

    std::string error;
    assert(validate_common_agent_dataset_descriptor(
        descriptor, common_agent_dataset_limits{}, error));

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
