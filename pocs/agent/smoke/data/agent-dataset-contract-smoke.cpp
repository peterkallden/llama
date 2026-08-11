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

    std::string error;
    assert(validate_common_agent_dataset_descriptor(
        descriptor, common_agent_dataset_limits{}, error));

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
