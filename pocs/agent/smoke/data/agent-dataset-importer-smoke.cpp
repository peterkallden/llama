#include "tools/agent/data/agent-dataset-importer.h"

#include <cassert>
#include <string>
#include <vector>

class recording_data_store final : public common_agent_data_store {
public:
    bool put_row(const std::string & dataset, const std::string & row_id,
            const std::string & row_json, std::string & error) override {
        datasets.push_back(dataset);
        row_ids.push_back(row_id);
        rows.push_back(row_json);
        error.clear();
        return true;
    }

    bool execute(const std::string &, const std::string &, std::string &, std::string & error) override {
        error = "not used by importer smoke";
        return false;
    }

    std::vector<std::string> datasets;
    std::vector<std::string> row_ids;
    std::vector<std::string> rows;
};

int main() {
    recording_data_store store;
    agent_dataset_import_request request;
    request.source_resource_uri = "agent-resource://uploads/sales.xlsx";
    request.source_workbook_name = "sales.xlsx";
    request.import_processor_id = "pandoc-xlsx-workbook-json-v1";
    request.import_processor_version = "pandoc-3.10.1";
    request.worksheet_json = R"({
        "worksheets": [{
            "name": "Sales", "index": 0, "range": "A1:B3",
            "columns": [
                {"name": "customer_id", "type": "integer", "nullable": false},
                {"name": "amount", "type": "decimal"}
            ],
            "rows": [
                {"customer_id": 1, "amount": 10.5},
                {"customer_id": 2, "amount": 12.0}
            ]
        }]
    })";

    std::vector<common_agent_dataset_descriptor> imported;
    std::string error;
    assert(import_agent_worksheet_envelope(store, request, imported, error));
    assert(imported.size() == 1);
    assert(imported[0].ref.name == "Sales");
    assert(imported[0].ref.row_count == 2);
    assert(imported[0].ref.source_resource_uri == request.source_resource_uri);
    assert(imported[0].source_sheet_name == "Sales");
    assert(imported[0].source_range == "A1:B3");
    assert(store.rows.size() == 2);
    assert(store.datasets[0] == imported[0].ref.uri);

    request.limits.max_rows = 1;
    assert(!import_agent_worksheet_envelope(store, request, imported, error));
    assert(error == "worksheet shape exceeds host limits");
    return 0;
}
