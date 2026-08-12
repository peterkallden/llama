#include "tools/agent/data/agent-dataset-importer.h"

#include <cassert>
#include <iostream>
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

    bool put_dataset_descriptor(const common_agent_dataset_descriptor & descriptor,
            std::string & error) override {
        descriptors.push_back(descriptor);
        error.clear();
        return true;
    }

    std::vector<std::string> datasets;
    std::vector<std::string> row_ids;
    std::vector<std::string> rows;
    std::vector<common_agent_dataset_descriptor> descriptors;
};

int main() {
    recording_data_store store;
    agent_dataset_import_request request;
    request.source_resource_uri = "agent-resource://uploads/sales.xlsx";
    request.source_workbook_name = "sales.xlsx";
    request.import_processor_id = "pandoc-xlsx-workbook-json-v1";
    request.import_processor_version = "pandoc-3.10.1";
    std::string error;

    // Minimal Pandoc 1.23 AST shape: header, table head, one table body.
    const std::string pandoc_json = R"({"blocks":[
      {"t":"Header","c":[1,[],[{"t":"Str","c":"Sales"}]]},
      {"t":"Table","c":[[],[],[],
        [[],[[[],[[[],null,1,1,[{"t":"Plain","c":[{"t":"Str","c":"customer_id"}]}]],[[],null,1,1,[{"t":"Plain","c":[{"t":"Str","c":"amount"}]}]]]]]],
        [[[],0,[],[
          [[],[[[],null,1,1,[{"t":"Plain","c":[{"t":"Str","c":"1"}]}]],[[],null,1,1,[{"t":"Plain","c":[{"t":"Str","c":"10.5"}]}]]]],
          [[],[[[],null,1,1,[{"t":"Plain","c":[{"t":"Str","c":"2"}]}]],[[],null,1,1,[{"t":"Plain","c":[{"t":"Str","c":"12.0"}]}]]]]]],[]]
      ]}
    ]})";
    std::string normalized;
    if (!normalize_agent_pandoc_workbook_json(pandoc_json, normalized, error)) {
        std::cerr << error << "\n";
        return 1;
    }
    request.worksheet_json = normalized;

    std::vector<common_agent_dataset_descriptor> imported;
    assert(import_agent_worksheet_envelope(store, request, imported, error));
    assert(imported.size() == 1);
    assert(imported[0].ref.name == "Sales");
    assert(imported[0].ref.row_count == 2);
    assert(imported[0].ref.source_resource_uri == request.source_resource_uri);
    assert(imported[0].source_sheet_name == "Sales");
    assert(store.rows.size() == 2);
    assert(store.descriptors.size() == 1);
    assert(store.datasets[0] == imported[0].ref.uri);

    request.limits.max_rows = 1;
    assert(!import_agent_worksheet_envelope(store, request, imported, error));
    assert(error == "worksheet shape exceeds host limits");
    return 0;
}
