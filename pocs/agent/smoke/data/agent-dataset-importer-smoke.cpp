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
    request.import_processor_id = "xlsx-workbook-json-v1";
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
      ]},
      {"t":"Header","c":[1,[],[{"t":"Str","c":"Customers"}]]},
      {"t":"Table","c":[[],[],[],
        [[],[[[],[[[],null,1,1,[{"t":"Plain","c":[{"t":"Str","c":"name"}]}]]]]]],
        [[[],0,[],[[[],[[[],null,1,1,[{"t":"Plain","c":[{"t":"Str","c":"Ada"}]}]]]]]],[]]
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
    assert(imported.size() == 2);
    assert(imported[0].ref.name == "Sales");
    assert(imported[0].ref.row_count == 2);
    assert(imported[0].ref.source_resource_uri == request.source_resource_uri);
    assert(imported[0].source_sheet_name == "Sales");
    assert(imported[0].origin.kind == "spreadsheet" &&
           imported[0].origin.source_node_id == "document-node://table/0" &&
           imported[0].origin.header_mode == common_agent_table_header_mode::explicit_);
    assert(imported[0].origin.header_confidence == 1.0);
    assert(store.rows.size() == 3);
    assert(imported[1].ref.name == "Customers");
    assert(store.descriptors.size() == 2);
    assert(store.datasets[0] == imported[0].ref.uri);

    agent_dataset_import_request document_request = request;
    document_request.source_resource_uri = "agent-resource://uploads/report.docx";
    document_request.source_workbook_name = "report.docx";
    document_request.source_representation = "document:table";
    document_request.source_representation_uri = "agent-resource://document/report.json";
    document_request.import_processor_id = "pandoc-docx-document-json-v1";
    document_request.sheet_name.reset();
    document_request.limits.max_rows = 100;
    std::string document_normalized;
    assert(normalize_agent_pandoc_document_json(pandoc_json, document_normalized, error));
    document_request.worksheet_json = document_normalized;
    assert(import_agent_worksheet_envelope(store, document_request, imported, error));
    assert(imported.size() == 2);
    assert(imported[0].ref.source_representation == "document:table");
    assert(imported[0].origin.kind == "document_table");
    assert(imported[0].origin.source_representation_uri == document_request.source_representation_uri);

    std::string csv_normalized;
    assert(normalize_agent_csv_text(
        "city,population,active\nStockholm,1000000,true\nKiruna,18000,false\n",
        "cities.csv", csv_normalized, error));
    agent_dataset_import_request csv_request = request;
    csv_request.source_resource_uri = "agent-resource://uploads/cities.csv";
    csv_request.source_workbook_name = "cities.csv";
    csv_request.source_representation = "csv:dataset";
    csv_request.source_representation_uri = csv_request.source_resource_uri;
    csv_request.import_processor_id = "csv-resource-import-v1";
    csv_request.import_processor_version = "1";
    csv_request.worksheet_json = csv_normalized;
    csv_request.sheet_name.reset();
    csv_request.limits.max_rows = 100;
    assert(import_agent_worksheet_envelope(store, csv_request, imported, error));
    assert(imported.size() == 1);
    assert(imported[0].ref.source_resource_uri == csv_request.source_resource_uri);
    assert(imported[0].ref.source_representation == "csv:dataset");
    assert(imported[0].columns.size() == 3 &&
           imported[0].columns[0].type == common_agent_dataset_column_type::string &&
           imported[0].columns[1].type == common_agent_dataset_column_type::integer &&
           imported[0].columns[2].type == common_agent_dataset_column_type::boolean);

    request.sheet_name = "Customers";
    request.limits.max_rows = 100;
    assert(import_agent_worksheet_envelope(store, request, imported, error));
    assert(imported.size() == 1 && imported[0].ref.name == "Customers");
    request.sheet_name.reset();
    request.limits.max_rows = 1;
    assert(!import_agent_worksheet_envelope(store, request, imported, error));
    assert(error == "worksheet shape exceeds host limits");
    return 0;
}
