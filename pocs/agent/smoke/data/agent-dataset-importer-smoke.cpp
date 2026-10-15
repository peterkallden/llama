#include "tools/agent/data/agent-dataset-importer.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <iostream>
#include <nlohmann/json.hpp>
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
    document_request.origin_kind = "document_table";
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

    const std::string json_stat = R"({
      "class":"dataset","id":["geo","time"],"size":[2,2],
      "dimension":{
        "geo":{"category":{"index":{"SE":0,"NO":1},"label":{"SE":"Sweden","NO":"Norway"}}},
        "time":{"category":{"index":{"2020":0,"2021":1},"label":{"2020":"2020","2021":"2021"}}}
      },"value":[10,11,20,21],"status":["a","b","c","d"]
    })";
    std::string structured;
    assert(normalize_agent_json_dataset(json_stat, "population", structured, error));
    auto structured_root = nlohmann::json::parse(structured);
    const auto & stat_rows = structured_root["worksheets"][0]["rows"];
    assert(stat_rows.size() == 4);
    assert(stat_rows[0]["geo"] == "SE" && stat_rows[0]["geo_label"] == "Sweden");
    assert(stat_rows[2]["geo"] == "NO" && stat_rows[2]["time"] == "2020");
    assert(stat_rows[3]["value"] == 21 && stat_rows[3]["status"] == "d");
    const std::string sparse_json_stat = R"({
      "class":"dataset","id":["geo","time"],"size":[2,2],
      "dimension":{
        "geo":{"category":{"index":["SE","NO"]}},
        "time":{"category":{"index":["2020","2021"]}}
      },"value":{"1":11,"3":21},"status":{"3":"e"}
    })";
    assert(normalize_agent_json_dataset(sparse_json_stat, "sparse", structured, error));
    const auto sparse_rows = nlohmann::json::parse(structured)["worksheets"][0]["rows"];
    assert(sparse_rows.size() == 4 && sparse_rows[0]["geo"] == "SE" &&
        sparse_rows[0]["value"].is_null() && sparse_rows[1]["time"] == "2021" &&
        sparse_rows[2]["value"].is_null() && sparse_rows[3]["geo"] == "NO" &&
        sparse_rows[3]["status"] == "e");
    auto malformed_stat = nlohmann::json::parse(sparse_json_stat);
    malformed_stat["value"] = {{"01", 11}};
    assert(!normalize_agent_json_dataset(malformed_stat.dump(), "bad-index", structured, error));
    malformed_stat = nlohmann::json::parse(sparse_json_stat);
    malformed_stat["status"] = {{"9", "e"}};
    assert(!normalize_agent_json_dataset(malformed_stat.dump(), "bad-status", structured, error));
    malformed_stat = nlohmann::json::parse(json_stat);
    malformed_stat["id"][1] = "geo_label";
    malformed_stat["dimension"]["geo_label"] = malformed_stat["dimension"]["time"];
    assert(!normalize_agent_json_dataset(malformed_stat.dump(), "label-collision", structured, error));
    malformed_stat = nlohmann::json::parse(sparse_json_stat);
    malformed_stat["dimension"]["geo"]["category"]["index"] = {"SE", "SE"};
    assert(!normalize_agent_json_dataset(malformed_stat.dump(), "duplicate-codes", structured, error));
    common_agent_dataset_limits small_limits;
    small_limits.max_rows = 3;
    assert(!normalize_agent_json_dataset(sparse_json_stat, "sparse-limit", structured, error, small_limits));
    small_limits.max_source_bytes = 1;
    assert(!normalize_agent_json_dataset("[]", "bytes", structured, error, small_limits));
    assert(!normalize_agent_json_dataset(R"({"class":17})", "bad-class", structured, error));
    assert(normalize_agent_json_dataset(
        R"([{"code":"A","label":"Alpha"},{"code":"B","label":"Beta"}])",
        "catalog", structured, error));
    assert(nlohmann::json::parse(structured)["worksheets"][0]["rows"].size() == 2);
    assert(!normalize_agent_json_dataset(R"([{"nested":{"x":1}}])", "nested", structured, error));
    assert(!normalize_agent_json_dataset(R"([])", "empty", structured, error));

    const std::string toc_xml = R"(<nt:tree xmlns:nt="urn:eu.europa.ec.eurostat.navtree">
      <nt:branch><nt:title language="en">Economy</nt:title><nt:code>econ</nt:code><nt:children>
        <nt:leaf type="table"><nt:title language="en">GDP by country</nt:title>
          <nt:code>nama_10_gdp</nt:code><nt:lastUpdate>2026-01-01</nt:lastUpdate>
          <nt:dataStart>1995</nt:dataStart><nt:dataEnd>2025</nt:dataEnd><nt:values>1200</nt:values>
          <nt:metadata>https://example.invalid/metadata</nt:metadata>
        </nt:leaf>
      </nt:children></nt:branch></nt:tree>)";
    assert(normalize_agent_xml_dataset(toc_xml, "Eurostat catalogue", structured, error));
    auto toc_rows = nlohmann::json::parse(structured)["worksheets"][0]["rows"];
    assert(toc_rows.size() == 1 && toc_rows[0]["dataset_code"] == "nama_10_gdp");
    assert(toc_rows[0]["category_path"] == "Economy" && toc_rows[0]["values"] == 1200);
    std::string dataset_toc = toc_xml;
    dataset_toc.replace(dataset_toc.find("type=\"table\""), 12, "type=\"dataset\"");
    assert(normalize_agent_xml_dataset(dataset_toc, "dataset entries", structured, error));

    const std::string sdmx_xml = R"(<Structure xmlns="http://www.sdmx.org/resources/sdmxml/schemas/v2_1/structure">
      <Dataflows><Dataflow id="NAMA_10_GDP"><Name xml:lang="en">GDP</Name>
        <Structure><Ref id="DSD_NAMA_10_GDP"/></Structure></Dataflow></Dataflows>
      <DataStructures><DataStructure id="DSD_NAMA_10_GDP"><DataStructureComponents>
        <DimensionList><Dimension id="geo" position="1"><Name xml:lang="en">Geopolitical entity</Name>
          <ConceptIdentity><Ref id="GEO_CONCEPT"/></ConceptIdentity>
          <LocalRepresentation><Enumeration><Ref id="CL_GEO" agencyID="ESTAT" version="1.0"/></Enumeration></LocalRepresentation>
        </Dimension></DimensionList>
      </DataStructureComponents></DataStructure></DataStructures>
      <Codelists><Codelist id="CL_GEO"><Code id="SE"><Name xml:lang="en">Sweden</Name></Code></Codelist></Codelists>
      <Constraints><ContentConstraint id="AVAILABLE" type="Actual">
        <CubeRegion include="false"><KeyValue id="geo"><Value>NO</Value></KeyValue></CubeRegion>
        <CubeRegion include="true"><KeyValue id="unit"><Value>PC</Value></KeyValue></CubeRegion>
      </ContentConstraint></Constraints>
    </Structure>)";
    assert(normalize_agent_xml_dataset(sdmx_xml, "Eurostat metadata", structured, error));
    auto sdmx_rows = nlohmann::json::parse(structured)["worksheets"][0]["rows"];
    assert(sdmx_rows.size() >= 3);
    bool found_code = false;
    bool found_dimension = false;
    bool found_constraint = false;
    for (const auto & row : sdmx_rows) {
        if (row.value("artifact_kind", "") == "Code") found_code = row.value("code", "") == "SE";
        if (row.value("artifact_kind", "") == "Dimension") {
            found_dimension = row.value("dimension_id", "") == "geo" && row.value("codelist_id", "") == "CL_GEO";
            assert(row["codelist_agency_id"] == "ESTAT" && row["codelist_version"] == "1.0");
        }
        if (row.value("artifact_kind", "") == "Value" && row.value("code", "") == "NO") {
            found_constraint = row["dimension_id"] == "geo" && row["constraint_id"] == "AVAILABLE" &&
                row["constraint_type"] == "Actual" && row["region_include"] == "false" &&
                row["constraint_region"] == 0;
        }
    }
    assert(found_code && found_dimension && found_constraint);

    const std::string record_xml = R"(<records><record id="1"><country>SE</country><value>42</value></record>
      <record id="2"><country>NO</country><value>24</value></record></records>)";
    assert(normalize_agent_xml_dataset(record_xml, "records", structured, error));
    assert(nlohmann::json::parse(structured)["worksheets"][0]["rows"].size() == 2);
    const std::string external_entity = R"(<!DOCTYPE x [<!ENTITY e SYSTEM "file:///etc/passwd">]><records><record><value>&e;</value></record></records>)";
    assert(!normalize_agent_xml_dataset(external_entity, "unsafe", structured, error));
    assert(error.find("DTD") != std::string::npos || error.find("malformed") != std::string::npos);

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
