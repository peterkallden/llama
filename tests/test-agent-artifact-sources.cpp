#include "agent/artifact/artifact-sources.h"
#include "hash/hash.h"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <algorithm>
#include <cassert>
#include <nlohmann/json.hpp>

int main() {
    common_agent_artifact_snapshot snapshot;
    common_runtime_resource_ref pdf;
    pdf.uri="agent-resource://pdf";pdf.name="/incoming/paper.pdf";
    pdf.metadata.source_provider="native";pdf.metadata.source_operation="web.fetch";
    pdf.metadata.source_request_json=R"({"url":"https://example.org/paper.pdf","max_bytes":10485760})";
    pdf.metadata.content_hash="pdf-hash";
    common_runtime_resource_ref ocr;
    ocr.uri="agent-resource://ocr";ocr.name="paper.txt";ocr.lineage.parent_uri=pdf.uri;
    common_agent_research_source source;
    source.source_id="paper";source.resource_ref=ocr;
    snapshot.research_sources.push_back(source);
    snapshot.observations.push_back({"fetch","web.fetch",
        R"({"status":200,"url":"https://example.org/paper.pdf","final_url":"https://cdn.example.org/paper(1).pdf","resource_ref":"agent-resource://pdf"})",
        1.0f,{}, {pdf},0,{}});
    snapshot.observations.push_back({"ocr","resource.read","{}",1.0f,{}, {ocr},0,{}});
    auto citations=common_agent_artifact_collect_sources(snapshot);
    const auto paper=std::find_if(citations.begin(),citations.end(),[](const auto & value){return value.source_id=="paper";});
    assert(paper!=citations.end());
    assert(paper->original_uri=="https://example.org/paper.pdf");
    assert(paper->resolved_uri=="https://cdn.example.org/paper(1).pdf");
    assert(paper->resource_uri==ocr.uri&&paper->file_name=="paper.txt");
    assert(paper->content_hash.empty()); // Do not label OCR bytes with the PDF's hash.
    assert(paper->request_summary.empty()); // Transport arguments are not search terms.
    const auto markdown=common_agent_artifact_render_source(*paper);
    assert(markdown.find("- paper.txt")!=std::string::npos);
    assert(markdown.find("[Original](<https://example.org/paper.pdf>)")!=std::string::npos);
    assert(markdown.find("[Retrieved from](<https://cdn.example.org/paper(1).pdf>)")!=std::string::npos);
    assert(markdown.find("[Saved source](<agent-resource://ocr>)")!=std::string::npos);
    assert(std::count_if(citations.begin(),citations.end(),[&](const auto & value){return value.resource_uri==ocr.uri;})==1);

    // The older canonical form omits workspace request provenance. Its
    // original SHA-256 identity must still verify after the V4 addition.
    common_agent_artifact_profile profile;profile.id="research";profile.revision="v1";
    profile.title="Report";profile.purpose="Test sources";
    profile.sections={{"findings","Findings","Findings"}};
    common_plan_state plan;plan.id="legacy-snapshot";plan.version=1;
    common_agent_research_workspace workspace;
    workspace.workspace_id="workspace";workspace.plan_id=plan.id;
    workspace.objective.objective_id="objective";workspace.objective.question="What does the paper report?";
    workspace.scope.namespace_id="test";workspace.scope.session_id="session";
    auto workspace_source=source;workspace_source.resource_ref=pdf;
    workspace.sources.push_back(workspace_source);
    common_agent_artifact_snapshot frozen;
    std::string text,error;
    assert(common_agent_artifact_snapshot_from_plan(plan,profile,frozen,error,&workspace,"checkpoint"));
    assert(common_agent_artifact_snapshot_to_json(frozen,text,error));
    auto legacy=nlohmann::ordered_json::parse(text);
    legacy.erase("fingerprint");legacy["schema_version"]=3;
    // Schema 3 predates the revision-bound research objective/task/assessment
    // fields introduced in schema 5. Remove them before computing its legacy
    // fingerprint, just as an actual persisted v3 artifact would.
    legacy.erase("research_revision");
    legacy.erase("research_objective_record");
    legacy.erase("research_tasks");
    for (auto & gap : legacy["research_gaps"]) gap.erase("assessment");
    for(auto & item:legacy["research_sources"]){
        item["resource_ref"].erase("source_provider");
        item["resource_ref"].erase("source_operation");
        item["resource_ref"].erase("source_request_json");
    }
    const auto material=legacy.dump();
    const auto old_fingerprint="sha256:"+hash_sha256_hex(material.data(),material.size());
    legacy["fingerprint"]=old_fingerprint;
    assert(common_agent_artifact_snapshot_from_json(legacy.dump(),frozen,error));
    assert(frozen.schema_version==3&&frozen.fingerprint==old_fingerprint);
    assert(common_agent_artifact_snapshot_to_json(frozen,text,error));
    assert(nlohmann::json::parse(text)["fingerprint"]==old_fingerprint);

    common_agent_artifact_snapshot api_snapshot;
    common_runtime_resource_ref api;
    api.uri="agent-resource://api";api.name="works.json";
    api.metadata.source_provider="example-api";api.metadata.source_operation="listWorks";
    api.metadata.source_request_json=R"({"query":{"search":"battery recycling","year":[2022,2025],"cursor":"opaque","api_key":"secret"},"sort":"date:desc","headers":{"Authorization":"secret"},"per_page":25})";
    api_snapshot.observations.push_back({"search","api.listWorks","{}",1.0f,{}, {api},0,{}});
    const auto api_citations=common_agent_artifact_collect_sources(api_snapshot);
    assert(api_citations.size()==1);
    const auto & api_source=api_citations.front();
    assert(api_source.original_uri.empty()); // Never invent an endpoint from the provider.
    assert(api_source.request_summary.find("battery recycling")!=std::string::npos);
    assert(api_source.request_summary.find("2022")!=std::string::npos);
    assert(api_source.request_summary.find("date:desc")!=std::string::npos);
    assert(api_source.request_summary.find("cursor")==std::string::npos);
    assert(api_source.request_summary.find("secret")==std::string::npos);
    assert(api_source.request_summary.find("per_page")==std::string::npos);

    common_agent_dataset_ref dataset;
    dataset.uri="dataset://works";dataset.name="Filtered works";
    dataset.source_resource_uri=api.uri;dataset.source_provider="example-api";dataset.source_operation="listWorks";
    dataset.source_request_json=R"({"search":"battery recycling","year":[2022,2025]})";
    api_snapshot.observations.back().dataset_refs.push_back(dataset);
    const auto dataset_citations=common_agent_artifact_collect_sources(api_snapshot);
    const auto dataset_source=std::find_if(dataset_citations.begin(),dataset_citations.end(),[&](const auto & value){return value.resource_uri==dataset.uri;});
    assert(dataset_source!=dataset_citations.end());
    assert(dataset_source->request_summary.find("battery recycling")!=std::string::npos);
    assert(common_agent_artifact_render_source(*dataset_source).find("[Saved source](<dataset://works>)")!=std::string::npos);

    common_agent_artifact_document document;
    document.schema_version=1;document.id="report";document.title="Report";
    document.source_bibliography=api_citations;
    assert(common_agent_artifact_document_to_json(document,text,error));
    common_agent_artifact_document restored;
    assert(common_agent_artifact_document_from_json(text,restored,error));
    assert(restored.source_bibliography.front().request_summary==api_source.request_summary);
    assert(common_agent_artifact_render_markdown(restored,text,error));
    assert(text.find("Request: query.search")!=std::string::npos);
    assert(text.find("[Saved source](<agent-resource://api>)")!=std::string::npos);
    document.markdown_renderer_version=2;
    assert(common_agent_artifact_render_markdown(document,text,error));
    assert(text.find("Request:")==std::string::npos&&text.find("[Saved source]")==std::string::npos);

    common_agent_artifact_source_citation file;
    file.source_id="file";file.file_name="notes[1].txt";file.resource_uri="agent-resource://file";
    assert(common_agent_artifact_render_source(file).find("notes\\[1\\].txt")!=std::string::npos);
    file.original_uri="javascript:alert(1)";
    assert(common_agent_artifact_render_source(file).find("javascript:")==std::string::npos);

    // Invalid/failing fetches and unrelated response URLs are not provenance.
    snapshot.observations.front().summary=R"({"status":403,"url":"https://wrong.example/","resource_ref":"agent-resource://pdf"})";
    snapshot.observations.front().confidence=0;
    citations=common_agent_artifact_collect_sources(snapshot);
    const auto fallback=std::find_if(citations.begin(),citations.end(),[](const auto & value){return value.source_id=="paper";});
    assert(fallback->original_uri=="https://example.org/paper.pdf"&&fallback->resolved_uri.empty());
    ocr.lineage.parent_uri=ocr.uri;
    snapshot.research_sources.front().resource_ref=ocr;
    snapshot.observations.back().resource_refs={ocr};
    citations=common_agent_artifact_collect_sources(snapshot); // Cycles terminate.
    assert(!citations.empty());
}
