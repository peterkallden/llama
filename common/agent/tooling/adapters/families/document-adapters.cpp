#include "agent/tooling/adapters/families/document-adapters.h"

#include "agent/tooling/adapters/support/adapter-support.h"
#include "agent/tooling/adapters/support/processing-output.h"
#include "agent/artifact/artifact-contract.h"
#include "resource/resource-contract.h"
#include "hash/hash.h"

#include <algorithm>
#include <cctype>
#include <ctime>
#include <mutex>

using json = common_adapter_json;

namespace {
std::mutex document_revision_mutex;
constexpr size_t document_max_bytes=4*1024*1024;
bool valid_operation_id(const std::string & value){
    return !value.empty()&&value.size()<=128&&std::all_of(value.begin(),value.end(),[](unsigned char c){
        return (c>='a'&&c<='z')||(c>='A'&&c<='Z')||(c>='0'&&c<='9')||c=='-'||c=='_'||c=='.';
    });
}

bool read_document(const common_native_tool_bindings & bindings,const std::string & uri,
        common_agent_artifact_document & document,std::string & error){
    if(bindings.resource_runtime.store==nullptr){error="resource store is unavailable";return false;}
    const auto authority=make_agent_resource_read_authority(bindings.resource_runtime,static_cast<int64_t>(std::time(nullptr)));
    agent_resource_descriptor descriptor;
    if(!bindings.resource_runtime.store->stat(uri,authority,descriptor,error))return false;
    if(descriptor.mime_type!="application/vnd.llama-agent.report+json"&&descriptor.mime_type!="application/vnd.llama-agent.document+json"){error="resource is not a structured document";return false;}
    if(descriptor.size_bytes>document_max_bytes){error="structured report exceeds the document read limit";return false;}
    std::string text;
    if(!bindings.resource_runtime.store->read_text(uri,authority,document_max_bytes,text,error))return false;
    if(!common_agent_artifact_document_from_json(text,document,error))return false;
    if((document.kind==common_agent_artifact_document_kind::report)!=(descriptor.mime_type=="application/vnd.llama-agent.report+json")){error="document resource media type and document kind disagree";return false;}
    return true;
}

bool put_document_resource(const common_native_tool_bindings & bindings,const std::string & name,
        const std::string & mime,const std::string & text,const std::string & tool,
        const std::string & cache_key,const std::string & parent,std::string & uri,std::string & error){
    if(bindings.resource_runtime.store==nullptr){error="resource store is unavailable";return false;}
    agent_resource_put_request request;request.name=name;request.description="Immutable host-managed document revision resource";
    request.mime_type=mime;request.text=text;request.scope=bindings.resource_runtime.project_id.empty()?common_runtime_resource_scope::session:common_runtime_resource_scope::project;
    request.source_provider="native";request.source_tool=tool;request.metadata.purpose="Immutable report revision or publication manifest";
    request.metadata.content_summary=name;request.metadata.usage_hint="Use document.open to inspect a structured document revision.";
    request.metadata.limitations="Imported source resources remain unchanged and authoritative.";request.metadata.processing_cache_key=cache_key;
    request.lineage.parent_uri=parent;request.lineage.derivation=tool;
    if(!parent.empty()){request.lineage.chunk_count=1;request.lineage.byte_length=text.size();}
    apply_agent_resource_runtime(bindings.resource_runtime,request);
    agent_resource_descriptor descriptor;if(!bindings.resource_runtime.store->put_text(request,descriptor,error))return false;
    uri=descriptor.uri;return true;
}

bool latest_document_head(const common_native_tool_bindings & bindings,const std::string & id,
        std::string & document_uri,std::string & markdown_uri,uint64_t & revision,
        std::string & operation_id,std::string & error){
    document_uri.clear();markdown_uri.clear();revision=0;operation_id.clear();
    if(bindings.resource_runtime.store==nullptr){error="resource store is unavailable";return false;}
    const auto authority=make_agent_resource_read_authority(bindings.resource_runtime,static_cast<int64_t>(std::time(nullptr)));
    std::vector<agent_resource_descriptor> resources;if(!bindings.resource_runtime.store->list(authority,resources,error))return false;
    int64_t latest_time=-1;uint64_t head_revision=0;std::string latest_manifest;
    const std::string cache_key="document-head:"+id;
    for(const auto & resource:resources)if(resource.metadata.processing_cache_key==cache_key){
        std::string candidate_text;if(!bindings.resource_runtime.store->read_text(resource.uri,authority,64*1024,candidate_text,error))return false;
        const auto candidate=json::parse(candidate_text,nullptr,false);
        if(!candidate.is_object()||candidate.value("document_id",std::string{})!=id){error="document head manifest is malformed";return false;}
        const auto candidate_revision=candidate.value("revision",uint64_t(0));
        if(candidate_revision>head_revision||(candidate_revision==head_revision&&resource.created_at>=latest_time)){
            head_revision=candidate_revision;latest_time=resource.created_at;latest_manifest=resource.uri;
        }
    }
    if(!latest_manifest.empty()){
        std::string text;if(!bindings.resource_runtime.store->read_text(latest_manifest,authority,64*1024,text,error))return false;
        const auto manifest=json::parse(text,nullptr,false);if(!manifest.is_object()||manifest.value("document_id",std::string{})!=id){error="document head manifest is malformed";return false;}
        document_uri=manifest.value("document_uri",std::string{});markdown_uri=manifest.value("markdown_uri",std::string{});revision=manifest.value("revision",uint64_t(0));operation_id=manifest.value("operation_id",std::string{});
        if(document_uri.empty()||revision==0){error="document head manifest is incomplete";return false;}
        common_agent_artifact_document published;
        if(!read_document(bindings,document_uri,published,error)||published.id!=id||published.revision!=revision||
                published.content_fingerprint!=manifest.value("content_fingerprint",std::string{})){
            error="document head does not match its published report";return false;
        }
        agent_resource_descriptor markdown;
        if(!bindings.resource_runtime.store->stat(markdown_uri,authority,markdown,error)||markdown.mime_type!="text/markdown"){
            error="document head Markdown is unavailable";return false;
        }
    }
    // A later report composition can publish after an editor revision.
    const auto export_prefix="artifact-export:"+id+":r";
    for(const auto & resource:resources){
        const auto & key=resource.metadata.processing_cache_key;
        if(key.rfind(export_prefix,0)!=0||key.size()<9||key.substr(key.size()-9)!=":manifest")continue;
        std::string text;if(!bindings.resource_runtime.store->read_text(resource.uri,authority,1024*1024,text,error))return false;
        const auto manifest=json::parse(text,nullptr,false);
        if(!manifest.is_object()||manifest.value("report_id",std::string{})!=id||manifest.value("revision",uint64_t(0))<=revision)continue;
        common_agent_artifact_document published;
        const auto report_uri=manifest.value("report_resource",std::string{});
        const auto markdown_ref=manifest.value("markdown_resource",std::string{});
        agent_resource_descriptor final_resource;
        std::string expected_markdown,stored_markdown;
        if(!read_document(bindings,report_uri,published,error)||published.id!=id||published.revision!=manifest.value("revision",uint64_t(0))||
                published.snapshot_fingerprint!=manifest.value("snapshot_fingerprint",std::string{})||
                !common_agent_artifact_render_markdown(published,expected_markdown,error)||
                !bindings.resource_runtime.store->read_text(markdown_ref,authority,document_max_bytes,stored_markdown,error)||stored_markdown!=expected_markdown||
                !bindings.resource_runtime.store->stat(manifest.value("resource",std::string{}),authority,final_resource,error)){
            error="published report manifest failed document verification";return false;
        }
        revision=published.revision;document_uri=report_uri;markdown_uri=markdown_ref;operation_id.clear();
    }
    if(revision>0){error.clear();return true;}
    int64_t latest_doc_time=-1;
    for(const auto & resource:resources){if(resource.mime_type!="application/vnd.llama-agent.report+json"&&resource.mime_type!="application/vnd.llama-agent.document+json")continue;
        std::string text;if(!bindings.resource_runtime.store->read_text(resource.uri,authority,document_max_bytes,text,error))continue;
        common_agent_artifact_document candidate;std::string parse_error;if(!common_agent_artifact_document_from_json(text,candidate,parse_error)||candidate.id!=id)continue;
        // Edited resources are published only by a head manifest. A crash
        // between report/operation/head writes must not advance the head.
        if(candidate.schema_version>=2||!candidate.editor_operation_id.empty()||!candidate.editor_parent_uri.empty())continue;
        if(candidate.revision>revision||(candidate.revision==revision&&resource.created_at>latest_doc_time)){
            revision=candidate.revision;document_uri=resource.uri;markdown_uri=candidate.editor_markdown_uri;latest_doc_time=resource.created_at;
        }
    }
    error.clear();return true;
}

bool find_document_operation(const common_native_tool_bindings & bindings,const std::string & id,
        const std::string & operation_id,json & operation,std::string & error){
    operation=json::object();
    if(bindings.resource_runtime.store==nullptr){error="resource store is unavailable";return false;}
    const auto authority=make_agent_resource_read_authority(bindings.resource_runtime,static_cast<int64_t>(std::time(nullptr)));
    std::vector<agent_resource_descriptor> resources;if(!bindings.resource_runtime.store->list(authority,resources,error))return false;
    const auto key="document-operation:"+id+":"+operation_id;
    const agent_resource_descriptor * latest=nullptr;
    for(const auto & resource:resources)if(resource.metadata.processing_cache_key==key&&
            (latest==nullptr||resource.created_at>latest->created_at))latest=&resource;
    if(latest==nullptr){error.clear();return true;}
    std::string text;if(!bindings.resource_runtime.store->read_text(latest->uri,authority,64*1024,text,error))return false;
    operation=json::parse(text,nullptr,false);
    if(!operation.is_object()||operation.value("document_id",std::string{})!=id||
            operation.value("operation_id",std::string{})!=operation_id||
            operation.value("document_uri",std::string{}).empty()||operation.value("revision",uint64_t(0))==0){
        error="stored document operation record is malformed";return false;
    }
    error.clear();return true;
}

bool publish_document_head(const common_native_tool_bindings & bindings,const std::string & id,
        const std::string & document_uri,const std::string & markdown_uri,uint64_t revision,
        const std::string & operation_id,const std::string & content_fingerprint,std::string & error){
    const json manifest={{"document_id",id},{"document_uri",document_uri},{"markdown_uri",markdown_uri},
        {"revision",revision},{"operation_id",operation_id},{"content_fingerprint",content_fingerprint}};
    std::string manifest_uri;
    return put_document_resource(bindings,id+"-head.json","application/vnd.llama-agent.document-head+json",
        manifest.dump(),"document.save","document-head:"+id,document_uri,manifest_uri,error);
}

json document_save_result(const std::string & document_uri,const std::string & markdown_uri,
        uint64_t revision,const std::string & fingerprint,bool reused=false,json diff=json::object()){
    if(diff.empty())diff={{"base_revision",revision},{"revision",revision},{"blocks",json::array()},{"truncated",false}};
    return {{"document_uri",document_uri},{"markdown_uri",markdown_uri},{"revision",revision},
        {"content_fingerprint",fingerprint},{"reused",reused},{"diff",std::move(diff)}};
}

std::string diff_preview(const std::string & value,size_t max_bytes){
    if(value.size()<=max_bytes)return value;
    size_t end=max_bytes;while(end>0&&end<value.size()&&(static_cast<unsigned char>(value[end])&0xc0u)==0x80u)--end;
    return value.substr(0,end)+"…";
}

json document_diff(const common_agent_artifact_document & before,const common_agent_artifact_document & after){
    json diff={{"base_revision",before.revision},{"revision",after.revision},{"blocks",json::array()},
        {"truncated",false}};
    if(before.title!=after.title){
        diff["title"]={{"old_text",diff_preview(before.title,512)},{"new_text",diff_preview(after.title,512)}};
        if(before.title.size()>512||after.title.size()>512)diff["truncated"]=true;
    }
    size_t budget=8192;
    const auto sections=std::max(before.sections.size(),after.sections.size());
    for(size_t si=0;si<sections;++si){
        const common_agent_artifact_section * old_section=si<before.sections.size()?&before.sections[si]:nullptr;
        const common_agent_artifact_section * new_section=si<after.sections.size()?&after.sections[si]:nullptr;
        const auto count=std::max(old_section?old_section->blocks.size():0,new_section?new_section->blocks.size():0);
        for(size_t bi=0;bi<count;++bi){
            const common_agent_artifact_block * old_block=old_section&&bi<old_section->blocks.size()?&old_section->blocks[bi]:nullptr;
            const common_agent_artifact_block * new_block=new_section&&bi<new_section->blocks.size()?&new_section->blocks[bi]:nullptr;
            if(old_block&&new_block&&old_block->text==new_block->text&&old_section->title==new_section->title)continue;
            if(budget<128||diff["blocks"].size()>=32){diff["truncated"]=true;return diff;}
            const size_t preview=std::min<size_t>(2048,budget/2);
            std::string old_text=old_block?diff_preview(old_block->text,preview):std::string{};
            std::string new_text=new_block?diff_preview(new_block->text,preview):std::string{};
            budget-=std::min(budget,old_text.size()+new_text.size()+128);
            diff["blocks"].push_back({{"section_id",new_section?new_section->id:old_section->id},
                {"section_title",diff_preview(new_section?new_section->title:old_section->title,256)},
                {"change",!old_block?"added":!new_block?"removed":"updated"},
                {"previous_block_id",old_block?old_block->id:std::string{}},
                {"block_id",new_block?new_block->id:std::string{}},
                {"old_text",old_text},{"new_text",new_text}});
            if((old_block&&old_block->text.size()>old_text.size())||(new_block&&new_block->text.size()>new_text.size()))diff["truncated"]=true;
        }
    }
    return diff;
}

common_tool_execution_result save_document(const common_native_tool_bindings & bindings,
        const std::string & document_uri,uint64_t expected_revision,const std::string & markdown,
        const std::string & requested_title,const std::string & operation_id,bool model_proposal=false){
    std::lock_guard<std::mutex> lock(document_revision_mutex);
    common_agent_artifact_document original;std::string error;
    if(!read_document(bindings,document_uri,original,error))return common_adapter_not_found_failure("tool.document.save.document_unavailable",error,"The saved report could not be opened.");
    if(original.revision!=expected_revision)return common_adapter_failure("tool.document.save.revision_conflict",common_tool_failure_class::conflict,false,
        "The selected report does not match the expected revision.","document revision does not match expected_revision");
    if(!requested_title.empty()&&original.kind!=common_agent_artifact_document_kind::document)
        return common_adapter_validation_failure("tool.document.save.title_not_editable","report titles are edited in the report Markdown heading");
    const std::string operation_input=json{{"title",requested_title},{"markdown",markdown}}.dump();
    const auto input_fingerprint="sha256:"+hash_sha256_hex(operation_input.data(),operation_input.size());
    json prior_operation_record;
    if(!find_document_operation(bindings,original.id,operation_id,prior_operation_record,error))return common_adapter_execution_failure("tool.document.save.operation_lookup_failed",error,"The prior save operation could not be checked.");
    if(!prior_operation_record.empty()){
        const std::string legacy_input=requested_title.empty()?markdown:requested_title+"\n"+markdown;
        const auto legacy_fingerprint="sha256:"+hash_sha256_hex(legacy_input.data(),legacy_input.size());
        const bool legacy_input_format=prior_operation_record.value("input_format",1)==1;
        const auto expected_input=legacy_input_format?legacy_fingerprint:input_fingerprint;
        if(prior_operation_record.value("base_document_uri",std::string{})!=document_uri||
                prior_operation_record.value("expected_revision",uint64_t(0))!=expected_revision||
                prior_operation_record.value("input_fingerprint",std::string{})!=expected_input)
            return common_adapter_failure("tool.document.save.operation_conflict",common_tool_failure_class::conflict,false,
                "This operation ID was already used for different document content.","document operation ID payload mismatch");
        std::string latest_uri,latest_markdown,latest_operation;uint64_t latest_revision=0;
        if(!latest_document_head(bindings,original.id,latest_uri,latest_markdown,latest_revision,latest_operation,error))return common_adapter_execution_failure("tool.document.save.head_unavailable",error,"The current report revision could not be verified.");
        const auto saved_revision=prior_operation_record.value("revision",uint64_t(0));
        const auto saved_document_uri=prior_operation_record.value("document_uri",std::string{});
        const auto saved_markdown_uri=prior_operation_record.value("markdown_uri",std::string{});
        common_agent_artifact_document saved_document;
        if(!read_document(bindings,saved_document_uri,saved_document,error)||saved_document.id!=original.id||saved_document.revision!=saved_revision)
            return common_adapter_execution_failure("tool.document.save.retry_unavailable",error,"The prior saved revision could not be restored.");
        std::string saved_markdown;
        if(!common_agent_artifact_render_editable_markdown(saved_document,saved_markdown,error))return common_adapter_execution_failure("tool.document.save.retry_unavailable",error,"The prior saved revision could not be restored.");
        const auto saved_fingerprint="sha256:"+hash_sha256_hex(saved_markdown.data(),saved_markdown.size());
        if(legacy_input_format){
            common_agent_artifact_document candidate;std::string candidate_markdown;
            if(!common_agent_artifact_apply_markdown_edit(original,markdown,candidate,error)||
                    !common_agent_artifact_render_editable_markdown(candidate,candidate_markdown,error)||
                    candidate_markdown!=saved_markdown||saved_document.title!=(requested_title.empty()?candidate.title:requested_title))
                return common_adapter_failure("tool.document.save.operation_conflict",common_tool_failure_class::conflict,false,
                    "This operation ID was already used for different document content.","legacy document operation payload mismatch");
        }
        if(saved_fingerprint!=prior_operation_record.value("content_fingerprint",std::string{}))
            return common_adapter_execution_failure("tool.document.save.retry_unavailable","stored operation fingerprint does not match saved content","The saved operation failed content verification.");
        if(saved_revision>expected_revision&&((latest_revision==expected_revision&&latest_uri==document_uri)||
                (latest_revision==saved_revision&&latest_uri==saved_document_uri&&latest_operation!=operation_id))){
            if(!publish_document_head(bindings,original.id,saved_document_uri,saved_markdown_uri,saved_revision,
                    operation_id,saved_fingerprint,error))return common_adapter_execution_failure("tool.document.save.retry_publish_failed",error,"The prior save was recovered but its current revision could not be published.");
        }else if(latest_revision<saved_revision){
            return common_adapter_failure("tool.document.save.operation_conflict",common_tool_failure_class::conflict,false,
                "The saved operation is not consistent with the current report revision.","document operation is ahead of the current head");
        }
        return common_adapter_success_json(document_save_result(saved_document_uri,saved_markdown_uri,saved_revision,
            saved_fingerprint,true,prior_operation_record.value("diff",json::object())));
    }
    std::string latest_uri,latest_markdown,prior_operation;uint64_t latest_revision=0;
    if(!latest_document_head(bindings,original.id,latest_uri,latest_markdown,latest_revision,prior_operation,error))return common_adapter_execution_failure("tool.document.save.head_unavailable",error,"The current report revision could not be verified.");
    if(latest_revision!=expected_revision||(!latest_uri.empty()&&latest_uri!=document_uri))
        return common_adapter_failure("tool.document.save.revision_conflict",common_tool_failure_class::conflict,false,
            "The report changed since it was opened. Your local draft is preserved.","expected revision or document URI is no longer the current report head");
    common_agent_artifact_document revised;
    if(!common_agent_artifact_apply_markdown_edit(original,markdown,revised,error))return common_adapter_validation_failure("tool.document.save.invalid_markdown",error,"The edited report could not be parsed.");
    if(!requested_title.empty())revised.title=requested_title;
    if(model_proposal)for(auto & section:revised.sections)for(auto & block:section.blocks){
        bool unchanged=false;
        for(const auto & previous_section:original.sections)for(const auto & previous:previous_section.blocks)
            if(previous.id==block.id&&previous.text==block.text)unchanged=true;
        if(!unchanged){block.author=common_agent_artifact_author::model;block.origin="model_review";}
    }
    std::string canonical;if(!common_agent_artifact_render_editable_markdown(revised,canonical,error))return common_adapter_validation_failure("tool.document.save.invalid_markdown",error);
    revised.content_fingerprint="sha256:"+hash_sha256_hex(canonical.data(),canonical.size());
    std::string original_markdown;
    if(!common_agent_artifact_render_editable_markdown(original,original_markdown,error))return common_adapter_execution_failure("tool.document.save.render_failed",error,"The prior report could not be rendered.");
    const auto original_fingerprint="sha256:"+hash_sha256_hex(original_markdown.data(),original_markdown.size());
    if(revised.content_fingerprint==original_fingerprint&&revised.title==original.title){
        const json operation_manifest={{"schema_version",1},{"input_format",2},{"document_id",original.id},{"operation_id",operation_id},
            {"base_document_uri",document_uri},{"expected_revision",expected_revision},{"input_fingerprint",input_fingerprint},
            {"document_uri",document_uri},{"markdown_uri",latest_markdown},{"revision",original.revision},
            {"content_fingerprint",original_fingerprint},{"diff",document_diff(original,original)}};
        std::string operation_uri;
        if(!put_document_resource(bindings,original.id+"-operation-"+operation_id+".json","application/vnd.llama-agent.document-operation+json",
                operation_manifest.dump(),"document.save","document-operation:"+original.id+":"+operation_id,document_uri,operation_uri,error))
            return common_adapter_execution_failure("tool.document.save.operation_write_failed",error,"The unchanged save could not be made retry-safe.");
        return common_adapter_success_json(document_save_result(document_uri,latest_markdown,original.revision,original_fingerprint,true));
    }
    revised.revision=expected_revision+1;revised.previous_revision_uri=document_uri;revised.editor_parent_uri=document_uri;revised.editor_operation_id=operation_id;
    const auto diff=document_diff(original,revised);
    std::string document_json,rendered_markdown;
    if(!common_agent_artifact_render_markdown(revised,rendered_markdown,error))return common_adapter_execution_failure("tool.document.save.render_failed",error,"The edited report could not be rendered.");
    const auto base="document:"+revised.id+":r"+std::to_string(revised.revision)+":"+operation_id;
    std::string new_document_uri,new_markdown_uri;
    if(!put_document_resource(bindings,revised.id+"-r"+std::to_string(revised.revision)+".md","text/markdown",rendered_markdown,"document.save",base+":markdown",document_uri,new_markdown_uri,error))return common_adapter_execution_failure("tool.document.save.markdown_write_failed",error,"The Markdown revision could not be saved.");
    revised.editor_markdown_uri=new_markdown_uri;
    if(!common_agent_artifact_document_to_json(revised,document_json,error))return common_adapter_execution_failure("tool.document.save.render_failed",error,"The report document could not be finalized.");
    if(document_json.size()>document_max_bytes)return common_adapter_validation_failure("tool.document.save.document_too_large","serialized document exceeds the document read limit","The edited document is too large to reopen safely.");
    const auto structured_mime=revised.kind==common_agent_artifact_document_kind::document?"application/vnd.llama-agent.document+json":"application/vnd.llama-agent.report+json";
    if(!put_document_resource(bindings,revised.id+"-r"+std::to_string(revised.revision)+"-document.json",structured_mime,document_json,"document.save",base+":document",document_uri,new_document_uri,error))return common_adapter_execution_failure("tool.document.save.document_write_failed",error,"The new document revision could not be saved.");
    const json operation_manifest={{"schema_version",1},{"input_format",2},{"document_id",revised.id},{"operation_id",operation_id},
        {"base_document_uri",document_uri},{"expected_revision",expected_revision},{"input_fingerprint",input_fingerprint},
        {"document_uri",new_document_uri},{"markdown_uri",new_markdown_uri},{"revision",revised.revision},
        {"content_fingerprint",revised.content_fingerprint},{"diff",diff}};
    std::string operation_uri;
    if(!put_document_resource(bindings,revised.id+"-operation-"+operation_id+".json","application/vnd.llama-agent.document-operation+json",
            operation_manifest.dump(),"document.save","document-operation:"+revised.id+":"+operation_id,new_document_uri,operation_uri,error))
        return common_adapter_execution_failure("tool.document.save.operation_write_failed",error,"The saved revision could not be made retry-safe.");
    if(!publish_document_head(bindings,revised.id,new_document_uri,new_markdown_uri,revised.revision,
            operation_id,revised.content_fingerprint,error))return common_adapter_execution_failure("tool.document.save.publish_failed",error,"The revision was stored but its head manifest could not be published.");
    return common_adapter_success_json(document_save_result(new_document_uri,new_markdown_uri,revised.revision,
        revised.content_fingerprint,false,diff));
}

std::string plain_text_as_markdown(const std::string & text) {
    if(text.empty())return {};
    size_t fence_size=3;
    size_t run=0;
    for(char c:text){if(c=='`'){++run;fence_size=std::max(fence_size,run+1);}else run=0;}
    const std::string fence(fence_size,'`');
    return fence+"text\n"+text+(text.empty()||text.back()=='\n'?"":"\n")+fence;
}

common_tool_execution_result import_document(const common_native_tool_bindings & bindings,
        const std::string & source_uri,const std::string & operation_id) {
    std::lock_guard<std::mutex> lock(document_revision_mutex);
    if(bindings.resource_runtime.store==nullptr)return common_adapter_execution_failure("tool.document.import.store_unavailable","resource store is unavailable","Document import is unavailable.");
    const auto authority=make_agent_resource_read_authority(bindings.resource_runtime,static_cast<int64_t>(std::time(nullptr)));
    agent_resource_descriptor source;std::string error;
    if(!bindings.resource_runtime.store->stat(source_uri,authority,source,error))return common_adapter_not_found_failure("tool.document.import.source_unavailable",error,"The selected source file is unavailable in this session.");
    if(source.size_bytes>document_max_bytes)return common_adapter_validation_failure("tool.document.import.source_too_large","source exceeds the 4 MiB import limit","The selected file is too large to import.");
    std::string source_bytes;
    if(!bindings.resource_runtime.store->read_bytes(source_uri,authority,document_max_bytes+1,source_bytes,error))return common_adapter_execution_failure("tool.document.import.source_read_failed",error,"The selected file could not be read.");
    if(source_bytes.size()>document_max_bytes)return common_adapter_validation_failure("tool.document.import.source_too_large","source exceeds the 4 MiB import limit","The selected file is too large to import.");
    std::string extension=source.name.substr(source.name.find_last_of('.')==std::string::npos?source.name.size():source.name.find_last_of('.')+1);
    std::transform(extension.begin(),extension.end(),extension.begin(),[](unsigned char c){return static_cast<char>(std::tolower(c));});
    const auto mime=common_normalize_resource_media_type(source.mime_type);
    const bool is_docx=mime=="application/vnd.openxmlformats-officedocument.wordprocessingml.document"||extension=="docx";
    const bool is_markdown=mime=="text/markdown"||extension=="md"||extension=="markdown";
    const bool is_text=mime=="text/plain"||extension=="txt";
    if(!is_docx&&!is_markdown&&!is_text)return common_adapter_validation_failure("tool.document.import.unsupported_format","document.import supports TXT, Markdown and DOCX resources","Choose a TXT, Markdown or DOCX file.");
    const std::string source_hash="sha256:"+hash_sha256_hex(source_bytes.data(),source_bytes.size());
    const std::string operation_cache_key="document-import-operation:"+operation_id;
    std::vector<agent_resource_descriptor> operation_resources;
    if(!bindings.resource_runtime.store->list(authority,operation_resources,error))return common_adapter_execution_failure("tool.document.import.index_failed",error,"The imported document index could not be checked.");
    for(const auto & prior:operation_resources)if(prior.metadata.processing_cache_key==operation_cache_key){
        std::string operation_text;
        if(!bindings.resource_runtime.store->read_text(prior.uri,authority,65536,operation_text,error))return common_adapter_execution_failure("tool.document.import.retry_unavailable",error,"The previous import could not be verified.");
        const auto record=json::parse(operation_text,nullptr,false);
        if(!record.is_object()||record.value("source_uri",std::string{})!=source_uri||record.value("source_fingerprint",std::string{})!=source_hash)
            return common_adapter_failure("tool.document.import.operation_conflict",common_tool_failure_class::conflict,false,
                "This import operation ID was already used for a different source.","document import operation payload mismatch");
        const auto prior_uri=record.value("document_uri",std::string{});common_agent_artifact_document prior_document;
        if(prior_uri.empty()||!read_document(bindings,prior_uri,prior_document,error))return common_adapter_execution_failure("tool.document.import.retry_unavailable",error,"The previous imported document revision could not be restored.");
        std::string current,md,op;uint64_t revision=0;
        if(!latest_document_head(bindings,prior_document.id,current,md,revision,op,error))
            return common_adapter_execution_failure("tool.document.import.retry_unavailable",error,"The previous document publication could not be verified.");
        if(current.empty()){
            if(prior_document.editor_markdown_uri.empty()||!publish_document_head(bindings,prior_document.id,prior_uri,
                    prior_document.editor_markdown_uri,prior_document.revision,operation_id,prior_document.content_fingerprint,error))
                return common_adapter_execution_failure("tool.document.import.retry_publish_failed",error,"The previous import was stored but its head could not be recovered.");
            current=prior_uri;revision=prior_document.revision;
        }
        return common_adapter_success_json(json{{"document_uri",current.empty()?prior_uri:current},{"document_id",prior_document.id},
            {"revision",revision?revision:prior_document.revision},{"original_uri",source_uri},{"reused",true},
            {"limitations",record.value("limitations",json::array())}});
    }
    std::string markdown=source_bytes;
    std::string importer_version="document-import-v2";
    std::vector<std::string> limitations;
    if(is_text&&!is_markdown&&!is_docx)markdown=plain_text_as_markdown(source_bytes);
    else if(is_docx){
        if(bindings.resource_processing_service==nullptr&&!bindings.resource_processing_provider_factory)return common_adapter_execution_failure("tool.document.import.processor_unavailable","DOCX Markdown processor is not configured","DOCX import is unavailable because its document processor is not configured.");
        agent_resource_processing_binding_request request;request.source_uri=source_uri;request.operation_id="document-import:"+source_hash;request.authority=authority;
        request.media_type.declared_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document";
        request.media_type.resolved_type=request.media_type.declared_type;request.media_type.content_verified=true;
        request.target_representation="markdown";request.target_media_type="text/markdown";
        request.purpose=agent_resource_processing_purpose::normalization;request.limits.max_source_bytes=document_max_bytes;
        request.limits.max_output_bytes=document_max_bytes;request.limits.max_duration_ms=180000;request.limits.max_generated_resources=1;
        std::shared_ptr<agent_resource_processing_provider> owned;auto * processor=bindings.resource_processing_service;
        if(bindings.resource_processing_provider_factory){owned=bindings.resource_processing_provider_factory(request);processor=owned.get();}
        if(processor==nullptr)return common_adapter_execution_failure("tool.document.import.processor_unavailable","DOCX processor provider could not be created","DOCX import is unavailable.");
        const auto processed=processor->process(request);
        agent_resource_processing_output output;
        if(!common_adapter_read_processing_output(processed,*bindings.resource_runtime.store,authority,"text/markdown",document_max_bytes,output,error))
            return common_adapter_execution_failure("tool.document.import.conversion_failed",error,"The DOCX could not be converted into editable Markdown.");
        markdown=std::move(output.bytes);importer_version=processed.processor_id.empty()?"pandoc-docx-markdown-v1":processed.processor_id;
        if(!output.metadata.limitations.empty())limitations.push_back(output.metadata.limitations);
        bool omitted_images=false;std::string prepared;
        if(!common_agent_artifact_prepare_docx_markdown(markdown,prepared,omitted_images,error))
            return common_adapter_execution_failure("tool.document.import.invalid_markdown",error,"The DOCX Markdown could not be prepared.");
        markdown=std::move(prepared);
        if(omitted_images)limitations.push_back("Images are represented by alternative text; their original media remains in the original DOCX.");
        importer_version="document-import-v2:"+importer_version;
    }
    const auto material=source_uri+"\n"+source_hash+"\n"+importer_version;
    const auto import_hash=hash_sha256_hex(material.data(),material.size());
    const auto import_key="document-import:sha256:"+import_hash;
    std::vector<agent_resource_descriptor> resources=std::move(operation_resources);
    for(const auto & candidate:resources)if(candidate.metadata.processing_cache_key==import_key){
        common_agent_artifact_document existing;
        if(!read_document(bindings,candidate.uri,existing,error))return common_adapter_execution_failure("tool.document.import.cache_invalid",error,"A prior imported document could not be verified.");
        std::string current,md,op;uint64_t revision=0;
        if(!latest_document_head(bindings,existing.id,current,md,revision,op,error))return common_adapter_execution_failure("tool.document.import.head_failed",error,"The current imported document revision could not be checked.");
        if(current.empty()){
            if(existing.editor_markdown_uri.empty()||!publish_document_head(bindings,existing.id,candidate.uri,
                    existing.editor_markdown_uri,existing.revision,operation_id,existing.content_fingerprint,error))
                return common_adapter_execution_failure("tool.document.import.retry_publish_failed",error,"The stored import could not be published.");
            current=candidate.uri;revision=existing.revision;
        }
        const auto selected=current.empty()?candidate.uri:current;
        const json operation_record={{"schema_version",1},{"operation_id",operation_id},{"source_uri",source_uri},
            {"source_fingerprint",source_hash},{"document_uri",selected},{"document_id",existing.id},{"limitations",limitations}};
        std::string operation_uri;
        if(!put_document_resource(bindings,existing.id+"-import-operation-"+operation_id+".json","application/vnd.llama-agent.document-operation+json",
                operation_record.dump(),"document.import",operation_cache_key,selected,operation_uri,error))return common_adapter_execution_failure("tool.document.import.operation_write_failed",error,"The import could not be made retry-safe.");
        return common_adapter_success_json(json{{"document_uri",selected},{"document_id",existing.id},{"revision",revision?revision:existing.revision},
            {"original_uri",source_uri},{"reused",true},{"limitations",limitations}});
    }
    common_agent_artifact_document document;document.schema_version=5;document.kind=common_agent_artifact_document_kind::document;
    document.markdown_renderer_version=4;document.id="document-"+import_hash.substr(0,24);
    document.title=source.name.empty()?"Imported document":source.name;
    if(const auto dot=document.title.find_last_of('.');dot!=std::string::npos&&dot>0)document.title.resize(dot);
    document.purpose="Imported user document";document.provenance="imported";document.source_resource_uri=source_uri;document.source_content_fingerprint=source_hash;document.importer_version=importer_version;
    common_agent_artifact_document empty=document;empty.sections={{"document-content","Content",false,{},{}}};
    if(!common_agent_artifact_apply_markdown_edit(empty,markdown,document,error))return common_adapter_validation_failure("tool.document.import.invalid_content",error,"The selected file could not be parsed as an editable document.");
    document.revision=1;document.content_fingerprint="sha256:"+hash_sha256_hex(markdown.data(),markdown.size());
    std::string canonical;if(!common_agent_artifact_render_editable_markdown(document,canonical,error))return common_adapter_execution_failure("tool.document.import.render_failed",error,"The imported document could not be prepared.");
    document.content_fingerprint="sha256:"+hash_sha256_hex(canonical.data(),canonical.size());
    std::string markdown_uri,document_uri,serialized;
    if(!put_document_resource(bindings,document.id+"-r1.md","text/markdown",canonical,"document.import",import_key+":markdown",source_uri,markdown_uri,error))return common_adapter_execution_failure("tool.document.import.markdown_store_failed",error,"The editable document content could not be stored.");
    document.editor_markdown_uri=markdown_uri;
    if(!common_agent_artifact_document_to_json(document,serialized,error))return common_adapter_execution_failure("tool.document.import.serialize_failed",error,"The imported document metadata could not be validated.");
    if(!put_document_resource(bindings,document.id+"-r1-document.json","application/vnd.llama-agent.document+json",serialized,"document.import",import_key,source_uri,document_uri,error))return common_adapter_execution_failure("tool.document.import.document_store_failed",error,"The imported document could not be stored.");
    const json operation_record={{"schema_version",1},{"operation_id",operation_id},{"source_uri",source_uri},
        {"source_fingerprint",source_hash},{"document_uri",document_uri},{"document_id",document.id},{"limitations",limitations}};
    std::string operation_uri;
    if(!put_document_resource(bindings,document.id+"-import-operation-"+operation_id+".json","application/vnd.llama-agent.document-operation+json",
            operation_record.dump(),"document.import",operation_cache_key,document_uri,operation_uri,error))return common_adapter_execution_failure("tool.document.import.operation_write_failed",error,"The imported document could not be made retry-safe.");
    if(!publish_document_head(bindings,document.id,document_uri,markdown_uri,1,operation_id,document.content_fingerprint,error))return common_adapter_execution_failure("tool.document.import.publish_failed",error,"The document was stored but its initial revision could not be published.");
    return common_adapter_success_json(json{{"document_uri",document_uri},{"document_id",document.id},{"revision",1},
        {"original_uri",source_uri},{"reused",false},{"limitations",limitations}});
}

common_tool_execution_result create_document(const common_native_tool_bindings & bindings,
        const std::string & requested_title,const std::string & operation_id) {
    std::lock_guard<std::mutex> lock(document_revision_mutex);
    if(bindings.resource_runtime.store==nullptr)return common_adapter_execution_failure("tool.document.create.store_unavailable","resource store is unavailable","Document creation is unavailable.");
    const std::string title=requested_title.empty()?"Untitled document":requested_title;
    const auto & runtime=bindings.resource_runtime;
    const std::string owner_scope=runtime.project_id.empty()?"session:"+bindings.tool_scope.session_id:"project:"+runtime.project_id;
    const std::string scope_key=bindings.tool_scope.namespace_id+"\n"+owner_scope;
    const std::string operation_key="document-create-operation:"+hash_sha256_hex(scope_key.data(),scope_key.size())+":"+operation_id;
    const auto authority=make_agent_resource_read_authority(runtime,static_cast<int64_t>(std::time(nullptr)));
    std::vector<agent_resource_descriptor> resources;std::string error;
    if(!runtime.store->list(authority,resources,error))return common_adapter_execution_failure("tool.document.create.index_failed",error,"The document index could not be checked.");
    for(const auto & resource:resources)if(resource.metadata.processing_cache_key==operation_key){
        std::string record_text;if(!runtime.store->read_text(resource.uri,authority,65536,record_text,error))return common_adapter_execution_failure("tool.document.create.retry_unavailable",error,"The previous document creation could not be verified.");
        const auto record=json::parse(record_text,nullptr,false);
        if(!record.is_object()||record.value("operation_id",std::string{})!=operation_id||record.value("title",std::string{})!=title)
            return common_adapter_failure("tool.document.create.operation_conflict",common_tool_failure_class::conflict,false,"This create operation ID was already used for different document metadata.","document create operation payload mismatch");
        const auto uri=record.value("document_uri",std::string{});common_agent_artifact_document document;
        if(uri.empty()||!read_document(bindings,uri,document,error))return common_adapter_execution_failure("tool.document.create.retry_unavailable",error,"The created document revision could not be restored.");
        std::string head,markdown_uri,head_operation;uint64_t revision=0;
        if(!latest_document_head(bindings,document.id,head,markdown_uri,revision,head_operation,error))return common_adapter_execution_failure("tool.document.create.head_unavailable",error,"The document head could not be verified.");
        if(head.empty()&&!publish_document_head(bindings,document.id,uri,document.editor_markdown_uri,document.revision,operation_id,document.content_fingerprint,error))
            return common_adapter_execution_failure("tool.document.create.retry_publish_failed",error,"The initial revision was stored but its head could not be published.");
        return common_adapter_success_json(json{{"document_uri",uri},{"document_id",document.id},
            {"revision",document.revision},{"title",document.title},{"reused",true}});
    }
    const std::string identity_material=scope_key+"\n"+operation_id;
    const std::string identity_hash=hash_sha256_hex(identity_material.data(),identity_material.size());
    common_agent_artifact_document document;document.schema_version=5;document.kind=common_agent_artifact_document_kind::document;
    document.markdown_renderer_version=4;document.id="document-"+identity_hash.substr(0,24);document.title=title;
    document.purpose="User-authored document";document.provenance="authored";document.revision=1;
    document.sections={{"document-content","Content",false,{},{}}};
    common_agent_artifact_document initialized;
    if(!common_agent_artifact_apply_markdown_edit(document,{},initialized,error))return common_adapter_validation_failure("tool.document.create.initial_content_invalid",error,"The new document could not be initialized.");
    initialized.title=title;initialized.provenance="authored";initialized.revision=1;
    std::string canonical;
    if(!common_agent_artifact_render_editable_markdown(initialized,canonical,error))return common_adapter_execution_failure("tool.document.create.render_failed",error,"The new document could not be rendered.");
    initialized.content_fingerprint="sha256:"+hash_sha256_hex(canonical.data(),canonical.size());
    std::string markdown_uri,document_uri,serialized;
    const auto base="document-create:"+initialized.id+":r1";
    if(!put_document_resource(bindings,initialized.id+"-r1.md","text/markdown",canonical,"document.create",base+":markdown",{},markdown_uri,error))
        return common_adapter_execution_failure("tool.document.create.markdown_store_failed",error,"The initial document content could not be stored.");
    initialized.editor_markdown_uri=markdown_uri;
    if(!common_agent_artifact_document_to_json(initialized,serialized,error))return common_adapter_execution_failure("tool.document.create.serialize_failed",error,"The new document metadata could not be validated.");
    if(!put_document_resource(bindings,initialized.id+"-r1-document.json","application/vnd.llama-agent.document+json",serialized,
            "document.create",base+":document",{},document_uri,error))return common_adapter_execution_failure("tool.document.create.document_store_failed",error,"The initial document revision could not be stored.");
    const json record={{"schema_version",1},{"operation_id",operation_id},{"title",title},{"document_id",initialized.id},
        {"document_uri",document_uri},{"markdown_uri",markdown_uri},{"revision",1}};
    std::string operation_uri;
    if(!put_document_resource(bindings,initialized.id+"-create-operation-"+operation_id+".json","application/vnd.llama-agent.document-operation+json",
            record.dump(),"document.create",operation_key,document_uri,operation_uri,error))return common_adapter_execution_failure("tool.document.create.operation_write_failed",error,"The new document could not be made retry-safe.");
    if(!publish_document_head(bindings,initialized.id,document_uri,markdown_uri,1,operation_id,initialized.content_fingerprint,error))
        return common_adapter_execution_failure("tool.document.create.publish_failed",error,"The initial document revision was stored but could not be published.");
    return common_adapter_success_json(json{{"document_uri",document_uri},{"document_id",initialized.id},{"revision",1},{"title",title},{"reused",false}});
}
} // namespace

bool common_adapter_document_current_revision(const common_native_tool_bindings & bindings,const std::string & id,
        std::string & document_uri,uint64_t & revision,std::string & error){
    std::string markdown_uri,operation_id;
    return latest_document_head(bindings,id,document_uri,markdown_uri,revision,operation_id,error);
}

bool common_try_register_document_tool_adapter(
        const common_tool_definition & definition,
        const common_native_tool_bindings & bindings,
        common_tool_registry & registry,
        bool & installed,
        std::string & error) {
    installed = false;
    const bool is_document_tool = definition.executor_id.rfind("builtin.document.",0)==0;
    if (!is_document_tool) return false;

    if(definition.executor_id=="builtin.document.create"){
        installed=common_adapter_register_definition(definition,registry,[bindings](const std::string & input){
            std::string parse_error;json args;
            if(!common_adapter_parse_object(input,args,parse_error)||!args.contains("operation_id")||!args["operation_id"].is_string()||
                    (args.contains("title")&&!args["title"].is_string()))
                return common_adapter_validation_failure("tool.document.create.invalid_arguments",parse_error.empty()?"document.create requires operation_id and an optional title":parse_error);
            const auto operation=args["operation_id"].get<std::string>();
            if(!valid_operation_id(operation))return common_adapter_validation_failure("tool.document.create.invalid_operation_id","operation_id must be 1-128 ASCII letters, digits, dot, underscore or hyphen");
            const auto title=args.value("title",std::string("Untitled document"));
            if(title.empty()||title.size()>256)return common_adapter_validation_failure("tool.document.create.invalid_title","title must contain 1-256 bytes");
            return create_document(bindings,title,operation);
        },error,false,true);
    } else if(definition.executor_id=="builtin.document.import"){
        installed=common_adapter_register_definition(definition,registry,[bindings](const std::string & input){
            std::string parse_error;json args;
            if(!common_adapter_parse_object(input,args,parse_error)||!args.contains("source_uri")||!args["source_uri"].is_string()||!args.contains("operation_id")||!args["operation_id"].is_string())
                return common_adapter_validation_failure("tool.document.import.invalid_arguments",parse_error.empty()?"document.import requires source_uri and operation_id":parse_error);
            const auto op=args["operation_id"].get<std::string>();if(!valid_operation_id(op))return common_adapter_validation_failure("tool.document.import.invalid_operation_id","operation_id must be 1-128 ASCII letters, digits, dot, underscore or hyphen");
            return import_document(bindings,args["source_uri"].get<std::string>(),op);
        },error,false,true);
    } else if(definition.executor_id=="builtin.document.open"){
        installed=common_adapter_register_definition(definition,registry,[bindings](const std::string & input){
            std::string parse_error;json arguments;
            if(!common_adapter_parse_object(input,arguments,parse_error)||!arguments.contains("document_uri")||!arguments["document_uri"].is_string())
                return common_adapter_validation_failure("tool.document.open.invalid_arguments",parse_error.empty()?"document.open requires document_uri":parse_error);
            const auto uri=arguments["document_uri"].get<std::string>();common_agent_artifact_document document;std::string error;
            if(!read_document(bindings,uri,document,error))return common_adapter_not_found_failure("tool.document.open.unavailable",error,"The report document is not available in this scope.");
            std::string markdown;if(!common_agent_artifact_render_editable_markdown(document,markdown,error))return common_adapter_execution_failure("tool.document.open.render_failed",error,"The report could not be opened for editing.");
            std::string rendered_markdown;
            if(!common_agent_artifact_render_markdown(document,rendered_markdown,error))return common_adapter_execution_failure("tool.document.open.render_failed",error,"The document could not be rendered for reading.");
            std::string serialized;if(!common_agent_artifact_document_to_json(document,serialized,error))return common_adapter_execution_failure("tool.document.open.serialize_failed",error,"The report metadata could not be opened.");
            const auto document_json=json::parse(serialized,nullptr,false);if(!document_json.is_object())return common_adapter_execution_failure("tool.document.open.serialize_failed","serialized report document is not valid JSON","The report metadata could not be opened.");
            json result={{"document_uri",uri},{"id",document.id},{"kind",document.kind==common_agent_artifact_document_kind::document?"document":"report"},{"revision",document.revision},{"title",document.title},
                {"markdown",markdown},{"rendered_markdown",rendered_markdown},{"document",document_json},
                {"bibliography",json::array()}};
            for(const auto & source:document.source_bibliography)result["bibliography"].push_back({{"source_id",source.source_id},{"title",source.title},{"original_uri",source.original_uri},{"resolved_uri",source.resolved_uri},{"file_name",source.file_name},{"resource_uri",source.resource_uri},{"retrieved_at",source.retrieved_at}});
            return common_adapter_success_json(result);
        },error,true,false);
    } else if(definition.executor_id=="builtin.document.save"){
        installed=common_adapter_register_definition(definition,registry,[bindings](const std::string & input){
            std::string parse_error;json args;
            if(!common_adapter_parse_object(input,args,parse_error)||!args.contains("document_uri")||!args["document_uri"].is_string()||!args.contains("expected_revision")||!args["expected_revision"].is_number_unsigned()||!args.contains("markdown")||!args["markdown"].is_string()||!args.contains("operation_id")||!args["operation_id"].is_string()||(args.contains("title")&&!args["title"].is_string()))
                return common_adapter_validation_failure("tool.document.save.invalid_arguments",parse_error.empty()?"document.save requires document_uri, expected_revision, markdown and operation_id":parse_error);
            const auto operation=args["operation_id"].get<std::string>();if(!valid_operation_id(operation))return common_adapter_validation_failure("tool.document.save.invalid_operation_id","operation_id must be 1-128 ASCII letters, digits, dot, underscore or hyphen");
            const auto title=args.value("title",std::string{});if(title.size()>256)return common_adapter_validation_failure("tool.document.save.invalid_title","title exceeds 256 bytes");
            return save_document(bindings,args["document_uri"].get<std::string>(),args["expected_revision"].get<uint64_t>(),args["markdown"].get<std::string>(),title,operation);
        },error,false,true);
    } else if(definition.executor_id=="builtin.document.propose_patch"){
        installed=common_adapter_register_definition(definition,registry,[bindings](const std::string & input){
            std::string parse_error;json args;
            if(!common_adapter_parse_object(input,args,parse_error)||!args.contains("document_uri")||!args["document_uri"].is_string()||!args.contains("expected_revision")||!args["expected_revision"].is_number_unsigned()||!args.contains("summary")||!args["summary"].is_string())
                return common_adapter_validation_failure("tool.document.propose_patch.invalid_arguments",parse_error.empty()?"document.propose_patch requires document_uri, expected_revision, summary and a block replacement":parse_error);
            const auto uri=args["document_uri"].get<std::string>();common_agent_artifact_document original,proposed;std::string error;
            if(!read_document(bindings,uri,original,error))return common_adapter_not_found_failure("tool.document.propose_patch.document_unavailable",error,"The report revision is unavailable.");
            std::string head,head_markdown,op;uint64_t revision=0;if(!latest_document_head(bindings,original.id,head,head_markdown,revision,op,error))return common_adapter_execution_failure("tool.document.propose_patch.head_unavailable",error,"The current report revision could not be checked.");
            const auto expected=args["expected_revision"].get<uint64_t>();if((revision&&revision!=expected)||(!head.empty()&&head!=uri)||original.revision!=expected)return common_adapter_failure("tool.document.propose_patch.stale",common_tool_failure_class::conflict,false,"This proposal targets an older report revision.","proposal base does not match current report head");
            std::string markdown;
            const bool block_patch=args.contains("operation")||args.contains("section_id")||args.contains("block_id")||args.contains("text");
            if(block_patch){
                if(args.contains("markdown")||!args.contains("operation")||!args["operation"].is_string()||!args.contains("text")||!args["text"].is_string()||args["text"].get_ref<const std::string &>().empty()||args["text"].get_ref<const std::string &>().size()>16384)
                    return common_adapter_validation_failure("tool.document.propose_patch.invalid_target","provide operation and bounded Markdown text without full markdown");
                const auto action=args["operation"].get<std::string>();
                if(action!="replace"&&action!="insert_after"&&action!="append")return common_adapter_validation_failure("tool.document.propose_patch.invalid_operation","operation must be replace, insert_after or append");
                auto replacement=original;bool found=false;
                if(action=="append"&&replacement.kind==common_agent_artifact_document_kind::document&&replacement.sections.empty())replacement.sections.push_back({"document-content","Content",false,{},{}});
                if(action=="append"){
                    if(replacement.kind==common_agent_artifact_document_kind::report&&(!args.contains("section_id")||!args["section_id"].is_string()))return common_adapter_validation_failure("tool.document.propose_patch.invalid_target","report append requires section_id");
                    for(auto & section:replacement.sections)if(replacement.kind==common_agent_artifact_document_kind::document||section.id==args.value("section_id",std::string{})){
                        common_agent_artifact_block block;block.text=args["text"].get<std::string>();block.author=common_agent_artifact_author::model;block.origin="model_review";block.review_state=common_agent_artifact_review_state::unreviewed;block.id="proposal-append-"+std::to_string(section.blocks.size());section.blocks.push_back(std::move(block));found=true;break;
                    }
                }else{
                    if(!args.contains("section_id")||!args["section_id"].is_string()||!args.contains("block_id")||!args["block_id"].is_string())return common_adapter_validation_failure("tool.document.propose_patch.invalid_target","replace and insert_after require section_id and block_id");
                    for(auto & section:replacement.sections)if(section.id==args["section_id"]){
                        for(auto it=section.blocks.begin();it!=section.blocks.end();++it)if(it->id==args["block_id"]){
                            found=true;
                            if(action=="replace")it->text=args["text"].get<std::string>();
                            else {common_agent_artifact_block block;block.id="proposal-insert-"+std::to_string(section.blocks.size());block.text=args["text"].get<std::string>();block.author=common_agent_artifact_author::model;block.origin="model_review";block.review_state=common_agent_artifact_review_state::unreviewed;section.blocks.insert(std::next(it),std::move(block));}
                            break;
                        }
                    }
                }
                if(!found)return common_adapter_validation_failure("tool.document.propose_patch.unknown_target","the selected section or block does not belong to this document revision");
                if(!common_agent_artifact_render_editable_markdown(replacement,markdown,error))return common_adapter_validation_failure("tool.document.propose_patch.invalid_target",error);
            }else if(args.contains("markdown")&&args["markdown"].is_string())markdown=args["markdown"].get<std::string>();
            else return common_adapter_validation_failure("tool.document.propose_patch.invalid_target","provide a block replacement or legacy markdown");
            if(!common_agent_artifact_apply_markdown_edit(original,markdown,proposed,error))return common_adapter_validation_failure("tool.document.propose_patch.invalid_markdown",error,"The proposed edit is invalid.");
            if(block_patch){
                bool same_shape=original.kind==common_agent_artifact_document_kind::document||proposed.sections.size()==original.sections.size();
                for(size_t i=0;same_shape&&i<original.sections.size();++i)
                    same_shape=proposed.sections[i].id==original.sections[i].id&&proposed.sections[i].title==original.sections[i].title;
                if(!same_shape)return common_adapter_validation_failure("tool.document.propose_patch.invalid_target","the proposal must preserve report section structure");
            }
            const std::string summary=args["summary"].get<std::string>();if(summary.empty()||summary.size()>1024)return common_adapter_validation_failure("tool.document.propose_patch.invalid_summary","proposal summary is outside bounds");
            const json proposal={{"schema_version",1},{"document_id",original.id},{"document_uri",uri},{"base_revision",expected},{"summary",summary},{"markdown",markdown},{"content_fingerprint",proposed.content_fingerprint}};
            const auto proposal_text=proposal.dump();
            if(proposal_text.size()>document_max_bytes)return common_adapter_validation_failure("tool.document.propose_patch.too_large","serialized proposal exceeds the proposal read limit");
            std::string proposal_uri;if(!put_document_resource(bindings,original.id+"-review-proposal-"+proposed.content_fingerprint.substr(7)+".json","application/vnd.llama-agent.document-proposal+json",proposal_text,"document.propose_patch","document-proposal:"+original.id+":"+std::to_string(expected)+":"+proposed.content_fingerprint,uri,proposal_uri,error))return common_adapter_execution_failure("tool.document.propose_patch.store_failed",error,"The review proposal could not be saved.");
            return common_adapter_success_json({{"proposal_uri",proposal_uri},{"base_revision",expected},{"summary",summary},{"content_fingerprint",proposed.content_fingerprint},{"diff",document_diff(original,proposed)}});
        },error,false,true);
    } else if(definition.executor_id=="builtin.document.apply_patch"){
        installed=common_adapter_register_definition(definition,registry,[bindings](const std::string & input){
            std::string parse_error;json args;
            if(!common_adapter_parse_object(input,args,parse_error)||!args.contains("document_uri")||!args["document_uri"].is_string()||!args.contains("proposal_uri")||!args["proposal_uri"].is_string()||!args.contains("expected_revision")||!args["expected_revision"].is_number_unsigned()||!args.contains("operation_id")||!args["operation_id"].is_string())
                return common_adapter_validation_failure("tool.document.apply_patch.invalid_arguments",parse_error.empty()?"document.apply_patch requires document_uri, proposal_uri, expected_revision and operation_id":parse_error);
            const auto operation_id=args["operation_id"].get<std::string>();if(!valid_operation_id(operation_id))return common_adapter_validation_failure("tool.document.apply_patch.invalid_operation_id","operation_id must be 1-128 ASCII letters, digits, dot, underscore or hyphen");
            const auto doc_uri=args["document_uri"].get<std::string>(),proposal_uri=args["proposal_uri"].get<std::string>();const auto expected=args["expected_revision"].get<uint64_t>();
            const auto authority=make_agent_resource_read_authority(bindings.resource_runtime,static_cast<int64_t>(std::time(nullptr)));agent_resource_descriptor descriptor;std::string error;
            if(bindings.resource_runtime.store==nullptr||!bindings.resource_runtime.store->stat(proposal_uri,authority,descriptor,error)||descriptor.mime_type!="application/vnd.llama-agent.document-proposal+json")return common_adapter_not_found_failure("tool.document.apply_patch.proposal_unavailable",error,"The selected proposal is unavailable.");
            if(descriptor.size_bytes>document_max_bytes)return common_adapter_validation_failure("tool.document.apply_patch.proposal_too_large","proposal exceeds the proposal read limit");
            std::string text;if(!bindings.resource_runtime.store->read_text(proposal_uri,authority,document_max_bytes,text,error))return common_adapter_not_found_failure("tool.document.apply_patch.proposal_unavailable",error,"The selected proposal is unavailable.");
            const auto proposal=json::parse(text,nullptr,false);common_agent_artifact_document current;if(!proposal.is_object()||!read_document(bindings,doc_uri,current,error)||proposal.value("document_id",std::string{})!=current.id||proposal.value("document_uri",std::string{})!=doc_uri||proposal.value("base_revision",uint64_t(0))!=expected||current.revision!=expected)return common_adapter_failure("tool.document.apply_patch.stale",common_tool_failure_class::conflict,false,"The proposal no longer matches this report revision.",error.empty()?"proposal is not bound to the selected current report":error);
            return save_document(bindings,doc_uri,expected,proposal.value("markdown",std::string{}),{},operation_id,true);
        },error,false,true);
    } else if (definition.executor_id == "builtin.document.tables" && bindings.document_tables) {
        installed = common_adapter_register_definition(definition, registry, [bindings](const std::string & input) {
            std::string parse_error;
            json arguments;
            if (!common_adapter_parse_object(input, arguments, parse_error) ||
                    !arguments.contains("resource") || !arguments["resource"].is_string()) {
                return common_adapter_validation_failure(
                    "tool.document.tables.invalid_arguments",
                    parse_error.empty() ? "document.tables requires resource" : std::move(parse_error));
            }
            const int max_results = arguments.value("max_results", 32);
            if (max_results < 1 || max_results > 64) {
                return common_adapter_validation_failure(
                    "tool.document.tables.invalid_limit",
                    "document.tables max_results is out of bounds");
            }
            return bindings.document_tables(arguments.dump());
        }, error);
    } else if (definition.executor_id == "builtin.document.table" && bindings.document_table) {
        installed = common_adapter_register_definition(definition, registry, [bindings](const std::string & input) {
            std::string parse_error;
            json arguments;
            if (!common_adapter_parse_object(input, arguments, parse_error) ||
                    !arguments.contains("resource") || !arguments["resource"].is_string()) {
                return common_adapter_validation_failure(
                    "tool.document.table.invalid_arguments",
                    parse_error.empty() ? "document.table requires resource" : std::move(parse_error));
            }
            size_t locators = 0;
            for (const auto * key : {"table", "table_index", "node_id"}) {
                if (arguments.contains(key)) ++locators;
            }
            if (locators != 1) {
                return common_adapter_validation_failure(
                    "tool.document.table.invalid_locator",
                    "document.table requires exactly one of table, table_index or node_id");
            }
            return bindings.document_table(arguments.dump());
        }, error);
    }
    return true;
}
