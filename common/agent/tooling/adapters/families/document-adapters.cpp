#include "agent/tooling/adapters/families/document-adapters.h"

#include "agent/tooling/adapters/support/adapter-support.h"
#include "agent/artifact/artifact-contract.h"
#include "resource/resource-contract.h"
#include "hash/hash.h"

#include <algorithm>
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
    if(descriptor.mime_type!="application/vnd.llama-agent.report+json"){error="resource is not a structured report document";return false;}
    std::string text;
    if(!bindings.resource_runtime.store->read_text(uri,authority,document_max_bytes,text,error))return false;
    return common_agent_artifact_document_from_json(text,document,error);
}

bool put_document_resource(const common_native_tool_bindings & bindings,const std::string & name,
        const std::string & mime,const std::string & text,const std::string & tool,
        const std::string & cache_key,const std::string & parent,std::string & uri,std::string & error){
    if(bindings.resource_runtime.store==nullptr){error="resource store is unavailable";return false;}
    agent_resource_put_request request;request.name=name;request.description="Immutable host-managed report revision resource";
    request.mime_type=mime;request.text=text;request.scope=bindings.resource_runtime.project_id.empty()?common_runtime_resource_scope::session:common_runtime_resource_scope::project;
    request.source_provider="native";request.source_tool=tool;request.metadata.purpose="Immutable report revision or publication manifest";
    request.metadata.content_summary=name;request.metadata.usage_hint="Use document.open to inspect a structured report revision.";
    request.metadata.limitations="The report source evidence remains authoritative.";request.metadata.processing_cache_key=cache_key;
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
        if(document_uri.empty()||revision==0){error="document head manifest is incomplete";return false;}return true;
    }
    int64_t latest_doc_time=-1;
    for(const auto & resource:resources){if(resource.mime_type!="application/vnd.llama-agent.report+json")continue;
        std::string text;if(!bindings.resource_runtime.store->read_text(resource.uri,authority,document_max_bytes,text,error))continue;
        common_agent_artifact_document candidate;std::string parse_error;if(!common_agent_artifact_document_from_json(text,candidate,parse_error)||candidate.id!=id)continue;
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
                {"section_title",new_section?new_section->title:old_section->title},
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
        const std::string & operation_id){
    std::lock_guard<std::mutex> lock(document_revision_mutex);
    common_agent_artifact_document original;std::string error;
    if(!read_document(bindings,document_uri,original,error))return common_adapter_not_found_failure("tool.document.save.document_unavailable",error,"The saved report could not be opened.");
    const auto input_fingerprint="sha256:"+hash_sha256_hex(markdown.data(),markdown.size());
    json prior_operation_record;
    if(!find_document_operation(bindings,original.id,operation_id,prior_operation_record,error))return common_adapter_execution_failure("tool.document.save.operation_lookup_failed",error,"The prior save operation could not be checked.");
    if(!prior_operation_record.empty()){
        if(prior_operation_record.value("base_document_uri",std::string{})!=document_uri||
                prior_operation_record.value("expected_revision",uint64_t(0))!=expected_revision||
                prior_operation_record.value("input_fingerprint",std::string{})!=input_fingerprint)
            return common_adapter_failure("tool.document.save.operation_conflict",common_tool_failure_class::conflict,false,
                "This operation ID was already used for different document content.","document operation ID payload mismatch");
        std::string latest_uri,latest_markdown,latest_operation;uint64_t latest_revision=0;
        if(!latest_document_head(bindings,original.id,latest_uri,latest_markdown,latest_revision,latest_operation,error))return common_adapter_execution_failure("tool.document.save.head_unavailable",error,"The current report revision could not be verified.");
        const auto saved_revision=prior_operation_record.value("revision",uint64_t(0));
        const auto saved_document_uri=prior_operation_record.value("document_uri",std::string{});
        const auto saved_markdown_uri=prior_operation_record.value("markdown_uri",std::string{});
        common_agent_artifact_document saved_document;
        if(!read_document(bindings,saved_document_uri,saved_document,error)||saved_document.revision!=saved_revision)
            return common_adapter_execution_failure("tool.document.save.retry_unavailable",error,"The prior saved revision could not be restored.");
        if((latest_revision==expected_revision&&latest_uri==document_uri)||
                (latest_revision==saved_revision&&latest_uri==saved_document_uri&&latest_operation!=operation_id)){
            if(!publish_document_head(bindings,original.id,saved_document_uri,saved_markdown_uri,saved_revision,
                    operation_id,saved_document.content_fingerprint,error))return common_adapter_execution_failure("tool.document.save.retry_publish_failed",error,"The prior save was recovered but its current revision could not be published.");
        }else if(latest_revision<saved_revision){
            return common_adapter_failure("tool.document.save.operation_conflict",common_tool_failure_class::conflict,false,
                "The saved operation is not consistent with the current report revision.","document operation is ahead of the current head");
        }
        return common_adapter_success_json(document_save_result(saved_document_uri,saved_markdown_uri,saved_revision,
            saved_document.content_fingerprint,true,prior_operation_record.value("diff",json::object())));
    }
    std::string latest_uri,latest_markdown,prior_operation;uint64_t latest_revision=0;
    if(!latest_document_head(bindings,original.id,latest_uri,latest_markdown,latest_revision,prior_operation,error))return common_adapter_execution_failure("tool.document.save.head_unavailable",error,"The current report revision could not be verified.");
    if(latest_revision!=expected_revision||(!latest_uri.empty()&&latest_uri!=document_uri))
        return common_adapter_failure("tool.document.save.revision_conflict",common_tool_failure_class::conflict,false,
            "The report changed since it was opened. Your local draft is preserved.","expected revision or document URI is no longer the current report head");
    common_agent_artifact_document revised;
    if(!common_agent_artifact_apply_markdown_edit(original,markdown,revised,error))return common_adapter_validation_failure("tool.document.save.invalid_markdown",error,"The edited report could not be parsed.");
    std::string canonical;if(!common_agent_artifact_render_editable_markdown(revised,canonical,error))return common_adapter_validation_failure("tool.document.save.invalid_markdown",error);
    revised.content_fingerprint="sha256:"+hash_sha256_hex(canonical.data(),canonical.size());
    if(revised.content_fingerprint==original.content_fingerprint){
        if(latest_revision==0)return common_adapter_success_json(document_save_result(document_uri,latest_markdown,original.revision,original.content_fingerprint,true));
        return common_adapter_success_json(document_save_result(latest_uri,latest_markdown,latest_revision,original.content_fingerprint,true));
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
    if(!put_document_resource(bindings,revised.id+"-r"+std::to_string(revised.revision)+"-report.json","application/vnd.llama-agent.report+json",document_json,"document.save",base+":report",document_uri,new_document_uri,error))return common_adapter_execution_failure("tool.document.save.report_write_failed",error,"The new report revision could not be saved.");
    const json operation_manifest={{"schema_version",1},{"document_id",revised.id},{"operation_id",operation_id},
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
} // namespace

bool common_try_register_document_tool_adapter(
        const common_tool_definition & definition,
        const common_native_tool_bindings & bindings,
        common_tool_registry & registry,
        bool & installed,
        std::string & error) {
    installed = false;
    const bool is_document_tool = definition.executor_id.rfind("builtin.document.",0)==0;
    if (!is_document_tool) return false;

    if(definition.executor_id=="builtin.document.open"){
        installed=common_adapter_register_definition(definition,registry,[bindings](const std::string & input){
            std::string parse_error;json arguments;
            if(!common_adapter_parse_object(input,arguments,parse_error)||!arguments.contains("document_uri")||!arguments["document_uri"].is_string())
                return common_adapter_validation_failure("tool.document.open.invalid_arguments",parse_error.empty()?"document.open requires document_uri":parse_error);
            const auto uri=arguments["document_uri"].get<std::string>();common_agent_artifact_document document;std::string error;
            if(!read_document(bindings,uri,document,error))return common_adapter_not_found_failure("tool.document.open.unavailable",error,"The report document is not available in this scope.");
            std::string markdown;if(!common_agent_artifact_render_editable_markdown(document,markdown,error))return common_adapter_execution_failure("tool.document.open.render_failed",error,"The report could not be opened for editing.");
            std::string serialized;if(!common_agent_artifact_document_to_json(document,serialized,error))return common_adapter_execution_failure("tool.document.open.serialize_failed",error,"The report metadata could not be opened.");
            const auto document_json=json::parse(serialized,nullptr,false);if(!document_json.is_object())return common_adapter_execution_failure("tool.document.open.serialize_failed","serialized report document is not valid JSON","The report metadata could not be opened.");
            json result={{"document_uri",uri},{"id",document.id},{"revision",document.revision},{"title",document.title},
                {"markdown",markdown},{"document",document_json},
                {"bibliography",json::array()}};
            for(const auto & source:document.source_bibliography)result["bibliography"].push_back({{"source_id",source.source_id},{"title",source.title},{"original_uri",source.original_uri},{"resolved_uri",source.resolved_uri},{"file_name",source.file_name},{"resource_uri",source.resource_uri},{"retrieved_at",source.retrieved_at}});
            return common_adapter_success_json(result);
        },error,true,false);
    } else if(definition.executor_id=="builtin.document.save"){
        installed=common_adapter_register_definition(definition,registry,[bindings](const std::string & input){
            std::string parse_error;json args;
            if(!common_adapter_parse_object(input,args,parse_error)||!args.contains("document_uri")||!args["document_uri"].is_string()||!args.contains("expected_revision")||!args["expected_revision"].is_number_unsigned()||!args.contains("markdown")||!args["markdown"].is_string()||!args.contains("operation_id")||!args["operation_id"].is_string())
                return common_adapter_validation_failure("tool.document.save.invalid_arguments",parse_error.empty()?"document.save requires document_uri, expected_revision, markdown and operation_id":parse_error);
            const auto operation=args["operation_id"].get<std::string>();if(!valid_operation_id(operation))return common_adapter_validation_failure("tool.document.save.invalid_operation_id","operation_id must be 1-128 ASCII letters, digits, dot, underscore or hyphen");
            return save_document(bindings,args["document_uri"].get<std::string>(),args["expected_revision"].get<uint64_t>(),args["markdown"].get<std::string>(),operation);
        },error,false,true);
    } else if(definition.executor_id=="builtin.document.propose_patch"){
        installed=common_adapter_register_definition(definition,registry,[bindings](const std::string & input){
            std::string parse_error;json args;
            if(!common_adapter_parse_object(input,args,parse_error)||!args.contains("document_uri")||!args["document_uri"].is_string()||!args.contains("expected_revision")||!args["expected_revision"].is_number_unsigned()||!args.contains("markdown")||!args["markdown"].is_string()||!args.contains("summary")||!args["summary"].is_string())
                return common_adapter_validation_failure("tool.document.propose_patch.invalid_arguments",parse_error.empty()?"document.propose_patch requires document_uri, expected_revision, markdown and summary":parse_error);
            const auto uri=args["document_uri"].get<std::string>();common_agent_artifact_document original,proposed;std::string error;
            if(!read_document(bindings,uri,original,error))return common_adapter_not_found_failure("tool.document.propose_patch.document_unavailable",error,"The report revision is unavailable.");
            std::string head,head_markdown,op;uint64_t revision=0;if(!latest_document_head(bindings,original.id,head,head_markdown,revision,op,error))return common_adapter_execution_failure("tool.document.propose_patch.head_unavailable",error,"The current report revision could not be checked.");
            const auto expected=args["expected_revision"].get<uint64_t>();if((revision&&revision!=expected)||(!head.empty()&&head!=uri)||original.revision!=expected)return common_adapter_failure("tool.document.propose_patch.stale",common_tool_failure_class::conflict,false,"This proposal targets an older report revision.","proposal base does not match current report head");
            if(!common_agent_artifact_apply_markdown_edit(original,args["markdown"].get<std::string>(),proposed,error))return common_adapter_validation_failure("tool.document.propose_patch.invalid_markdown",error,"The proposed edit is invalid.");
            const std::string summary=args["summary"].get<std::string>();if(summary.empty()||summary.size()>1024)return common_adapter_validation_failure("tool.document.propose_patch.invalid_summary","proposal summary is outside bounds");
            const json proposal={{"schema_version",1},{"document_id",original.id},{"document_uri",uri},{"base_revision",expected},{"summary",summary},{"markdown",args["markdown"]},{"content_fingerprint",proposed.content_fingerprint}};
            std::string proposal_uri;if(!put_document_resource(bindings,original.id+"-review-proposal.json","application/vnd.llama-agent.document-proposal+json",proposal.dump(),"document.propose_patch","document-proposal:"+original.id+":"+std::to_string(expected)+":"+proposed.content_fingerprint,uri,proposal_uri,error))return common_adapter_execution_failure("tool.document.propose_patch.store_failed",error,"The review proposal could not be saved.");
            return common_adapter_success_json({{"proposal_uri",proposal_uri},{"base_revision",expected},{"summary",summary},{"content_fingerprint",proposed.content_fingerprint}});
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
            std::string text;if(!bindings.resource_runtime.store->read_text(proposal_uri,authority,64*1024,text,error))return common_adapter_not_found_failure("tool.document.apply_patch.proposal_unavailable",error,"The selected proposal is unavailable.");
            const auto proposal=json::parse(text,nullptr,false);common_agent_artifact_document current;if(!proposal.is_object()||!read_document(bindings,doc_uri,current,error)||proposal.value("document_id",std::string{})!=current.id||proposal.value("document_uri",std::string{})!=doc_uri||proposal.value("base_revision",uint64_t(0))!=expected||current.revision!=expected)return common_adapter_failure("tool.document.apply_patch.stale",common_tool_failure_class::conflict,false,"The proposal no longer matches this report revision.",error.empty()?"proposal is not bound to the selected current report":error);
            return save_document(bindings,doc_uri,expected,proposal.value("markdown",std::string{}),operation_id);
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
