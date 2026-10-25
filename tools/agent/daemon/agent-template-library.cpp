#include "agent-template-library.h"
#include "agent/agent-package-json.h"
#include "tools/agent/resource/agent-resource-store.h"
#include "memory/memory-in-memory.h"
#include "plan/plan-in-memory.h"
#include "hash/hash.h"
#include <fstream>
#include <sstream>
#include <ctime>
#include <algorithm>

using json=nlohmann::ordered_json;
namespace {
constexpr size_t max_package_bytes=16*1024*1024;
constexpr size_t max_receipt_bytes=4*max_package_bytes;
common_agent_bootstrap_config installation(const common_agent_template_library_manifest & m) {
    common_agent_bootstrap_config c;
    c.namespace_id=m.level=="host"?"llama-agent-host":m.namespace_id;
    c.session_id="bootstrap";c.project_id=m.project_id;c.now=std::time(nullptr);
    c.definition_prefix=m.definition_prefix;return c;
}
bool project_package(const common_agent_bootstrap_package & source,
        const common_agent_template_library_manifest & manifest,common_memory_store & memories,
        common_plan_store & plans,std::string & text,std::string & error) {
    auto result=source;
    for(auto & b:result.blueprints){
        auto p=plans.get(manifest.definition_prefix+"blueprint:"+b.id,error);
        if(!p || p->kind!=common_plan_kind::blueprint){error="library blueprint is missing: "+b.id;return false;}
        b.source_revision=p->source_revision;b.selection_description=p->selection_description;
        b.purpose=p->purpose;b.goal=p->goal;b.success_criteria=p->success_criteria;b.steps=p->steps;
        b.required_capabilities=p->required_capabilities;b.constraints=p->constraints;b.assumptions=p->assumptions;
        b.workflow_policy=p->workflow_policy;b.procedure_refs=p->procedure_refs;
        b.workflow_bindings=p->workflow_bindings;b.next_action=p->next_action;
        const auto c=installation(manifest);
        if(p->namespace_id!=c.namespace_id || p->project_id!=c.project_id){error="library blueprint owner changed";return false;}
    }
    for(auto & w:result.workflows){
        auto p=plans.get(manifest.definition_prefix+"workflow:"+w.id,error);
        const auto c=installation(manifest);
        if(!p || p->kind!=common_plan_kind::workflow || !p->workflow_definition ||
                p->namespace_id!=c.namespace_id || p->project_id!=c.project_id){error="library workflow is missing or has another owner";return false;}
        w.source_revision=p->source_revision;w.definition=*p->workflow_definition;
    }
    for(auto & d:result.procedures){
        auto p=memories.get(manifest.definition_prefix+"procedure:"+d.id,error);
        const auto c=installation(manifest);
        if(!p || p->kind!=common_memory_kind::procedure || p->namespace_id!=c.namespace_id ||
                p->project_id!=c.project_id){error="library procedure is missing or has another owner";return false;}
        d.content=p->content;d.summary=p->summary;d.importance=p->importance;d.confidence=p->confidence;
    }
    return common_agent_package_to_json(result,text,error,false);
}
bool verify(const common_agent_bootstrap_package & package,const common_agent_template_library_manifest & manifest,
        common_memory_store & memories,common_plan_store & plans,std::string & error){
    std::set<std::pair<std::string, std::string>> workflow_identities;
    for (const auto & item : package.workflows) {
        if (!workflow_identities.emplace(item.definition.workflow_ref, item.definition.workflow_revision).second) {
            error = "library contains ambiguous workflow reference and revision";
            return false;
        }
    }
    std::map<std::string, std::string> blueprints;
    std::set<std::string> workflows, procedures;
    for (const auto & item : package.blueprints) blueprints[item.id] = manifest.definition_prefix + "blueprint:" + item.id;
    for (const auto & item : package.workflows) workflows.insert(manifest.definition_prefix + "workflow:" + item.id);
    for (const auto & item : package.procedures) procedures.insert(manifest.definition_prefix + "procedure:" + item.id);
    if (blueprints != manifest.blueprints || workflows != manifest.workflows || procedures != manifest.procedures ||
            package.name != manifest.package_name || package.version != manifest.package_version) {
        error = "library receipt membership does not match its package";
        return false;
    }
    common_memory_in_memory_store expected_memories;common_plan_in_memory_store expected_plans;
    if(!expected_memories.open("",error)||!expected_plans.open("",error))return false;
    common_agent_bootstrap_result receipt;
    auto c=installation(manifest);
    if(!common_agent_install_bootstrap_package(expected_memories,expected_plans,c,package,
        [](const std::string &,std::vector<float> & v,std::string &){v={1.0f};return true;},receipt,error) ||
        !receipt.rejected_items.empty()){if(error.empty())error="library package has rejected definitions";return false;}
    std::string expected,actual;
    if(!project_package(package,manifest,expected_memories,expected_plans,expected,error)||
            !project_package(package,manifest,memories,plans,actual,error))return false;
    if(actual!=expected){error="immutable library definitions differ from their package receipt";return false;}
    return true;
}
std::string fingerprint(const common_agent_template_library_config & c,const std::string & canonical){
    const auto payload=json{{"level",c.level},{"namespace_id",c.namespace_id},{"project_id",c.project_id},
        {"package",json::parse(canonical)}}.dump();return hash_sha256_hex(payload.data(),payload.size());
}

}
bool load_agent_template_libraries(const std::vector<common_agent_template_library_config> & libraries,
        common_memory_store & memories,common_plan_store & plans,agent_resource_store & resources,
        common_agent_bootstrap_embedder embed,common_agent_template_library_snapshot & snapshot,
        std::vector<common_blueprint_candidate> & candidates,std::string & error,
    bool restore_fresh_ephemeral_definitions){
    snapshot={};candidates.clear();error.clear();
    common_agent_template_library_snapshot staged;
    std::vector<common_blueprint_candidate> staged_candidates;
    agent_resource_read_authority authority;authority.namespace_id="llama-agent-host";authority.session_id="bootstrap";authority.now=std::time(nullptr);
    std::vector<agent_resource_descriptor> prior;
    if(!resources.list(authority,prior,error))return false;
    for(const auto & config:libraries){
        std::ifstream file(config.package,std::ios::binary);if(!file){error="cannot read template library package: "+config.id;return false;}
        file.seekg(0, std::ios::end);
        const auto length = file.tellg();
        if (length < 0 || static_cast<uint64_t>(length) > max_package_bytes) {
            error="template library package exceeds limit or cannot be sized"; return false;
        }
        file.seekg(0);
        std::string input(static_cast<size_t>(length), '\0');
        if (!input.empty() && !file.read(input.data(), static_cast<std::streamsize>(input.size()))) {
            error="template library package read failed"; return false;
        }
        if(input.size()>max_package_bytes){error="template library package exceeds limit";return false;}
        common_agent_bootstrap_package package;
        if(!common_agent_package_parse_json(input,package,error))return false;
        std::string canonical;if(!common_agent_package_to_json(package,canonical,error,false))return false;
        if (canonical.size() > max_package_bytes) { error="normalized library package exceeds limit"; return false; }
        common_agent_template_library_manifest m;m.library_id=config.id;m.level=config.level;
        m.namespace_id=config.namespace_id;m.project_id=config.project_id;m.package_name=package.name;m.package_version=package.version;
        m.fingerprint=fingerprint(config,canonical);m.definition_prefix="library:"+config.id+":"+m.fingerprint+":";
        for(const auto & b:package.blueprints)m.blueprints[b.id]=m.definition_prefix+"blueprint:"+b.id;
        for(const auto & w:package.workflows)m.workflows.insert(m.definition_prefix+"workflow:"+w.id);
        for(const auto & p:package.procedures)m.procedures.insert(m.definition_prefix+"procedure:"+p.id);
        common_agent_bootstrap_config c=installation(m);common_agent_bootstrap_result receipt;
        if(!common_agent_install_bootstrap_package(memories,plans,c,package,embed,receipt,error)||
                !receipt.rejected_items.empty()||!verify(package,m,memories,plans,error)){
            if(error.empty())error="template library import rejected a definition";return false;
        }
        bool found=false;
        for(const auto & resource:prior){
            if(resource.source_provider!="builtin.template-library"||resource.source_tool!="host-import")continue;
            std::string bytes;if(!resources.read_text(resource.uri,authority,max_receipt_bytes,bytes,error))return false;
            auto envelope=json::parse(bytes,nullptr,false);
            if(!envelope.is_object()||!envelope.contains("manifest")||!envelope.contains("package")){error="invalid host library receipt";return false;}
            common_agent_template_library_manifest historical;
            if(!common_agent_template_library_manifest_parse(envelope["manifest"].dump(),historical,error))return false;
            if(historical.library_id!=config.id || historical.level!=config.level || historical.namespace_id!=config.namespace_id || historical.project_id!=config.project_id)continue;
            common_agent_bootstrap_package old;
            std::string old_canonical;
            if(!common_agent_package_parse_json(envelope["package"].dump(),old,error)||
                    !common_agent_package_to_json(old,old_canonical,error,false)||
                    fingerprint(config,old_canonical)!=historical.fingerprint){
                if(error.empty())error="historical library receipt failed verification";return false;
            }
            // Resource receipts survive an in-memory daemon restart. Restore
            // only through the ordinary scoped installer, after validating the
            // receipt, and only when the caller opened both stores fresh.
            // Persistent/live stores must still expose missing definitions as
            // integrity failures, rather than silently repairing corruption.
            if (restore_fresh_ephemeral_definitions) {
                common_agent_bootstrap_result restored;
                if (!common_agent_install_bootstrap_package(memories,plans,installation(historical),
                        old,embed,restored,error) || !restored.rejected_items.empty()) {
                    if(error.empty())error="historical library restoration rejected a definition";
                    return false;
                }
            }
            if(!verify(old,historical,memories,plans,error)) return false;
            historical.resource_uri=resource.uri;historical.current=historical.fingerprint==m.fingerprint;
            if(historical.current){found=true;m.resource_uri=resource.uri;continue;}
            const bool duplicate=std::any_of(staged.revisions.begin(),staged.revisions.end(),[&](const auto & r){return r.library_id==historical.library_id&&r.fingerprint==historical.fingerprint;});
            if(!duplicate)staged.revisions.push_back(std::move(historical));
        }
        if(!found){
            agent_resource_put_request put;put.name="Template library "+config.id+" "+package.version;
            put.mime_type="application/vnd.llama-agent.template-library+json";
            put.namespace_id=authority.namespace_id;put.session_id=authority.session_id;put.scope=common_runtime_resource_scope::session;
            put.source_provider="builtin.template-library";put.source_tool="host-import";put.created_at=authority.now;put.expires_at=0;
            put.text=json{{"manifest",json::parse(common_agent_template_library_manifest_json(m))},{"package",json::parse(canonical)}}.dump();
            agent_resource_descriptor stored;
            if(!resources.put_text(put,stored,error))return false;
            std::string reread;if(!resources.read_text(stored.uri,authority,max_receipt_bytes,reread,error)||reread!=put.text){error="library receipt readback failed";return false;}
            m.resource_uri=stored.uri;
        }
        for(const auto & b:m.blueprints){auto plan=plans.get(b.second,error);if(!plan)return false;staged_candidates.push_back(common_agent_template_candidate(config.id+":"+b.first,*plan));}
        staged.revisions.push_back(std::move(m));
    }
    for(const auto & m:staged.revisions){
        for(const auto & b:m.blueprints){auto p=plans.get(b.second,error);if(!p)return false;staged.definitions.emplace(p->id,*p);}
        for(const auto & id:m.workflows){auto p=plans.get(id,error);if(!p)return false;staged.definitions.emplace(id,*p);}
        for(const auto & id:m.procedures){auto p=memories.get(id,error);if(!p)return false;staged.procedures.emplace(id,*p);}
    }
    snapshot=std::move(staged);candidates=std::move(staged_candidates);return true;
}
