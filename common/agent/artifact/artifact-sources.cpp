#include "artifact-sources.h"

#include <nlohmann/json.hpp>
#include <algorithm>
#include <cctype>
#include <functional>
#include <map>
#include <set>

namespace {
using json = nlohmann::ordered_json;

std::string string_field(const json & value, const char * key) {
    const auto it=value.find(key);
    return it!=value.end()&&it->is_string()?it->get<std::string>():std::string{};
}

bool web_uri(const std::string & value) {
    const size_t begin=value.rfind("https://",0)==0?8:value.rfind("http://",0)==0?7:0;
    if(!begin||value.size()<=begin)return false;
    const auto authority=value.substr(begin,value.find_first_of("/?#",begin)-begin);
    return !authority.empty()&&authority.find('@')==std::string::npos&&
        std::none_of(value.begin(),value.end(),[](unsigned char c){return c<=32||c==127;});
}

std::string basename(const std::string & name) {
    const auto last=name.find_last_of("/\\");
    return last==std::string::npos?name:name.substr(last+1);
}

std::string markdown_text(const std::string & value) {
    std::string out;
    for(char c:value){
        if(c=='\n'||c=='\r'){out+=' ';continue;}
        if(c=='<'||c=='>'){out+=c=='<'?"&lt;":"&gt;";continue;}
        if(c=='\\'||c=='['||c==']'||c=='*'||c=='_'||c=='`')out+='\\';
        out+=c;
    }
    return out;
}

std::string link(const std::string & label,const std::string & uri) {
    // Angle-bracket destinations support parentheses in URLs. Encode the
    // delimiter characters rather than allowing metadata to close the link.
    std::string escaped;
    for(unsigned char c:uri){
        if(c=='<'||c=='>'||c=='\\'||c<=32||c==127){
            constexpr char hex[]="0123456789ABCDEF";
            escaped+='%';escaped+=hex[c>>4];escaped+=hex[c&15];
        }else escaped+=static_cast<char>(c);
    }
    return "["+label+"](<"+escaped+">)";
}

bool ignored_parameter(const std::string & name) {
    std::string key;
    for(unsigned char c:name)if(c!='_'&&c!='-')key+=static_cast<char>(std::tolower(c));
    static const std::set<std::string> ignored={"url","finalurl","requestedurl","headers","body",
        "authorization","auth","password","secret","apikey","key","token","accesstoken",
        "credential","credentials","cursor","nextcursor","nexttoken","continuationtoken",
        "page","pagesize","perpage","limit","offset","maxbytes","timeout","timeoutms","accept"};
    return ignored.count(key)||key.find("secret")!=std::string::npos||
        key.find("password")!=std::string::npos||key.find("token")!=std::string::npos;
}

std::string request_summary(const std::string & request) {
    // This is the host-retained request, not model-generated report text.
    if(request.size()>65536)return {};
    const auto value=json::parse(request,nullptr,false);
    if(!value.is_object())return {};
    std::vector<std::pair<std::string,std::string>> fields;
    bool omitted=false;
    std::function<void(const json &,const std::string &,size_t)> visit;
    visit=[&](const json & object,const std::string & prefix,size_t depth){
        for(auto it=object.begin();it!=object.end();++it){
            if(ignored_parameter(it.key()))continue;
            const auto name=prefix.empty()?it.key():prefix+"."+it.key();
            if(fields.size()>=12||name.size()>80){omitted=true;continue;}
            if(it->is_object()){
                if(depth<3)visit(*it,name,depth+1);else omitted=true;
                continue;
            }
            if(it->is_null())continue;
            if(it->is_array()&&!std::all_of(it->begin(),it->end(),[](const json & v){return v.is_primitive();})){
                omitted=true;continue;
            }
            const auto text=it->dump();
            if(text.size()>200){omitted=true;continue;}
            fields.emplace_back(name,text);
        }
    };
    visit(value,{},0);
    std::sort(fields.begin(),fields.end());
    std::string out;
    for(const auto & field:fields){
        const auto text=field.first+": "+field.second;
        if(out.size()+text.size()+2>1024){omitted=true;continue;}
        if(!out.empty())out+="; ";
        out+=text;
    }
    if(omitted)out+=(out.empty()?"":"; ")+std::string("additional request arguments omitted");
    return out;
}

void apply_metadata(common_agent_artifact_source_citation & source,
        const common_runtime_resource_metadata & metadata) {
    if(source.source_provider.empty())source.source_provider=metadata.source_provider;
    if(source.source_operation.empty())source.source_operation=metadata.source_operation;
    if(source.request_summary.empty())source.request_summary=request_summary(metadata.source_request_json);
    if(source.retrieved_at.empty()&&metadata.retrieved_at>0)source.retrieved_at=std::to_string(metadata.retrieved_at);
    const auto request=json::parse(metadata.source_request_json,nullptr,false);
    if(request.is_object()&&source.original_uri.empty()){
        const auto url=string_field(request,"url");
        if(web_uri(url))source.original_uri=url;
    }
}
} // namespace

std::vector<common_agent_artifact_source_citation> common_agent_artifact_collect_sources(
        const common_agent_artifact_snapshot & snapshot) {
    std::map<std::string,common_runtime_resource_ref> resources;
    std::map<std::string,json> fetched;
    for(const auto & source:snapshot.research_sources)if(source.resource_ref)resources.emplace(source.resource_ref->uri,*source.resource_ref);
    for(const auto & observation:snapshot.observations){
        for(const auto & resource:observation.resource_refs){
            auto inserted=resources.emplace(resource.uri,resource);
            if(!inserted.second){
                auto & previous=inserted.first->second;
                if(previous.name.empty())previous.name=resource.name;
                if(previous.lineage.parent_uri.empty())previous.lineage=resource.lineage;
                if(previous.metadata.source_provider.empty())previous.metadata.source_provider=resource.metadata.source_provider;
                if(previous.metadata.source_operation.empty())previous.metadata.source_operation=resource.metadata.source_operation;
                if(previous.metadata.source_request_json.empty())previous.metadata.source_request_json=resource.metadata.source_request_json;
                if(previous.metadata.content_hash.empty())previous.metadata.content_hash=resource.metadata.content_hash;
                if(previous.metadata.retrieved_at==0)previous.metadata.retrieved_at=resource.metadata.retrieved_at;
            }
        }
        if(observation.source!="web.fetch"||observation.confidence<=0)continue;
        auto result=json::parse(observation.summary,nullptr,false);
        if(!result.is_object())continue;
        if(result.contains("result")&&result["result"].is_object())result=result["result"];
        if(result.contains("structuredContent")&&result["structuredContent"].is_object())result=result["structuredContent"];
        const auto status=result.find("status");
        if(status==result.end()||!status->is_number_integer()||status->get<int>()<200||status->get<int>()>=300)continue;
        const auto uri=string_field(result,"resource_ref");
        if(!uri.empty()&&std::any_of(observation.resource_refs.begin(),observation.resource_refs.end(),
                [&](const auto & resource){return resource.uri==uri;}))fetched[uri]=std::move(result);
    }
    auto enrich=[&](common_agent_artifact_source_citation & citation,const std::string & first){
        std::set<std::string> visited;
        auto uri=first;
        while(!uri.empty()&&visited.size()<128&&visited.insert(uri).second){
            const auto resource=resources.find(uri);
            if(resource==resources.end())break;
            if(citation.file_name.empty())citation.file_name=basename(resource->second.name);
            if(uri==first&&citation.content_hash.empty())citation.content_hash=resource->second.metadata.content_hash;
            apply_metadata(citation,resource->second.metadata);
            const auto fetch=fetched.find(uri);
            if(fetch!=fetched.end()){
                const auto requested=string_field(fetch->second,"url"),resolved=string_field(fetch->second,"final_url");
                if(web_uri(requested))citation.original_uri=requested;
                if(web_uri(resolved))citation.resolved_uri=resolved;
                const auto title=string_field(fetch->second,"title");
                if(citation.title.empty()&&!title.empty())citation.title=title;
            }
            uri=resource->second.lineage.parent_uri;
        }
    };
    std::map<std::string,common_agent_artifact_source_citation> citations;
    for(const auto & source:snapshot.research_sources){
        auto & citation=citations[source.source_id];
        citation.source_id=source.source_id;citation.title=source.title;citation.origin=source.origin;
        citation.retrieved_at=source.retrieved_at;citation.content_hash=source.content_hash;
        if(web_uri(source.origin))citation.original_uri=source.origin;
        if(source.resource_ref){
            citation.resource_uri=source.resource_ref->uri;
            apply_metadata(citation,source.resource_ref->metadata);
            enrich(citation,citation.resource_uri);
        }
    }
    for(const auto & evidence:snapshot.research_evidence){
        auto & citation=citations[evidence.source_id];citation.source_id=evidence.source_id;
        if(std::find(citation.evidence_ids.begin(),citation.evidence_ids.end(),evidence.evidence_id)==citation.evidence_ids.end())citation.evidence_ids.push_back(evidence.evidence_id);
        if(!evidence.source_location.empty()&&std::find(citation.locations.begin(),citation.locations.end(),evidence.source_location)==citation.locations.end())citation.locations.push_back(evidence.source_location);
    }
    for(const auto & observation:snapshot.observations){
        for(const auto & resource:observation.resource_refs){
            auto existing=std::find_if(citations.begin(),citations.end(),[&](const auto & entry){
                return entry.second.resource_uri==resource.uri&&(entry.second.content_hash.empty()||
                    resource.metadata.content_hash.empty()||entry.second.content_hash==resource.metadata.content_hash);
            });
            auto & citation=existing==citations.end()?citations[resource.uri]:existing->second;
            if(citation.source_id.empty())citation.source_id=resource.uri;
            citation.resource_uri=resource.uri;enrich(citation,resource.uri);
        }
        for(const auto & dataset:observation.dataset_refs){
            auto & citation=citations[dataset.uri];citation.source_id=dataset.uri;citation.title=dataset.name;
            citation.resource_uri=dataset.uri;citation.content_hash=dataset.content_hash;
            citation.source_provider=dataset.source_provider;citation.source_operation=dataset.source_operation;
            citation.request_summary=request_summary(dataset.source_request_json);
            const auto request=json::parse(dataset.source_request_json,nullptr,false);
            if(request.is_object()&&web_uri(string_field(request,"url")))citation.original_uri=string_field(request,"url");
            if(citation.original_uri.empty()&&web_uri(dataset.source_resource_uri))citation.original_uri=dataset.source_resource_uri;
            if(dataset.retrieved_at>0)citation.retrieved_at=std::to_string(dataset.retrieved_at);
            enrich(citation,dataset.source_resource_uri);
        }
    }
    std::vector<common_agent_artifact_source_citation> out;
    for(auto & entry:citations)out.push_back(std::move(entry.second));
    return out;
}

std::string common_agent_artifact_render_source(const common_agent_artifact_source_citation & source) {
    auto title=source.title;
    if(title.empty()||title==source.source_operation||title==source.origin||
            title==source.source_provider+"."+source.source_operation){
        title=source.file_name;
        if(title.empty()&&!source.source_provider.empty())title=source.source_provider+
            (source.source_operation.empty()?"":" / "+source.source_operation);
        if(title.empty())title=source.original_uri.empty()?source.source_id:source.original_uri;
    }
    std::string out="- "+markdown_text(title);
    if(web_uri(source.original_uri))out+=" — "+link("Original",source.original_uri);
    if(web_uri(source.resolved_uri)&&source.resolved_uri!=source.original_uri)out+=" · "+link("Retrieved from",source.resolved_uri);
    if(!source.resource_uri.empty())out+=" · "+link("Saved source",source.resource_uri);
    if(!source.origin.empty()&&!web_uri(source.origin))out+=" ("+markdown_text(source.origin)+")";
    if(!source.request_summary.empty())out+="  \n  Request: "+markdown_text(source.request_summary);
    if(!source.retrieved_at.empty())out+="; retrieved "+markdown_text(source.retrieved_at);
    if(!source.content_hash.empty())out+="; fingerprint "+markdown_text(source.content_hash);
    if(!source.locations.empty()){
        out+="; locations: ";
        for(size_t i=0;i<source.locations.size();++i){if(i)out+=", ";out+=markdown_text(source.locations[i]);}
    }
    return out+"\n";
}
