#include "reflection-references.h"
#include <algorithm>
#include <cctype>
#include <set>

common_reflection_evidence_references::common_reflection_evidence_references(const common_plan_state & plan) {
    for (const auto & observation : plan.observations) {
        if (observation.source != "blueprint_outline") continue;
        const auto outline=nlohmann::ordered_json::parse(observation.summary,nullptr,false);
        if (!outline.is_array()) break;
        for (const auto & phase:outline) if (phase.is_object() && phase.contains("id") && phase["id"].is_string()) {
            const auto id=phase["id"].get<std::string>();
            if (!id.empty()) phase_handles.emplace("p"+std::to_string(phase_handles.size()+1),id);
        }
        break;
    }
    // A validated checkpoint can expose its predecessor IDs as historical
    // context. Resolve only exact phase keys from the same method revision.
    std::map<std::string,std::string> origins;
    for (const auto & observation:plan.observations) {
        if (observation.source!="blueprint_outline") continue;
        const auto outline=nlohmann::ordered_json::parse(observation.summary,nullptr,false);
        if (outline.is_array()) for (const auto & phase:outline)
            if (phase.is_object() && phase.contains("origin_phase_id") && phase["origin_phase_id"].is_string() &&
                    phase.contains("id") && phase["id"].is_string())
                origins[phase["origin_phase_id"].get<std::string>()]=phase["id"].get<std::string>();
        break;
    }
    for (const auto & observation:plan.observations) {
        if (observation.source!="prior_work_checkpoint") continue;
        const auto checkpoint=nlohmann::ordered_json::parse(observation.summary,nullptr,false);
        if (!checkpoint.is_object() || checkpoint.value("schema",std::string{})!="prior-work.v2" ||
                !checkpoint.contains("method") || !checkpoint["method"].is_object() ||
                checkpoint["method"].value("revision",std::string{})!=plan.source_revision ||
                checkpoint["method"].value("blueprint_id",std::string{})!=plan.derived_from_plan_id.value_or(std::string{}) ||
                !checkpoint.contains("method_progress") || !checkpoint["method_progress"].is_string()) continue;
        const auto text=checkpoint["method_progress"].get<std::string>();
        const auto begin=text.find("\n{");
        if (begin==std::string::npos) continue;
        const auto end=text.find('\n',begin+1);
        const auto progress=nlohmann::ordered_json::parse(text.substr(begin+1,end-begin-1),nullptr,false);
        if (!progress.is_object() || !progress.contains("outline") || !progress["outline"].is_array()) continue;
        for (const auto & phase:progress["outline"]) {
            if (!phase.is_object() || !phase.contains("id") || !phase["id"].is_string() ||
                    !phase.contains("origin_phase_id") || !phase["origin_phase_id"].is_string()) continue;
            const auto found=origins.find(phase["origin_phase_id"].get<std::string>());
            if (found!=origins.end()) prior_phase_aliases[phase["id"].get<std::string>()]=found->second;
        }
    }
    std::set<std::string> seen;
    for (const auto & observation:plan.observations) {
        if (!observation.id.empty() && seen.insert(observation.id).second)
            handles.emplace("e"+std::to_string(handles.size()+1),observation.id);
    }
}

std::string common_reflection_evidence_references::project(const std::string & text) const {
    struct replacement {size_t start,length;std::string handle;};
    std::vector<replacement> found;
    const auto id_char=[](unsigned char c) {
        return std::isalnum(c) || c=='_' || c==':' || c=='-' || c=='.' || c=='/';
    };
    for (const auto & item:handles) {
        size_t position=0;
        while ((position=text.find(item.second,position))!=std::string::npos) {
            const auto end=position+item.second.size();
            if ((position==0 || !id_char(text[position-1])) && (end==text.size() || !id_char(text[end])))
                found.push_back({position,item.second.size(),item.first});
            position=end;
        }
    }
    std::sort(found.begin(),found.end(),[](const auto & a,const auto & b) {
        return a.start<b.start || (a.start==b.start && a.length>b.length);
    });
    std::string out;size_t position=0;
    for (const auto & match:found) {
        if (match.start<position) continue;
        out.append(text,position,match.start-position);out+=match.handle;
        position=match.start+match.length;
    }
    out.append(text,position,std::string::npos);return out;
}

std::string common_reflection_evidence_references::project_context(const std::string & text) const {
    using json=nlohmann::ordered_json;
    const auto alias=[&](const std::string & id) -> std::string {
        for (const auto & item:handles) if (item.second==id) return item.first;
        return id;
    };
    const auto fields=[&](auto && self,json & value) -> void {
        if (value.is_object()) {
            for (auto & item:value.items()) {
                if (item.key()=="evidence_ids" && item.value().is_array()) {
                    for (auto & id:item.value()) if (id.is_string()) id=alias(id.get<std::string>());
                } else if((item.key()=="observation_id" || item.key()=="evidence_id") && item.value().is_string()) {
                    item.value()=alias(item.value().get<std::string>());
                } else self(self,item.value());
            }
        } else if (value.is_array()) for (auto & item:value) self(self,item);
    };
    std::string result;
    size_t position=0;
    while (position<text.size()) {
        const auto end=text.find('\n',position);
        auto line=text.substr(position,end==std::string::npos?std::string::npos:end-position);
        if (line.rfind("  evidence: ",0)==0) {
            const auto value=json::parse(line.substr(12),nullptr,false);
            if (value.is_string()) line="  evidence: "+json(alias(value.get<std::string>())).dump();
        } else if (line.rfind("- host-validated workflow choice=",0)==0) {
            const std::string key="; source_evidence=";
            const auto start=line.find(key);
            if (start!=std::string::npos) {
                const auto value_start=start+key.size();
                for (const auto & item:handles) {
                    const auto value=json(item.second).dump();
                    const auto end=value_start+value.size();
                    if (line.compare(value_start,value.size(),value)==0 && (end==line.size() || line[end]==';')) {
                        line.replace(value_start,value.size(),json(item.first).dump());break;
                    }
                }
            }
        } else if (line.rfind("Observation ",0)==0) {
            const auto delimiter=line.find(": ",12);
            if (delimiter!=std::string::npos)
                line="Observation "+alias(line.substr(12,delimiter-12))+line.substr(delimiter);
        } else if (!line.empty() && (line.front()=='{' || line.front()=='[')) {
            auto value=json::parse(line,nullptr,false);
            if (value.is_object() || value.is_array()) {fields(fields,value);line=value.dump();}
        }
        result+=line;
        if (end==std::string::npos) break;
        result+='\n';position=end+1;
    }
    return result;
}

std::string common_reflection_evidence_references::project_method(const std::string & input) const {
    auto result=project_context(input);
    for (const auto & entry:phase_handles) {
        const auto original=nlohmann::ordered_json(entry.second).dump();
        const auto replacement=nlohmann::ordered_json(entry.first).dump();
        size_t position=0;
        while ((position=result.find(original,position))!=std::string::npos) {
            result.replace(position,original.size(),replacement);position+=replacement.size();
        }
    }
    return result;
}

bool common_reflection_evidence_references::restore(common_reflection_result & review,std::string & error) const {
    auto candidate=review;
    const auto restore_ids=[&](std::vector<std::string> & ids) {
        for (auto & id:ids) {
            const auto found=handles.find(id);
            if (found!=handles.end()) {id=found->second;continue;}
            // Some tool results expose the canonical identity as well as the
            // compact handle. Only an exact issued identity is equivalent.
            if(std::none_of(handles.begin(),handles.end(),[&](const auto & entry){return entry.second==id;})) {
                error="unknown review evidence handle: "+id.substr(0,64);return false;
            }
        }
        return true;
    };
    for(auto & part:candidate.parts) {
        std::vector<std::string> target{part.observation_id};
        if(!restore_ids(target) || !restore_ids(part.evidence_ids)) return false;
        part.observation_id=target.front();
    }
    for (auto & phase:candidate.phase_reviews) {
        const auto found=phase_handles.find(phase.phase_id);
        if (found!=phase_handles.end()) phase.phase_id=found->second;
        else if (const auto prior=prior_phase_aliases.find(phase.phase_id); prior!=prior_phase_aliases.end())
            phase.phase_id=prior->second;
        if (!restore_ids(phase.evidence_ids)) return false;
    }
    for (auto & issue:candidate.issues) if (!restore_ids(issue.evidence_ids)) return false;
    for (auto & operation:candidate.proposed_plan_operations) {
        if (!restore_ids(operation.evidence_ids)) return false;
        if (operation.assumption && !restore_ids(operation.assumption->evidence_ids)) return false;
    }
    review=std::move(candidate);error.clear();return true;
}
