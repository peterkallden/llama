#include "artifact-contract.h"

#include <nlohmann/json.hpp>
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <set>

using json = nlohmann::ordered_json;

namespace {
std::string fingerprint(const std::string & value) {
    // Stable non-cryptographic identity only. Resource content hashes remain
    // the integrity authority; this key is for snapshot equivalence.
    uint64_t h = 1469598103934665603ULL;
    for (unsigned char c : value) { h ^= c; h *= 1099511628211ULL; }
    char text[17];
    std::snprintf(text, sizeof(text), "%016llx", static_cast<unsigned long long>(h));
    return text;
}
std::string snapshot_fingerprint(const common_agent_artifact_snapshot & snapshot) {
    json material = {{"plan", snapshot.source_plan_id}, {"profile", snapshot.profile_id},
        {"profile_revision", snapshot.profile_revision}, {"observations", json::array()}, {"assumptions", json::array()}};
    for (const auto & observation : snapshot.observations) {
        json refs = json::array();
        for (const auto & ref : observation.resource_refs) refs.push_back(ref.uri);
        material["observations"].push_back({{"id", observation.id}, {"source", observation.source},
            {"summary", observation.summary}, {"evidence_ids", observation.evidence_ids},
            {"resource_refs", std::move(refs)}});
    }
    for (const auto & assumption : snapshot.assumptions) material["assumptions"].push_back({
        {"id", assumption.id}, {"statement", assumption.statement}, {"valid", assumption.valid},
        {"confidence", assumption.confidence}});
    return fingerprint(material.dump());
}
const char * block_kind(common_agent_artifact_block_kind v) {
    switch (v) { case common_agent_artifact_block_kind::observation: return "observation";
        case common_agent_artifact_block_kind::conclusion: return "conclusion";
        case common_agent_artifact_block_kind::recommendation: return "recommendation"; }
    return "observation";
}
bool parse_kind(const std::string & s, common_agent_artifact_block_kind & v) {
    if (s == "observation") v = common_agent_artifact_block_kind::observation;
    else if (s == "conclusion") v = common_agent_artifact_block_kind::conclusion;
    else if (s == "recommendation") v = common_agent_artifact_block_kind::recommendation;
    else return false;
    return true;
}
const char * validation_name(common_agent_artifact_validation v) {
    switch (v) { case common_agent_artifact_validation::unvalidated: return "unvalidated";
        case common_agent_artifact_validation::references_resolve: return "references_resolve";
        case common_agent_artifact_validation::gap: return "gap";
        case common_agent_artifact_validation::contradicted: return "contradicted"; }
    return "unvalidated";
}
bool parse_validation(const std::string & s, common_agent_artifact_validation & v) {
    if (s == "unvalidated") v = common_agent_artifact_validation::unvalidated;
    else if (s == "references_resolve") v = common_agent_artifact_validation::references_resolve;
    else if (s == "gap") v = common_agent_artifact_validation::gap;
    else if (s == "contradicted") v = common_agent_artifact_validation::contradicted;
    else return false;
    return true;
}
}

bool common_agent_artifact_profile_valid(const common_agent_artifact_profile & p, std::string & error) {
    if (p.id.empty() || p.id.size() > 128 || p.revision.empty() || p.revision.size() > 128 ||
            p.title.empty() || p.title.size() > 256 || p.purpose.size() > 2048 ||
            p.sections.empty() || p.sections.size() > 32) { error = "artifact profile has invalid identity or bounds"; return false; }
    std::set<std::string> ids;
    for (const auto & section : p.sections) {
        if (section.id.empty() || section.id.size() > 128 || section.title.empty() ||
                section.title.size() > 256 || section.purpose.size() > 2048 || !ids.insert(section.id).second) {
            error = "artifact profile contains an invalid or duplicate section"; return false;
        }
    }
    error.clear(); return true;
}

bool common_agent_artifact_profile_from_json(const std::string & text, common_agent_artifact_profile & out, std::string & error) {
    const auto value=json::parse(text,nullptr,false);
    if(!value.is_object()||!value.contains("sections")||!value["sections"].is_array()){error="artifact profile JSON is malformed";return false;}
    out={};out.id=value.value("id",std::string{});out.revision=value.value("revision",std::string{});out.title=value.value("title",std::string{});out.purpose=value.value("purpose",std::string{});
    for(const auto & item:value["sections"]){if(!item.is_object()){error="artifact profile section is malformed";return false;}common_agent_artifact_section_profile section;section.id=item.value("id",std::string{});section.title=item.value("title",std::string{});section.purpose=item.value("purpose",std::string{});for(const auto & source:item.value("required_observation_sources",json::array())){if(!source.is_string()){error="artifact profile source must be text";return false;}section.required_observation_sources.push_back(source.get<std::string>());}out.sections.push_back(std::move(section));}
    return common_agent_artifact_profile_valid(out,error);
}

bool common_agent_artifact_snapshot_from_plan(const common_plan_state & plan,
        const common_agent_artifact_profile & profile, common_agent_artifact_snapshot & out,
        std::string & error) {
    if (!common_agent_artifact_profile_valid(profile, error)) return false;
    if (plan.id.empty() || plan.version == 0) { error = "artifact snapshot requires a persisted plan revision"; return false; }
    for (const auto & observation : plan.observations) {
        if (observation.source != "artifact_snapshot") continue;
        common_agent_artifact_snapshot frozen;
        if (!common_agent_artifact_snapshot_from_json(observation.summary, frozen, error)) return false;
        if (frozen.source_plan_id != plan.id || frozen.profile_id != profile.id ||
                frozen.profile_revision != profile.revision) {
            error = "frozen artifact snapshot does not match the active plan and profile revision";
            return false;
        }
        out = std::move(frozen);
        error.clear();
        return true;
    }
    out = {};
    out.id = "snapshot:" + plan.id;
    out.source_plan_id = plan.id; out.plan_version = plan.version;
    out.profile_id = profile.id; out.profile_revision = profile.revision;
    out.assumptions = plan.assumptions;
    bool coverage_complete = !plan.observations.empty();
    for (const auto & o : plan.observations) {
        if (o.source == "artifact_section" || o.source == "artifact_snapshot") continue;
        out.observations.push_back(o);
        const auto parsed = json::parse(o.summary, nullptr, false);
        if (parsed.is_object() && parsed.contains("coverage_complete") && parsed.value("coverage_complete", false) == false) coverage_complete = false;
        if (parsed.is_object() && parsed.contains("unresolved_claims") && parsed["unresolved_claims"].is_array())
            for (const auto & id : parsed["unresolved_claims"]) if (id.is_string()) out.unresolved_gaps.push_back(id.get<std::string>());
    }
    for (const auto & section : profile.sections) {
        for (const auto & required_source : section.required_observation_sources) {
            const bool present = std::any_of(out.observations.begin(), out.observations.end(), [&](const auto & observation) {
                return observation.source == required_source;
            });
            if (!present) out.unresolved_gaps.push_back("required_source:" + required_source);
        }
    }
    out.assumptions = plan.assumptions;
    out.coverage_complete = coverage_complete && !out.observations.empty() && out.unresolved_gaps.empty();
    out.fingerprint = snapshot_fingerprint(out); error.clear(); return true;
}

bool common_agent_artifact_validate_block(const common_agent_artifact_block & block,
        const common_agent_artifact_snapshot & snapshot, common_agent_artifact_block & out,
        std::string & error) {
    out = block;
    if (block.id.empty() || block.id.size() > 128 || block.text.empty() || block.text.size() > 12000 ||
            block.evidence_ids.size() > 64 || block.source_refs.size() > 64) {
        error = "artifact block violates size or identity bounds"; return false;
    }
    std::set<std::string> evidence, sources;
    for (const auto & o : snapshot.observations) {
        for (const auto & id : o.evidence_ids) evidence.insert(id);
        for (const auto & ref : o.resource_refs) if (!ref.uri.empty()) sources.insert(ref.uri);
        if (!o.id.empty()) sources.insert(o.id);
    }
    bool refs_valid = true;
    for (const auto & id : block.evidence_ids) refs_valid = refs_valid && evidence.count(id);
    for (const auto & ref : block.source_refs) refs_valid = refs_valid && sources.count(ref);
    out.validation = !refs_valid ? common_agent_artifact_validation::gap :
        block.evidence_ids.empty() && block.source_refs.empty() ? common_agent_artifact_validation::gap :
        common_agent_artifact_validation::references_resolve;
    out.validation_note = !refs_valid ? "One or more references are absent from the source snapshot." :
        block.evidence_ids.empty() && block.source_refs.empty() ? "No source or evidence reference was supplied." :
        "References resolve in the snapshot; this does not independently prove semantic entailment.";
    if (!refs_valid) { error.clear(); }
    else error.clear();
    return true;
}

bool common_agent_artifact_parse_section_dsl(const std::string & text,
        const common_agent_artifact_section_profile & profile,
        const common_agent_artifact_snapshot & snapshot, common_agent_artifact_section & out,
        std::string & error) {
    out = {}; out.id = profile.id; out.title = profile.title;
    common_agent_artifact_block block; bool active = false; size_t count = 0;
    auto flush = [&]() -> bool {
        if (!active) return true;
        common_agent_artifact_block checked;
        if (!common_agent_artifact_validate_block(block, snapshot, checked, error)) return false;
        out.blocks.push_back(std::move(checked)); block = {}; active = false; ++count; return true;
    };
    size_t begin = 0;
    while (begin <= text.size()) {
        const auto end = text.find('\n', begin);
        auto line = text.substr(begin, end == std::string::npos ? std::string::npos : end - begin);
        if (!line.empty() && line.back() == '\r') line.pop_back();
        const auto colon = line.find(':');
        if (colon != std::string::npos) {
            auto key = line.substr(0, colon), value = line.substr(colon + 1);
            while (!value.empty() && value.front() == ' ') value.erase(value.begin());
            if (key == "block") {
                if (!flush()) return false;
                active = true; block.id = profile.id + ":" + std::to_string(count + 1);
                if (!parse_kind(value, block.kind)) { error = "artifact block kind must be observation, conclusion or recommendation"; return false; }
            } else if (key == "text") {
                if (!active) { error="artifact section field appears before block";return false; }
                block.text = value;
            } else if (key == "evidence" || key == "source") {
                if (!active) { error="artifact section field appears before block";return false; }
                size_t item_begin=0; while(item_begin<=value.size()){const auto item_end=value.find(',',item_begin);auto ref=value.substr(item_begin,item_end==std::string::npos?std::string::npos:item_end-item_begin);while(!ref.empty()&&ref.front()==' ')ref.erase(ref.begin());while(!ref.empty()&&ref.back()==' ')ref.pop_back();if(!ref.empty())(key=="evidence"?block.evidence_ids:block.source_refs).push_back(std::move(ref));if(item_end==std::string::npos)break;item_begin=item_end+1;}
            } else if (!line.empty()) { error="unknown artifact section field: "+key;return false; }
        } else if (!line.empty()) { error="artifact section expects flat key: value lines";return false; }
        if (end == std::string::npos) break;
        begin = end + 1;
    }
    if (!flush()) return false;
    if (out.blocks.empty() || out.blocks.size() > 32) { error="artifact section must contain 1 to 32 blocks";return false; }
    error.clear(); return true;
}

bool common_agent_artifact_document_to_json(const common_agent_artifact_document & d, std::string & out, std::string & error) {
    json value = {{"schema_version",1},{"id",d.id},{"title",d.title},{"purpose",d.purpose},
        {"profile_id",d.profile_id},{"profile_revision",d.profile_revision},{"revision",d.revision},
        {"previous_revision_uri",d.previous_revision_uri},{"snapshot_uri",d.snapshot_uri},
        {"source_plan_id",d.source_plan_id},{"source_plan_version",d.source_plan_version},
        {"coverage_complete",d.coverage_complete},{"unresolved_gaps",d.unresolved_gaps},{"sections",json::array()}};
    if (d.id.empty() || d.title.empty() || d.sections.size() > 32) { error="artifact document has invalid identity or section count"; return false; }
    for (const auto & s : d.sections) {
        json section={{"id",s.id},{"title",s.title},{"blocks",json::array()}};
        for (const auto & b : s.blocks) section["blocks"].push_back({{"id",b.id},{"kind",block_kind(b.kind)},
            {"text",b.text},{"evidence_ids",b.evidence_ids},{"source_refs",b.source_refs},
            {"validation",validation_name(b.validation)},{"validation_note",b.validation_note},{"origin",b.origin}});
        value["sections"].push_back(std::move(section));
    }
    out=value.dump(); error.clear(); return true;
}

bool common_agent_artifact_snapshot_to_json(const common_agent_artifact_snapshot & s, std::string & out, std::string & error) {
    json value={{"schema_version",1},{"id",s.id},{"source_plan_id",s.source_plan_id},{"plan_version",s.plan_version},
        {"profile_id",s.profile_id},{"profile_revision",s.profile_revision},{"fingerprint",s.fingerprint},
        {"coverage_complete",s.coverage_complete},{"unresolved_gaps",s.unresolved_gaps},{"observations",json::array()},{"assumptions",json::array()}};
    for(const auto & o:s.observations){json refs=json::array();for(const auto & r:o.resource_refs)refs.push_back({{"uri",r.uri},{"name",r.name},{"mime_type",r.mime_type}});value["observations"].push_back({{"id",o.id},{"source",o.source},{"summary",o.summary},{"evidence_ids",o.evidence_ids},{"resource_refs",std::move(refs)}});}
    for(const auto & a:s.assumptions)value["assumptions"].push_back({{"id",a.id},{"statement",a.statement},{"confidence",a.confidence},{"valid",a.valid},{"evidence_ids",a.evidence_ids}});
    out=value.dump();error.clear();return true;
}

bool common_agent_artifact_snapshot_from_json(const std::string & text,
        common_agent_artifact_snapshot & s, std::string & error) {
    const auto value = json::parse(text, nullptr, false);
    if (!value.is_object() || value.value("schema_version", 0) != 1 ||
            !value.contains("observations") || !value["observations"].is_array() ||
            !value.contains("assumptions") || !value["assumptions"].is_array()) {
        error = "unsupported or malformed artifact snapshot";
        return false;
    }
    s = {};
    s.id = value.value("id", std::string{});
    s.source_plan_id = value.value("source_plan_id", std::string{});
    s.plan_version = value.value("plan_version", uint64_t(0));
    s.profile_id = value.value("profile_id", std::string{});
    s.profile_revision = value.value("profile_revision", std::string{});
    s.fingerprint = value.value("fingerprint", std::string{});
    s.coverage_complete = value.value("coverage_complete", false);
    for (const auto & gap : value.value("unresolved_gaps", json::array())) {
        if (!gap.is_string()) { error = "artifact snapshot gap is not text"; return false; }
        s.unresolved_gaps.push_back(gap.get<std::string>());
    }
    for (const auto & item : value["observations"]) {
        if (!item.is_object() || !item.contains("evidence_ids") || !item["evidence_ids"].is_array() ||
                !item.contains("resource_refs") || !item["resource_refs"].is_array()) {
            error = "artifact snapshot observation is malformed";
            return false;
        }
        common_plan_observation observation;
        observation.id = item.value("id", std::string{});
        observation.source = item.value("source", std::string{});
        observation.summary = item.value("summary", std::string{});
        for (const auto & id : item["evidence_ids"]) {
            if (!id.is_string()) { error = "artifact snapshot evidence id is not text"; return false; }
            observation.evidence_ids.push_back(id.get<std::string>());
        }
        for (const auto & ref : item["resource_refs"]) {
            if (!ref.is_object()) { error = "artifact snapshot resource reference is malformed"; return false; }
            common_runtime_resource_ref resource;
            resource.uri = ref.value("uri", std::string{});
            resource.name = ref.value("name", std::string{});
            resource.mime_type = ref.value("mime_type", std::string{});
            observation.resource_refs.push_back(std::move(resource));
        }
        s.observations.push_back(std::move(observation));
    }
    for (const auto & item : value["assumptions"]) {
        if (!item.is_object() || !item.contains("evidence_ids") || !item["evidence_ids"].is_array()) {
            error = "artifact snapshot assumption is malformed";
            return false;
        }
        common_plan_assumption assumption;
        assumption.id = item.value("id", std::string{});
        assumption.statement = item.value("statement", std::string{});
        assumption.confidence = item.value("confidence", 0.5f);
        assumption.valid = item.value("valid", true);
        for (const auto & id : item["evidence_ids"]) {
            if (!id.is_string()) { error = "artifact snapshot assumption evidence id is not text"; return false; }
            assumption.evidence_ids.push_back(id.get<std::string>());
        }
        s.assumptions.push_back(std::move(assumption));
    }
    if (s.id.empty() || s.source_plan_id.empty() || s.plan_version == 0 || s.profile_id.empty() ||
            s.profile_revision.empty() || s.fingerprint.empty()) {
        error = "artifact snapshot identity is incomplete";
        return false;
    }
    if (s.id != "snapshot:" + s.source_plan_id || snapshot_fingerprint(s) != s.fingerprint) {
        error = "artifact snapshot identity or content fingerprint is invalid";
        return false;
    }
    error.clear();
    return true;
}

bool common_agent_artifact_document_from_plan(const common_plan_state & plan,
        const common_agent_artifact_profile & profile, const std::string & snapshot_uri,
        common_agent_artifact_document & out, std::string & error) {
    common_agent_artifact_snapshot snapshot;
    if(!common_agent_artifact_snapshot_from_plan(plan,profile,snapshot,error))return false;
    out={};out.id="report:"+plan.id;out.title=profile.title;out.purpose=profile.purpose;out.profile_id=profile.id;
    out.profile_revision=profile.revision;out.snapshot_uri=snapshot_uri;out.source_plan_id=plan.id;out.source_plan_version=snapshot.plan_version;
    out.coverage_complete=snapshot.coverage_complete;out.unresolved_gaps=snapshot.unresolved_gaps;
    for(const auto & profile_section:profile.sections){common_agent_artifact_section section;section.id=profile_section.id;section.title=profile_section.title;
        for(const auto & observation:plan.observations){if(observation.source!="artifact_section")continue;const auto value=json::parse(observation.summary,nullptr,false);if(!value.is_object()||value.value("id",std::string{})!=profile_section.id||!value.contains("blocks")||!value["blocks"].is_array())continue;
            for(const auto & bj:value["blocks"]){common_agent_artifact_block b;b.id=bj.value("id",std::string{});b.text=bj.value("text",std::string{});b.validation_note=bj.value("validation_note",std::string{});const auto kind=bj.value("kind",std::string("observation"));if(!parse_kind(kind,b.kind)){error="stored artifact block kind is invalid";return false;}for(const auto & x:bj.value("evidence_ids",json::array()))if(x.is_string())b.evidence_ids.push_back(x.get<std::string>());for(const auto & x:bj.value("source_refs",json::array()))if(x.is_string())b.source_refs.push_back(x.get<std::string>());common_agent_artifact_block checked;if(!common_agent_artifact_validate_block(b,snapshot,checked,error))return false;section.blocks.push_back(std::move(checked));}}
        if(section.blocks.empty()){common_agent_artifact_block gap;gap.id=profile_section.id+":gap";gap.kind=common_agent_artifact_block_kind::observation;gap.text="No validated content was produced for this section.";gap.validation=common_agent_artifact_validation::gap;gap.validation_note="Section synthesis is missing from the source plan.";gap.origin="host";section.blocks.push_back(std::move(gap));}
        out.sections.push_back(std::move(section));}
    error.clear();return true;
}

bool common_agent_artifact_document_from_json(const std::string & text, common_agent_artifact_document & d, std::string & error) {
    const auto v=json::parse(text,nullptr,false);
    if (!v.is_object() || v.value("schema_version",0)!=1 || !v.contains("sections") || !v["sections"].is_array()) { error="unsupported or malformed artifact document"; return false; }
    d={}; d.id=v.value("id",std::string{}); d.title=v.value("title",std::string{}); d.purpose=v.value("purpose",std::string{});
    d.profile_id=v.value("profile_id",std::string{}); d.profile_revision=v.value("profile_revision",std::string{});
    d.revision=v.value("revision",uint64_t(1)); d.previous_revision_uri=v.value("previous_revision_uri",std::string{});
    d.snapshot_uri=v.value("snapshot_uri",std::string{}); d.source_plan_id=v.value("source_plan_id",std::string{}); d.source_plan_version=v.value("source_plan_version",uint64_t(0));
    d.coverage_complete=v.value("coverage_complete",false);
    for(const auto & gap:v.value("unresolved_gaps",json::array())){if(!gap.is_string()){error="artifact document gap is not text";return false;}d.unresolved_gaps.push_back(gap.get<std::string>());}
    for (const auto & sj : v["sections"]) {
        if (!sj.is_object() || !sj.contains("blocks") || !sj["blocks"].is_array()) { error="artifact section is malformed"; return false; }
        common_agent_artifact_section s; s.id=sj.value("id",std::string{}); s.title=sj.value("title",std::string{});
        for (const auto & bj : sj["blocks"]) {
            if (!bj.is_object() || !bj.contains("evidence_ids") || !bj["evidence_ids"].is_array() || !bj.contains("source_refs") || !bj["source_refs"].is_array()) { error="artifact block is malformed"; return false; }
            common_agent_artifact_block b; b.id=bj.value("id",std::string{}); b.text=bj.value("text",std::string{});
            b.origin=bj.value("origin",std::string("model")); b.validation_note=bj.value("validation_note",std::string{});
            if (!parse_kind(bj.value("kind",std::string{}),b.kind) || !parse_validation(bj.value("validation",std::string("unvalidated")),b.validation)) { error="artifact block enum is invalid"; return false; }
            for (const auto & x:bj["evidence_ids"]) { if(!x.is_string()){error="artifact evidence id is not text";return false;} b.evidence_ids.push_back(x.get<std::string>()); }
            for (const auto & x:bj["source_refs"]) { if(!x.is_string()){error="artifact source ref is not text";return false;} b.source_refs.push_back(x.get<std::string>()); }
            s.blocks.push_back(std::move(b));
        }
        d.sections.push_back(std::move(s));
    }
    if(d.id.empty()||d.title.empty()){error="artifact document identity is missing";return false;} error.clear();return true;
}

bool common_agent_artifact_render_markdown(const common_agent_artifact_document & d, std::string & out, std::string & error) {
    if(d.id.empty()||d.title.empty()){error="artifact document identity is missing";return false;}
    out="# "+d.title+"\n\n";
    if(!d.purpose.empty()) out+="> "+d.purpose+"\n\n";
    if(!d.coverage_complete){out+="**Coverage incomplete.** The source snapshot does not establish complete coverage.";for(const auto & gap:d.unresolved_gaps)out+="\n- Unresolved: "+gap;out+="\n\n";}
    for(const auto & s:d.sections){out+="## "+s.title+"\n\n";for(const auto & b:s.blocks){
        out+=(b.kind==common_agent_artifact_block_kind::observation?"- ":b.kind==common_agent_artifact_block_kind::conclusion?"**Conclusion.** ":"**Recommendation.** ")+b.text;
        if(!b.evidence_ids.empty()||!b.source_refs.empty()){out+="  \n  Sources: ";bool first=true;for(const auto & x:b.evidence_ids){if(!first)out+=", ";out+=x;first=false;}for(const auto & x:b.source_refs){if(!first)out+=", ";out+=x;first=false;}}
        if(b.validation==common_agent_artifact_validation::gap||b.validation==common_agent_artifact_validation::contradicted) out+="  \n  **Evidence status: "+std::string(validation_name(b.validation))+" — "+b.validation_note+"**";
        out+="\n\n";}}
    error.clear();return true;
}

bool common_agent_artifact_completion_summary(const common_plan_state & plan, std::string & out) {
    // Successful tool observations are written only after host execution and
    // are already scoped to this plan. Do not depend on a step retaining its
    // transient tool-call payload after completion.
    for (auto observation = plan.observations.rbegin(); observation != plan.observations.rend(); ++observation) {
        if (observation->source != "artifact.export" || observation->confidence <= 0.0f) continue;
        auto result = json::parse(observation->summary, nullptr, false);
        if (!result.is_object()) continue;
        if (result.contains("structuredContent") && result["structuredContent"].is_object()) {
            result = result["structuredContent"];
        } else if (result.contains("result") && result["result"].is_object()) {
            result = result["result"];
        }
        const auto report = result.value("report_resource", std::string{});
        const auto markdown = result.value("markdown_resource", std::string{});
        const auto snapshot = result.value("snapshot_resource", std::string{});
        if (report.rfind("agent-resource://", 0) != 0 || markdown.rfind("agent-resource://", 0) != 0 ||
                snapshot.rfind("agent-resource://", 0) != 0 ||
                result.value("profile_id", std::string{}).empty() ||
                !result.contains("coverage_complete") ||
                !result["coverage_complete"].is_boolean()) continue;
        const bool coverage_complete = result["coverage_complete"].get<bool>();
        out = coverage_complete
            ? "Report composed from the frozen source snapshot.\nStructured report: " + report +
                "\nMarkdown: " + markdown
            : "Report composed, but source coverage is incomplete; the available material does not establish a supported conclusion.\nStructured report: " +
                report + "\nMarkdown: " + markdown;
        return true;
    }
    out.clear();
    return false;
}
