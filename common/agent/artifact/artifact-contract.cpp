#include "artifact-contract.h"

#include <nlohmann/json.hpp>
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <map>
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
std::string bounded_utf8(const std::string & value, size_t max_bytes) {
    if (value.size() <= max_bytes) return value;
    size_t end = max_bytes;
    while (end > 0 && end < value.size() &&
            (static_cast<unsigned char>(value[end]) & 0xc0u) == 0x80u) --end;
    return value.substr(0, end);
}
std::string snapshot_fingerprint_v1(const common_agent_artifact_snapshot & snapshot) {
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
const char * research_relation(common_agent_research_evidence_relation v) {
    switch (v) { case common_agent_research_evidence_relation::supports: return "supports";
        case common_agent_research_evidence_relation::contradicts: return "contradicts";
        case common_agent_research_evidence_relation::qualifies: return "qualifies";
        case common_agent_research_evidence_relation::contextualizes: return "contextualizes";
        case common_agent_research_evidence_relation::inconclusive: return "inconclusive"; }
    return "inconclusive";
}
const char * research_origin(common_agent_research_evidence_origin v) {
    switch (v) { case common_agent_research_evidence_origin::direct_source: return "direct_source";
        case common_agent_research_evidence_origin::normalized_tool_result: return "normalized_tool_result";
        case common_agent_research_evidence_origin::model_inference: return "model_inference"; }
    return "normalized_tool_result";
}
json snapshot_material(const common_agent_artifact_snapshot & s) {
    json material={{"schema_version",2},{"id",s.id},{"source_plan_id",s.source_plan_id},
        {"plan_version",s.plan_version},{"profile_id",s.profile_id},{"profile_revision",s.profile_revision},
        {"research_workspace_id",s.research_workspace_id},{"research_checkpoint_id",s.research_checkpoint_id},
        {"research_objective",s.research_objective},{"source_coverage",{
            {"total_gaps",s.research_coverage.total_gaps},{"answered_gaps",s.research_coverage.answered_gaps},
            {"blocked_gaps",s.research_coverage.blocked_gaps},{"unresolved_critical_gaps",s.research_coverage.unresolved_critical_gaps},
            {"objective_coverage",s.research_coverage.objective_coverage},{"evidence_quality",s.research_coverage.evidence_quality},
            {"source_diversity",s.research_coverage.source_diversity}}},
        {"coverage_complete",s.coverage_complete},{"unresolved_gaps",s.unresolved_gaps},
        {"omitted_source_refs",s.omitted_source_refs},{"observations",json::array()},
        {"assumptions",json::array()},{"research_sources",json::array()},{"research_evidence",json::array()},
        {"research_comparisons",json::array()},{"research_gaps",json::array()}};
    for (const auto & o : s.observations) {
        json refs=json::array(); for (const auto & r:o.resource_refs) refs.push_back({{"uri",r.uri},{"name",r.name},
            {"description",r.description},{"mime_type",r.mime_type},{"size_bytes",r.size_bytes},
            {"scope",static_cast<int>(r.scope)},{"purpose",r.metadata.purpose},{"content_summary",r.metadata.content_summary},
            {"usage_hint",r.metadata.usage_hint},{"limitations",r.metadata.limitations},{"keywords",r.metadata.keywords},
            {"entities",r.metadata.entities},{"processing_cache_key",r.metadata.processing_cache_key},
            {"declared_language",r.metadata.declared_language},{"resolved_language",r.metadata.resolved_language},
            {"language_confidence",r.metadata.language_confidence},{"language_source",r.metadata.language_source},
            {"source_provider",r.metadata.source_provider},{"source_operation",r.metadata.source_operation},
            {"source_request_json",r.metadata.source_request_json},{"retrieved_at",r.metadata.retrieved_at},
            {"content_hash",r.metadata.content_hash},{"parent_uri",r.lineage.parent_uri},
            {"chunk_index",r.lineage.chunk_index},{"chunk_count",r.lineage.chunk_count},{"byte_offset",r.lineage.byte_offset},
            {"byte_length",r.lineage.byte_length},{"overlap_bytes",r.lineage.overlap_bytes},{"derivation",r.lineage.derivation}});
        json datasets=json::array(); for (const auto & d:o.dataset_refs) datasets.push_back({{"uri",d.uri},{"name",d.name},
            {"row_count",d.row_count},{"column_count",d.column_count},{"source_resource_uri",d.source_resource_uri},
            {"source_representation",d.source_representation},{"source_provider",d.source_provider},
            {"source_operation",d.source_operation},{"source_request_json",d.source_request_json},
            {"retrieved_at",d.retrieved_at},{"content_hash",d.content_hash}});
        material["observations"].push_back({{"id",o.id},{"source",o.source},{"summary",o.summary},
            {"confidence",o.confidence},{"created_at",o.created_at},{"evidence_ids",o.evidence_ids},
            {"resource_refs",refs},{"dataset_refs",datasets}});
    }
    for (const auto & a:s.assumptions) material["assumptions"].push_back({{"id",a.id},{"statement",a.statement},
        {"valid",a.valid},{"confidence",a.confidence},{"evidence_ids",a.evidence_ids}});
    for (const auto & s0:s.research_sources) material["research_sources"].push_back({{"id",s0.source_id},
        {"title",s0.title},{"origin",s0.origin},{"authority",s0.authority},{"retrieved_at",s0.retrieved_at},
        {"content_hash",s0.content_hash},{"kind",static_cast<int>(s0.kind)},
        {"resource_ref",s0.resource_ref?json{{"uri",s0.resource_ref->uri},{"name",s0.resource_ref->name},
            {"description",s0.resource_ref->description},{"mime_type",s0.resource_ref->mime_type},
            {"size_bytes",s0.resource_ref->size_bytes},{"scope",static_cast<int>(s0.resource_ref->scope)},
            {"content_hash",s0.resource_ref->metadata.content_hash},{"retrieved_at",s0.resource_ref->metadata.retrieved_at},
            {"processing_cache_key",s0.resource_ref->metadata.processing_cache_key},
            {"parent_uri",s0.resource_ref->lineage.parent_uri},{"derivation",s0.resource_ref->lineage.derivation}}:json(nullptr)},
        {"quality_score",s0.quality_score},
        {"primary_source",s0.primary_source},{"memory_id",s0.memory_id},{"availability",static_cast<int>(s0.availability)},
        {"role",s0.role},{"required",s0.required}});
    for (const auto & e:s.research_evidence) material["research_evidence"].push_back({{"id",e.evidence_id},
        {"source_id",e.source_id},{"claim_id",e.claim_id},{"statement",e.statement},{"source_location",e.source_location},
        {"relation",research_relation(e.relation)},{"origin",research_origin(e.origin)},{"relevance",e.relevance},
        {"confidence",e.confidence},{"directly_observed",e.directly_observed},{"model_inferred",e.model_inferred},
        {"supporting_excerpt",e.supporting_excerpt},{"excerpt_host_verified",e.supporting_excerpt_host_verified}});
    for (const auto & c:s.research_comparisons) material["research_comparisons"].push_back({{"id",c.comparison_id},
        {"gap_id",c.gap_id},{"source_ids",c.source_ids},{"evidence_ids",c.evidence_ids},
        {"relation",static_cast<int>(c.relation)},{"summary",c.summary}});
    for (const auto & g:s.research_gaps) material["research_gaps"].push_back({{"id",g.gap_id},{"question",g.question},
        {"reason",g.reason},{"completion_criterion",g.completion_criterion},{"priority",g.priority},
        {"status",static_cast<int>(g.status)},{"evidence_ids",g.evidence_ids},{"child_gap_ids",g.child_gap_ids}});
    return material;
}
std::string snapshot_fingerprint(const common_agent_artifact_snapshot & snapshot) {
    return fingerprint(snapshot_material(snapshot).dump());
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
const char * support_name(common_agent_artifact_support v) {
    switch(v){case common_agent_artifact_support::unverified:return "unverified";case common_agent_artifact_support::linked_support:return "linked_support";case common_agent_artifact_support::contradicted:return "contradicted";case common_agent_artifact_support::mixed:return "mixed";}
    return "unverified";
}
bool parse_validation(const std::string & s, common_agent_artifact_validation & v) {
    if (s == "unvalidated") v = common_agent_artifact_validation::unvalidated;
    else if (s == "references_resolve") v = common_agent_artifact_validation::references_resolve;
    else if (s == "gap") v = common_agent_artifact_validation::gap;
    else if (s == "contradicted") v = common_agent_artifact_validation::contradicted;
    else return false;
    return true;
}
bool parse_support(const std::string & s, common_agent_artifact_support & v) {
    if(s=="unverified")v=common_agent_artifact_support::unverified;else if(s=="linked_support")v=common_agent_artifact_support::linked_support;else if(s=="contradicted")v=common_agent_artifact_support::contradicted;else if(s=="mixed")v=common_agent_artifact_support::mixed;else return false;return true;
}
struct artifact_handles {
    std::map<std::string, std::string> evidence;
    std::map<std::string, std::string> sources;
    std::map<std::string, std::string> evidence_ids_to_handle;
    std::map<std::string, std::string> source_ids_to_handle;
};
artifact_handles make_artifact_handles(const common_agent_artifact_snapshot & snapshot) {
    artifact_handles out;
    size_t eindex = 0, sindex = 0, oindex = 0;
    auto add_evidence = [&](const std::string & id) {
        if (id.empty() || out.evidence_ids_to_handle.count(id)) return;
        const auto handle = "E" + std::to_string(++eindex);
        out.evidence[handle] = id;
        out.evidence_ids_to_handle[id] = handle;
    };
    auto add_source = [&](const std::string & id, const std::string & canonical) {
        if (id.empty()) return;
        auto it = out.source_ids_to_handle.find(id);
        if (it == out.source_ids_to_handle.end()) {
            const auto handle = "S" + std::to_string(++sindex);
            out.source_ids_to_handle[id] = handle;
            out.sources[handle] = canonical.empty() ? id : canonical;
        } else if (!canonical.empty()) {
            out.sources[it->second] = canonical;
        }
    };
    for (const auto & e : snapshot.research_evidence) add_evidence(e.evidence_id);
    for (const auto & o : snapshot.observations) for (const auto & e : o.evidence_ids) add_evidence(e);
    for (const auto & source : snapshot.research_sources) {
        const auto canonical = source.resource_ref ? source.resource_ref->uri : std::string{};
        add_source(source.source_id, canonical);
        if (!canonical.empty()) add_source(canonical, canonical);
    }
    for (const auto & o : snapshot.observations) {
        if (!o.id.empty()) {
            const auto handle = "O" + std::to_string(++oindex);
            out.sources[handle] = o.id;
            out.source_ids_to_handle[o.id] = handle;
        }
        for (const auto & ref : o.resource_refs) if (!ref.uri.empty()) add_source(ref.uri, ref.uri);
        for (const auto & ref : o.dataset_refs) if (!ref.uri.empty()) add_source(ref.uri, ref.uri);
    }
    return out;
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
        std::set<std::string> kinds;
        for (const auto & kind : section.allowed_block_kinds) {
            common_agent_artifact_block_kind parsed;
            if (!parse_kind(kind, parsed) || !kinds.insert(kind).second) {
                error = "artifact section contains an invalid or duplicate allowed block kind";
                return false;
            }
        }
    }
    error.clear(); return true;
}

bool common_agent_artifact_profile_from_json(const std::string & text, common_agent_artifact_profile & out, std::string & error) {
    const auto value=json::parse(text,nullptr,false);
    if(!value.is_object()||!value.contains("sections")||!value["sections"].is_array()){error="artifact profile JSON is malformed";return false;}
    out={};out.id=value.value("id",std::string{});out.revision=value.value("revision",std::string{});out.title=value.value("title",std::string{});out.purpose=value.value("purpose",std::string{});
    for(const auto & item:value["sections"]){if(!item.is_object()){error="artifact profile section is malformed";return false;}common_agent_artifact_section_profile section;section.id=item.value("id",std::string{});section.title=item.value("title",std::string{});section.purpose=item.value("purpose",std::string{});for(const auto & source:item.value("required_observation_sources",json::array())){if(!source.is_string()){error="artifact profile source must be text";return false;}section.required_observation_sources.push_back(source.get<std::string>());}if(item.contains("allowed_block_kinds")){if(!item["allowed_block_kinds"].is_array()){error="artifact allowed_block_kinds must be an array";return false;}section.allowed_block_kinds.clear();for(const auto & kind:item["allowed_block_kinds"]){if(!kind.is_string()){error="artifact block kind must be text";return false;}section.allowed_block_kinds.push_back(kind.get<std::string>());}}out.sections.push_back(std::move(section));}
    return common_agent_artifact_profile_valid(out,error);
}

bool common_agent_artifact_snapshot_from_plan(const common_plan_state & plan,
        const common_agent_artifact_profile & profile, common_agent_artifact_snapshot & out,
        std::string & error, const common_agent_research_workspace * research,
        const std::string & research_checkpoint_id) {
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
    if (research != nullptr) {
        std::string validation_error;
        if (!common_agent_research_workspace_validate(*research, validation_error) ||
                (!research->plan_id.empty() && research->plan_id != plan.id)) {
            error = "research workspace cannot be attached to this report snapshot: " + validation_error;
            return false;
        }
        out.research_workspace_id = research->workspace_id;
        out.research_checkpoint_id = research_checkpoint_id;
        out.research_objective = research->objective.question;
        out.research_coverage = research->coverage;
        out.research_sources = research->sources;
        out.research_evidence = research->evidence;
        out.research_comparisons = research->comparisons;
        out.research_gaps = research->gaps;
    }
    out.assumptions = plan.assumptions;
    bool coverage_complete = research != nullptr
        ? research->coverage.objective_coverage >= 1.0 &&
            research->coverage.answered_gaps >= research->coverage.total_gaps &&
            research->coverage.unresolved_critical_gaps == 0
        : !plan.observations.empty();
    for (const auto & o : plan.observations) {
        if (o.source == "artifact_section" || o.source == "artifact_snapshot") continue;
        out.observations.push_back(o);
        const auto parsed = json::parse(o.summary, nullptr, false);
        if (parsed.is_object() && parsed.contains("coverage_complete") && parsed.value("coverage_complete", false) == false) coverage_complete = false;
        if (parsed.is_object() && parsed.contains("unresolved_claims") && parsed["unresolved_claims"].is_array())
            for (const auto & id : parsed["unresolved_claims"]) if (id.is_string()) out.unresolved_gaps.push_back(id.get<std::string>());
    }
    if (research != nullptr) {
        if (research->coverage.unresolved_critical_gaps > 0) {
            out.unresolved_gaps.push_back("research:unresolved_critical_gaps:" +
                std::to_string(research->coverage.unresolved_critical_gaps));
        }
        for (const auto & gap : research->gaps) {
            if (gap.status == common_agent_research_gap_status::open ||
                    gap.status == common_agent_research_gap_status::investigating ||
                    gap.status == common_agent_research_gap_status::blocked) {
                out.unresolved_gaps.push_back("research_gap:" + gap.gap_id);
            }
        }
        for (const auto & comparison : research->comparisons) if (comparison.relation == common_agent_research_comparison_relation::conflict)
            out.unresolved_gaps.push_back("research_conflict:" + comparison.comparison_id);
        for (const auto & evidence : research->evidence) if (evidence.relation == common_agent_research_evidence_relation::contradicts)
            out.unresolved_gaps.push_back("contradictory_evidence:" + evidence.evidence_id);
        coverage_complete = coverage_complete && research->coverage.unresolved_critical_gaps == 0 &&
            research->coverage.objective_coverage >= 1.0 && research->coverage.answered_gaps >= research->coverage.total_gaps;
        if (research_checkpoint_id.empty()) {
            out.unresolved_gaps.push_back("research_checkpoint_missing:" + research->workspace_id);
            coverage_complete = false;
        }
        for (const auto & evidence : research->evidence) {
            if (std::none_of(research->sources.begin(), research->sources.end(), [&](const auto & source) {
                    return source.source_id == evidence.source_id;
                })) {
                out.unresolved_gaps.push_back("evidence_source_missing:" + evidence.evidence_id);
                coverage_complete = false;
            }
        }
    }
    for (const auto & section : profile.sections) {
        for (const auto & required_source : section.required_observation_sources) {
            const bool present = std::any_of(out.observations.begin(), out.observations.end(), [&](const auto & observation) {
                return observation.source == required_source;
            }) || (research != nullptr &&
                ((required_source == "research.source" && !research->sources.empty()) ||
                 (required_source == "research.evidence" && !research->evidence.empty())));
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
    for (const auto & item : snapshot.research_evidence) {
        evidence.insert(item.evidence_id);
        if (!item.claim_id.empty()) evidence.insert(item.claim_id);
    }
    for (const auto & item : snapshot.research_sources) {
        sources.insert(item.source_id);
        if (item.resource_ref && !item.resource_ref->uri.empty()) sources.insert(item.resource_ref->uri);
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
    bool has_support=false,has_contradiction=false;
    for(const auto & id:block.evidence_ids)for(const auto & evidence:snapshot.research_evidence)if(evidence.evidence_id==id){
        has_support=has_support||evidence.relation==common_agent_research_evidence_relation::supports;
        has_contradiction=has_contradiction||evidence.relation==common_agent_research_evidence_relation::contradicts;
    }
    out.support=has_support&&has_contradiction?common_agent_artifact_support::mixed:
        has_contradiction?common_agent_artifact_support::contradicted:
        has_support?common_agent_artifact_support::linked_support:common_agent_artifact_support::unverified;
    if (!refs_valid) { error.clear(); }
    else error.clear();
    return true;
}

bool common_agent_artifact_parse_section_dsl(const std::string & text,
        const common_agent_artifact_section_profile & profile,
        const common_agent_artifact_snapshot & snapshot, common_agent_artifact_section & out,
        std::string & error, const std::vector<std::string> & included_refs,
        bool restrict_to_included_refs) {
    out = {}; out.id = profile.id; out.title = profile.title;
    common_agent_artifact_block block; bool active = false; size_t count = 0;
    auto flush = [&]() -> bool {
        if (!active) return true;
        common_agent_artifact_block checked;
        if (!common_agent_artifact_validate_block(block, snapshot, checked, error)) return false;
        out.blocks.push_back(std::move(checked)); block = {}; active = false; ++count; return true;
    };
    auto handles = make_artifact_handles(snapshot);
    if(restrict_to_included_refs){
        const std::set<std::string> allowed(included_refs.begin(),included_refs.end());
        auto restrict_map=[&](auto & aliases){for(auto it=aliases.begin();it!=aliases.end();)if(!allowed.count(it->second))it=aliases.erase(it);else ++it;};
        restrict_map(handles.evidence);restrict_map(handles.sources);
    }
    auto resolve_handle=[](std::string & ref,const std::map<std::string,std::string> & handles){auto it=handles.find(ref);if(it!=handles.end())ref=it->second;};
    size_t begin = 0;
    while (begin <= text.size()) {
        const auto end = text.find('\n', begin);
        auto line = text.substr(begin, end == std::string::npos ? std::string::npos : end - begin);
        if (!line.empty() && line.back() == '\r') line.pop_back();
        const auto colon = line.find(':');
        if (colon != std::string::npos) {
            auto key = line.substr(0, colon), value = line.substr(colon + 1);
            while (!value.empty() && value.front() == ' ') value.erase(value.begin());
            if (key == "status") {
                if(value=="no_findings"){if(active||!out.blocks.empty()){error="no_findings cannot be combined with blocks";return false;}out.no_findings=true;}
                else if(value!="findings"){error="artifact status must be findings or no_findings";return false;}
            } else if (key == "block") {
                if(out.no_findings){error="no_findings cannot be combined with blocks";return false;}
                if (!flush()) return false;
                active = true; block.id = profile.id + ":" + std::to_string(count + 1);
                if (!parse_kind(value, block.kind)) { error = "artifact block kind must be observation, conclusion or recommendation"; return false; }
                if(std::find(profile.allowed_block_kinds.begin(),profile.allowed_block_kinds.end(),value)==profile.allowed_block_kinds.end()){error="artifact block kind is not allowed by section profile";return false;}
            } else if (key == "text") {
                if (!active) { error="artifact section field appears before block";return false; }
                if(!value.empty()&&value.front()=='"'){auto parsed=json::parse(value,nullptr,false);if(!parsed.is_string()){error="artifact text field has invalid quoted-string syntax";return false;}value=parsed.get<std::string>();}
                if(!block.text.empty())block.text+='\n';
                block.text += value;
            } else if (key == "evidence" || key == "source") {
                if (!active) { error="artifact section field appears before block";return false; }
                size_t item_begin=0; while(item_begin<=value.size()){const auto item_end=value.find(',',item_begin);auto ref=value.substr(item_begin,item_end==std::string::npos?std::string::npos:item_end-item_begin);while(!ref.empty()&&ref.front()==' ')ref.erase(ref.begin());while(!ref.empty()&&ref.back()==' ')ref.pop_back();if(!ref.empty()){resolve_handle(ref,key=="evidence"?handles.evidence:handles.sources);(key=="evidence"?block.evidence_ids:block.source_refs).push_back(std::move(ref));}if(item_end==std::string::npos)break;item_begin=item_end+1;}
            } else if (!line.empty()) { error="unknown artifact section field: "+key;return false; }
        } else if (!line.empty()) {if(!active||block.text.empty()){error="artifact section continuation requires a text field";return false;}block.text+='\n';block.text+=line;}
        if (end == std::string::npos) break;
        begin = end + 1;
    }
    if (!flush()) return false;
    if ((!out.no_findings&&out.blocks.empty()) || out.blocks.size() > 32) { error="artifact section must contain blocks or explicit no_findings";return false; }
    error.clear(); return true;
}

bool common_agent_artifact_parse_section_json(const std::string & text,
        const common_agent_artifact_section_profile & profile,
        const common_agent_artifact_snapshot & snapshot, common_agent_artifact_section & out,
        std::string & error, const std::vector<std::string> & included_refs,
        bool restrict_to_included_refs) {
    const auto value=json::parse(text,nullptr,false);
    if(!value.is_object()||!value.contains("status")||!value["status"].is_string()){error="artifact section JSON requires status";return false;}
    out={};out.id=profile.id;out.title=profile.title;
    const auto status=value["status"].get<std::string>();
    if(status=="no_findings"){if(value.contains("blocks")&&(!value["blocks"].is_array()||!value["blocks"].empty())){error="no_findings cannot include blocks";return false;}out.no_findings=true;error.clear();return true;}
    if(status!="findings"||!value.contains("blocks")||!value["blocks"].is_array()||value["blocks"].empty()||value["blocks"].size()>32){error="findings require 1 to 32 blocks";return false;}
    auto handles=make_artifact_handles(snapshot);
    if(restrict_to_included_refs){
        const std::set<std::string> allowed(included_refs.begin(),included_refs.end());
        auto restrict_map=[&](auto & aliases){for(auto it=aliases.begin();it!=aliases.end();)if(!allowed.count(it->second))it=aliases.erase(it);else ++it;};
        restrict_map(handles.evidence);restrict_map(handles.sources);
    }
    auto resolve=[](std::string & ref,const std::map<std::string,std::string> & aliases){auto it=aliases.find(ref);if(it!=aliases.end())ref=it->second;};
    for(const auto & item:value["blocks"]){if(!item.is_object()||!item.contains("kind")||!item["kind"].is_string()||!item.contains("text")||!item["text"].is_string()){error="artifact block requires kind and text";return false;}
        common_agent_artifact_block block;block.id=profile.id+":"+std::to_string(out.blocks.size()+1);const auto kind=item["kind"].get<std::string>();
        if(!parse_kind(kind,block.kind)||std::find(profile.allowed_block_kinds.begin(),profile.allowed_block_kinds.end(),kind)==profile.allowed_block_kinds.end()){error="artifact block kind is invalid for section";return false;}
        block.text=item["text"].get<std::string>();for(const auto & ref_json:item.value("evidence",json::array())){if(!ref_json.is_string()){error="evidence handle must be text";return false;}auto ref=ref_json.get<std::string>();resolve(ref,handles.evidence);block.evidence_ids.push_back(std::move(ref));}
        for(const auto & ref_json:item.value("source",json::array())){if(!ref_json.is_string()){error="source handle must be text";return false;}auto ref=ref_json.get<std::string>();resolve(ref,handles.sources);block.source_refs.push_back(std::move(ref));}
        common_agent_artifact_block checked;if(!common_agent_artifact_validate_block(block,snapshot,checked,error))return false;out.blocks.push_back(std::move(checked));}
    error.clear();return true;
}

bool common_agent_artifact_document_to_json(const common_agent_artifact_document & d, std::string & out, std::string & error) {
    json value = {{"schema_version",1},{"id",d.id},{"title",d.title},{"purpose",d.purpose},
        {"profile_id",d.profile_id},{"profile_revision",d.profile_revision},{"revision",d.revision},
        {"previous_revision_uri",d.previous_revision_uri},{"snapshot_uri",d.snapshot_uri},
        {"snapshot_fingerprint",d.snapshot_fingerprint},
        {"source_plan_id",d.source_plan_id},{"source_plan_version",d.source_plan_version},
        {"coverage_complete",d.coverage_complete},{"unresolved_gaps",d.unresolved_gaps},{"sections",json::array()}};
    if (d.id.empty() || d.title.empty() || d.sections.size() > 32) { error="artifact document has invalid identity or section count"; return false; }
    for (const auto & s : d.sections) {
        json section={{"id",s.id},{"title",s.title},{"no_findings",s.no_findings},{"blocks",json::array()}};
        for (const auto & b : s.blocks) section["blocks"].push_back({{"id",b.id},{"kind",block_kind(b.kind)},
            {"text",b.text},{"evidence_ids",b.evidence_ids},{"source_refs",b.source_refs},
            {"validation",validation_name(b.validation)},{"validation_note",b.validation_note},{"support",support_name(b.support)},{"origin",b.origin}});
        value["sections"].push_back(std::move(section));
    }
    out=value.dump(); error.clear(); return true;
}

bool common_agent_artifact_snapshot_to_json(const common_agent_artifact_snapshot & s, std::string & out, std::string & error) {
    auto value=snapshot_material(s);
    if (s.fingerprint.empty() || s.fingerprint != snapshot_fingerprint(s)) { error="artifact snapshot fingerprint is missing or stale"; return false; }
    value["fingerprint"]=s.fingerprint;
    out=value.dump();error.clear();return true;
}

bool common_agent_artifact_render_section_context(const common_agent_artifact_snapshot & snapshot,
        const std::vector<std::string> & required_sources, size_t char_budget,
        std::string & context_json, std::vector<std::string> & included_refs,
        std::vector<std::string> & omitted_refs, std::string & error) {
    included_refs.clear();omitted_refs.clear();
    // Model context is a bounded view, not a second serialization of the
    // frozen snapshot. Keep stable identities and an explicit gap summary;
    // the complete gaps and provenance remain in the host-owned snapshot.
    json gap_summary=json::array();
    for (size_t i=0;i<std::min<size_t>(snapshot.unresolved_gaps.size(),2);++i)
        gap_summary.push_back(bounded_utf8(snapshot.unresolved_gaps[i],48));
    // Identity, full lineage and the complete checkpoint remain in the
    // immutable host snapshot. The model only needs task intent, coverage and
    // compact source/evidence handles; repeating long host IDs here crowds
    // out the actual evidence under small per-step budgets.
    json root={{"objective",bounded_utf8(snapshot.research_objective,512)},
        {"coverage",snapshot.coverage_complete?"complete":"incomplete"},
        {"records",json::array()}};
    if(!snapshot.unresolved_gaps.empty())root["gap_count"]=snapshot.unresolved_gaps.size();
    if(!gap_summary.empty())root["gaps"]=std::move(gap_summary);
    if(snapshot.unresolved_gaps.size()>2)root["more_gaps"]=true;
    const auto handles = make_artifact_handles(snapshot);
    struct record { std::string ref; json value; bool required=false; std::string related_source_ref; };
    std::vector<record> records;
    for(size_t i=0;i<snapshot.research_sources.size();++i){const auto & s=snapshot.research_sources[i];const auto alias=handles.source_ids_to_handle.find(s.source_id);json v={{"source_handle",alias==handles.source_ids_to_handle.end()?"S"+std::to_string(i+1):alias->second},
        {"title",bounded_utf8(s.title,40)}};
        records.push_back({s.source_id,std::move(v),s.required,{}});}
    for(size_t i=0;i<snapshot.research_evidence.size();++i){const auto & e=snapshot.research_evidence[i];const auto alias=handles.evidence_ids_to_handle.find(e.evidence_id);
        const auto source_alias=handles.source_ids_to_handle.find(e.source_id);
        const auto source_record=std::find_if(snapshot.research_sources.begin(),snapshot.research_sources.end(),
            [&](const auto & source){return source.source_id==e.source_id;});
        const auto & grounded_text=e.supporting_excerpt_host_verified&&!e.supporting_excerpt.empty()
            ? e.supporting_excerpt : e.statement;
        json v={{"evidence_handle",alias==handles.evidence_ids_to_handle.end()?"E"+std::to_string(i+1):alias->second},
        {"source_handle",source_alias==handles.source_ids_to_handle.end()?std::string{}:source_alias->second},
        {"text",bounded_utf8(grounded_text,128)},{"relation",research_relation(e.relation)},
        {"verified",e.supporting_excerpt_host_verified}};
        // A resource URI is already bound to its source handle. Repeating it
        // in every evidence row consumes scarce prompt space without adding
        // a location. Keep distinct page/paragraph/range locations.
        const bool location_is_source_uri=source_record!=snapshot.research_sources.end()&&
            source_record->resource_ref&&e.source_location==source_record->resource_ref->uri;
        if(!e.source_location.empty()&&!location_is_source_uri)
            v["location"]=bounded_utf8(e.source_location,32);
        if(e.model_inferred)v["model_inferred"]=true;
        records.push_back({e.evidence_id,std::move(v),false,e.source_id});}
    for(size_t i=0;i<snapshot.research_comparisons.size();++i){const auto & c=snapshot.research_comparisons[i];
        // Agreeing comparisons are already represented by their linked source
        // evidence. Only conflict/qualification comparisons add unique context.
        if(c.relation==common_agent_research_comparison_relation::agree)continue;
        records.push_back({c.comparison_id,
            json{{"kind","comparison"},{"id",c.comparison_id},{"relation",static_cast<int>(c.relation)},{"summary",bounded_utf8(c.summary,96)}},false,{}});}
    for(const auto & g:snapshot.research_gaps){
        if(g.status==common_agent_research_gap_status::sufficiently_answered||
                g.status==common_agent_research_gap_status::abandoned)continue;
        records.push_back({g.gap_id,json{{"kind","gap"},{"id",g.gap_id},{"question",bounded_utf8(g.question,96)},
            {"reason",bounded_utf8(g.reason,48)},{"status",static_cast<int>(g.status)}},false,{}});}
    size_t observation_index=0;
    for(const auto & o:snapshot.observations){if(o.source=="artifact_snapshot"||o.source=="artifact_section")continue;
        // Plan observations are included only when the profile/transition
        // explicitly asks this section to use that source. Research sources
        // and evidence have their own typed projection above.
        if(required_sources.empty()||
                std::find(required_sources.begin(),required_sources.end(),o.source)==required_sources.end())continue;
        json refs=json::array();for(const auto & ref:o.resource_refs)refs.push_back(ref.uri);
        json datasets=json::array();for(const auto & d:o.dataset_refs){const auto alias=handles.source_ids_to_handle.find(d.uri);datasets.push_back({{"handle",alias==handles.source_ids_to_handle.end()?std::string{}:alias->second},{"uri",d.uri},{"name",d.name},{"rows",d.row_count},{"columns",d.column_count},{"source_uri",d.source_resource_uri},{"content_hash",d.content_hash}});}
        json evhandles=json::array();std::set<std::string> seen_evidence_handles;
        for(const auto & id:o.evidence_ids){const auto it=handles.evidence_ids_to_handle.find(id);if(it!=handles.evidence_ids_to_handle.end()&&seen_evidence_handles.insert(it->second).second)evhandles.push_back(it->second);}
        json source_handles_json=json::array();std::set<std::string> seen_source_handles;
        if(!o.id.empty()){const auto it=handles.source_ids_to_handle.find(o.id);if(it!=handles.source_ids_to_handle.end()&&seen_source_handles.insert(it->second).second)source_handles_json.push_back(it->second);}
        for(const auto & ref:o.resource_refs){const auto it=handles.source_ids_to_handle.find(ref.uri);if(it!=handles.source_ids_to_handle.end()&&seen_source_handles.insert(it->second).second)source_handles_json.push_back(it->second);}
        for(const auto & ref:o.dataset_refs){const auto it=handles.source_ids_to_handle.find(ref.uri);if(it!=handles.source_ids_to_handle.end()&&seen_source_handles.insert(it->second).second)source_handles_json.push_back(it->second);}
        records.push_back({o.id,json{{"kind","observation"},{"handle","O"+std::to_string(++observation_index)},{"id",o.id},{"source",o.source},
            {"summary",o.summary},{"confidence",o.confidence},{"evidence_handles",evhandles},{"source_handles",source_handles_json},{"resource_uris",refs},{"datasets",datasets},{"created_at",o.created_at}},false,{}});}
    // Keep evidence and its source together. A sequence of source-only rows
    // can otherwise consume the budget before the corresponding evidence is
    // visible, yielding handles that are technically present but useless.
    std::vector<record> ordered_records;
    std::set<std::string> ordered_refs;
    auto append_record=[&](const record & candidate){
        if(ordered_refs.insert(candidate.ref).second)ordered_records.push_back(candidate);
    };
    for(const auto & item:records)if(item.required&&item.value.contains("source_handle"))append_record(item);
    for(const auto & evidence:records)if(evidence.value.contains("evidence_handle")){
        const auto source=std::find_if(records.begin(),records.end(),[&](const record & candidate){
            return candidate.ref==evidence.related_source_ref&&candidate.value.contains("source_handle")&&!candidate.value.contains("evidence_handle");
        });
        if(source!=records.end())append_record(*source);
        append_record(evidence);
    }
    for(const auto & item:records)append_record(item);
    records=std::move(ordered_records);
    auto escaped_dump=[&](const json & j){auto s=j.dump();std::string safe;safe.reserve(s.size());for(char c:s){if(c=='<')safe+="\\u003c";else if(c=='>')safe+="\\u003e";else if(c=='&')safe+="\\u0026";else safe+=c;}return safe;};
    for(const auto & item:records){
        if(!item.related_source_ref.empty()&&std::find(included_refs.begin(),included_refs.end(),item.related_source_ref)==included_refs.end()){
            omitted_refs.push_back(item.ref);continue;
        }
        root["records"].push_back(item.value);auto rendered=escaped_dump(root);
        if(rendered.size()>char_budget){root["records"].erase(root["records"].end()-1);omitted_refs.push_back(item.ref);}
        else {
            included_refs.push_back(item.ref);
            const auto & v=item.value;
            for(const auto key:{"id","source_id","claim_id","resource_uri","uri"})if(v.contains(key)&&v[key].is_string())included_refs.push_back(v[key].get<std::string>());
            for(const auto key:{"resource_uris"})if(v.contains(key)&&v[key].is_array())for(const auto & ref:v[key])if(ref.is_string())included_refs.push_back(ref.get<std::string>());
            for(const auto key:{"evidence_handles","source_handles"})if(v.contains(key)&&v[key].is_array())for(const auto & handle:v[key])if(handle.is_string()){
                const auto & aliases=std::string(key)=="evidence_handles"?handles.evidence:handles.sources;
                const auto alias=aliases.find(handle.get<std::string>());if(alias!=aliases.end())included_refs.push_back(alias->second);
            }
            for(const auto & handle_field:{std::string("source_handle"),std::string("evidence_handle")})if(v.contains(handle_field)&&v[handle_field].is_string()){
                const auto & aliases=handle_field=="source_handle"?handles.sources:handles.evidence;
                const auto alias=aliases.find(v[handle_field].get<std::string>());if(alias!=aliases.end())included_refs.push_back(alias->second);
            }
        }}
    context_json=escaped_dump(root);
    if(context_json.size()>char_budget){error="artifact section context budget cannot fit required snapshot metadata (required="+
        std::to_string(context_json.size())+", budget="+std::to_string(char_budget)+")";return false;}
    error.clear();return true;
}

std::string common_agent_artifact_section_dsl_grammar(
        const common_agent_artifact_section_profile & profile, const std::string & rendered_context_json) {
    const auto context=json::parse(rendered_context_json,nullptr,false);
    if(!context.is_object()||!context.contains("records")||!context["records"].is_array())return {};
    std::set<std::string> evidence_handles,source_handles,kinds;
    auto add_handle=[](std::set<std::string> & handles,const json & value){
        if(value.is_string()){
            const auto handle=value.get<std::string>();
            if(handle.size()>1&&std::isalpha(static_cast<unsigned char>(handle.front()))&&
                    std::all_of(handle.begin()+1,handle.end(),[](unsigned char c){return std::isdigit(c);}))handles.insert(handle);
        }
    };
    for(const auto & record:context["records"]){
        if(!record.is_object())continue;
        for(const auto key:{"evidence_handle","evidence_handles"})if(record.contains(key)){
            if(record[key].is_array())for(const auto & value:record[key])add_handle(evidence_handles,value);
            else add_handle(evidence_handles,record[key]);
        }
        for(const auto key:{"source_handle","source_handles","handle"})if(record.contains(key)){
            if(record[key].is_array())for(const auto & value:record[key])add_handle(source_handles,value);
            else add_handle(source_handles,record[key]);
        }
    }
    for(const auto & kind:profile.allowed_block_kinds)if(kind=="observation"||kind=="conclusion"||kind=="recommendation")kinds.insert(kind);
    auto alternatives=[](const std::set<std::string> & values){
        std::string result;for(const auto & value:values){if(!result.empty())result+=" | ";result+="\""+value+"\"";}return result;
    };
    // Supporting evidence can be relevant to one report section and irrelevant
    // to another (for example, a date finding is not itself a limitation).
    // Keep the explicit empty result available; section purpose and host-side
    // coverage validation decide whether it is appropriate.
    std::string grammar="root ::= \"status: no_findings\"";
    if(!kinds.empty()&&(!evidence_handles.empty()||!source_handles.empty())){
        grammar+=" | \"status: findings\\n\" block";
        grammar+="\nblock ::= entry | entry \"\\n\" entry\nentry ::= \"block: \" kind \"\\ntext: \" text";
        if(!evidence_handles.empty())grammar+=" \"\\nevidence: \" evidence";
        if(!source_handles.empty())grammar+=" \"\\nsource: \" source";
        // A single terminal punctuation token closes the field. Disallowing
        // earlier sentence terminators lets constrained decoding stop at the
        // first complete sentence instead of filling the entire text budget.
        grammar+="\nkind ::= "+alternatives(kinds)+"\ntext ::= [^.!?\\n]{1,159}[.!?]";
        if(!evidence_handles.empty())grammar+="\nevidence ::= "+alternatives(evidence_handles);
        if(!source_handles.empty())grammar+="\nsource ::= "+alternatives(source_handles);
    }
    return grammar;
}

bool common_agent_artifact_snapshot_from_json(const std::string & text,
        common_agent_artifact_snapshot & s, std::string & error) {
    const auto value = json::parse(text, nullptr, false);
    const int schema_version=value.is_object()?value.value("schema_version",0):0;
    if (!value.is_object() || (schema_version != 1 && schema_version != 2) ||
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
    s.research_workspace_id=value.value("research_workspace_id",std::string{});
    s.research_checkpoint_id=value.value("research_checkpoint_id",std::string{});
    s.research_objective=value.value("research_objective",std::string{});
    if (value.contains("source_coverage") && value["source_coverage"].is_object()) {
        const auto & c=value["source_coverage"];
        s.research_coverage.total_gaps=c.value("total_gaps",0);s.research_coverage.answered_gaps=c.value("answered_gaps",0);
        s.research_coverage.blocked_gaps=c.value("blocked_gaps",0);s.research_coverage.unresolved_critical_gaps=c.value("unresolved_critical_gaps",0);
        s.research_coverage.objective_coverage=c.value("objective_coverage",0.0);s.research_coverage.evidence_quality=c.value("evidence_quality",0.0);
        s.research_coverage.source_diversity=c.value("source_diversity",0.0);
    }
    s.fingerprint = value.value("fingerprint", std::string{});
    s.coverage_complete = value.value("coverage_complete", false);
    for (const auto & gap : value.value("unresolved_gaps", json::array())) {
        if (!gap.is_string()) { error = "artifact snapshot gap is not text"; return false; }
        s.unresolved_gaps.push_back(gap.get<std::string>());
    }
    for (const auto & ref:value.value("omitted_source_refs",json::array())) { if(!ref.is_string()){error="artifact omitted reference is not text";return false;} s.omitted_source_refs.push_back(ref.get<std::string>()); }
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
        observation.confidence=item.value("confidence",0.5f);
        observation.created_at=item.value("created_at",int64_t(0));
        for (const auto & id : item["evidence_ids"]) {
            if (!id.is_string()) { error = "artifact snapshot evidence id is not text"; return false; }
            observation.evidence_ids.push_back(id.get<std::string>());
        }
        for (const auto & ref : item["resource_refs"]) {
            if (!ref.is_object()) { error = "artifact snapshot resource reference is malformed"; return false; }
            common_runtime_resource_ref resource;
            resource.uri = ref.value("uri", std::string{});
            resource.name = ref.value("name", std::string{});
            resource.description=ref.value("description",std::string{});
            resource.mime_type = ref.value("mime_type", std::string{});
            resource.size_bytes=ref.value("size_bytes",size_t(0));
            resource.scope=static_cast<common_runtime_resource_scope>(ref.value("scope",0));
            resource.metadata.purpose=ref.value("purpose",std::string{});resource.metadata.content_summary=ref.value("content_summary",std::string{});
            resource.metadata.usage_hint=ref.value("usage_hint",std::string{});resource.metadata.limitations=ref.value("limitations",std::string{});
            resource.metadata.keywords=ref.value("keywords",std::vector<std::string>{});resource.metadata.entities=ref.value("entities",std::vector<std::string>{});
            resource.metadata.processing_cache_key=ref.value("processing_cache_key",std::string{});resource.metadata.declared_language=ref.value("declared_language",std::string{});
            resource.metadata.resolved_language=ref.value("resolved_language",std::string{});resource.metadata.language_confidence=ref.value("language_confidence",0.0);
            resource.metadata.language_source=ref.value("language_source",std::string{});
            resource.metadata.source_provider=ref.value("source_provider",std::string{});
            resource.metadata.source_operation=ref.value("source_operation",std::string{});
            resource.metadata.source_request_json=ref.value("source_request_json",std::string{});
            resource.metadata.retrieved_at=ref.value("retrieved_at",int64_t(0));
            resource.metadata.content_hash=ref.value("content_hash",std::string{});
            resource.lineage.parent_uri=ref.value("parent_uri",std::string{});
            resource.lineage.chunk_index=ref.value("chunk_index",size_t(0));resource.lineage.chunk_count=ref.value("chunk_count",size_t(0));
            resource.lineage.derivation=ref.value("derivation",std::string{});
            resource.lineage.byte_offset=ref.value("byte_offset",size_t(0));
            resource.lineage.byte_length=ref.value("byte_length",size_t(0));
            resource.lineage.overlap_bytes=ref.value("overlap_bytes",size_t(0));
            observation.resource_refs.push_back(std::move(resource));
        }
        for(const auto & d:item.value("dataset_refs",json::array())) if(d.is_object()) observation.dataset_refs.push_back({
            d.value("uri",std::string{}),d.value("name",std::string{}),d.value("row_count",size_t(0)),
            d.value("column_count",size_t(0)),d.value("source_resource_uri",std::string{}),
            d.value("source_representation",std::string{}),d.value("source_provider",std::string{}),
            d.value("source_operation",std::string{}),d.value("source_request_json",std::string{}),
            d.value("retrieved_at",int64_t(0)),d.value("content_hash",std::string{})});
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
    for(const auto & item:value.value("research_sources",json::array())) if(item.is_object()) {
        common_agent_research_source source;source.source_id=item.value("id",std::string{});source.title=item.value("title",std::string{});
        source.origin=item.value("origin",std::string{});source.authority=item.value("authority",std::string{});source.retrieved_at=item.value("retrieved_at",std::string{});
        source.content_hash=item.value("content_hash",std::string{});source.kind=static_cast<common_agent_research_source_kind>(item.value("kind",1));
        source.quality_score=item.value("quality_score",0.0);source.primary_source=item.value("primary_source",false);source.memory_id=item.value("memory_id",std::string{});
        source.availability=static_cast<common_agent_research_source_availability>(item.value("availability",1));source.role=item.value("role",std::string("reference"));source.required=item.value("required",false);
        if(item.contains("resource_ref")&&item["resource_ref"].is_object()){const auto & r=item["resource_ref"];common_runtime_resource_ref ref;ref.uri=r.value("uri",std::string{});ref.name=r.value("name",std::string{});ref.description=r.value("description",std::string{});ref.mime_type=r.value("mime_type",std::string{});ref.size_bytes=r.value("size_bytes",size_t(0));ref.scope=static_cast<common_runtime_resource_scope>(r.value("scope",0));ref.metadata.content_hash=r.value("content_hash",std::string{});ref.metadata.retrieved_at=r.value("retrieved_at",int64_t(0));ref.metadata.processing_cache_key=r.value("processing_cache_key",std::string{});ref.lineage.parent_uri=r.value("parent_uri",std::string{});ref.lineage.derivation=r.value("derivation",std::string{});if(!ref.uri.empty())source.resource_ref=std::move(ref);}
        else {const auto uri=item.value("resource_uri",std::string{});if(!uri.empty()){common_runtime_resource_ref ref;ref.uri=uri;source.resource_ref=ref;}}
        s.research_sources.push_back(std::move(source));
    }
    for(const auto & item:value.value("research_evidence",json::array())) if(item.is_object()) {
        common_agent_research_evidence e;e.evidence_id=item.value("id",std::string{});e.source_id=item.value("source_id",std::string{});e.claim_id=item.value("claim_id",std::string{});
        e.statement=item.value("statement",std::string{});e.source_location=item.value("source_location",std::string{});const auto relation=item.value("relation",std::string("inconclusive"));
        e.relation=relation=="supports"?common_agent_research_evidence_relation::supports:relation=="contradicts"?common_agent_research_evidence_relation::contradicts:relation=="qualifies"?common_agent_research_evidence_relation::qualifies:relation=="contextualizes"?common_agent_research_evidence_relation::contextualizes:common_agent_research_evidence_relation::inconclusive;
        const auto origin=item.value("origin",std::string("normalized_tool_result"));e.origin=origin=="direct_source"?common_agent_research_evidence_origin::direct_source:origin=="model_inference"?common_agent_research_evidence_origin::model_inference:common_agent_research_evidence_origin::normalized_tool_result;
        e.relevance=item.value("relevance",0.0);e.confidence=item.value("confidence",0.0);e.directly_observed=item.value("directly_observed",false);e.model_inferred=item.value("model_inferred",false);
        e.supporting_excerpt=item.value("supporting_excerpt",std::string{});e.supporting_excerpt_host_verified=item.value("excerpt_host_verified",item.value("supporting_excerpt_host_verified",false));s.research_evidence.push_back(std::move(e));
    }
    for(const auto & item:value.value("research_comparisons",json::array())) if(item.is_object()) {common_agent_research_source_comparison c;c.comparison_id=item.value("id",std::string{});c.gap_id=item.value("gap_id",std::string{});c.source_ids=item.value("source_ids",std::vector<std::string>{});c.evidence_ids=item.value("evidence_ids",std::vector<std::string>{});c.relation=static_cast<common_agent_research_comparison_relation>(item.value("relation",0));c.summary=item.value("summary",std::string{});s.research_comparisons.push_back(std::move(c));}
    for(const auto & item:value.value("research_gaps",json::array())) if(item.is_object()) {common_agent_research_gap g;g.gap_id=item.value("id",std::string{});g.question=item.value("question",std::string{});g.reason=item.value("reason",std::string{});g.completion_criterion=item.value("completion_criterion",std::string{});g.priority=item.value("priority",0);g.status=static_cast<common_agent_research_gap_status>(item.value("status",0));g.evidence_ids=item.value("evidence_ids",std::vector<std::string>{});g.child_gap_ids=item.value("child_gap_ids",std::vector<std::string>{});s.research_gaps.push_back(std::move(g));}
    if (s.id.empty() || s.source_plan_id.empty() || s.plan_version == 0 || s.profile_id.empty() ||
            s.profile_revision.empty() || s.fingerprint.empty()) {
        error = "artifact snapshot identity is incomplete";
        return false;
    }
    const auto expected_fingerprint=schema_version==1?snapshot_fingerprint_v1(s):snapshot_fingerprint(s);
    if (s.id != "snapshot:" + s.source_plan_id || expected_fingerprint != s.fingerprint) {
        error = "artifact snapshot identity or content fingerprint is invalid";
        return false;
    }
    if (schema_version == 1) s.fingerprint = snapshot_fingerprint(s);
    error.clear();
    return true;
}

bool common_agent_artifact_document_from_plan(const common_plan_state & plan,
        const common_agent_artifact_profile & profile, const std::string & snapshot_uri,
        common_agent_artifact_document & out, std::string & error, const std::string & requested_report_id) {
    common_agent_artifact_snapshot snapshot;
    if(!common_agent_artifact_snapshot_from_plan(plan,profile,snapshot,error))return false;
    out={};out.id=requested_report_id.empty()?"report:"+plan.id:requested_report_id;out.title=profile.title;out.purpose=profile.purpose;out.profile_id=profile.id;
    out.profile_revision=profile.revision;out.snapshot_uri=snapshot_uri;out.snapshot_fingerprint=snapshot.fingerprint;out.source_plan_id=plan.id;out.source_plan_version=snapshot.plan_version;
    uint64_t max_revision=0;std::string latest_report_uri;bool same_snapshot_found=false;
    for(const auto & observation:plan.observations){if(observation.source!="artifact.export")continue;const auto prior=json::parse(observation.summary,nullptr,false);if(!prior.is_object()||prior.value("report_id",std::string{})!=out.id)continue;
        const auto revision=prior.value("revision",uint64_t(0));if(revision<1)continue;
        if(revision>max_revision){max_revision=revision;latest_report_uri=prior.value("report_resource",std::string{});}
        if(prior.value("snapshot_fingerprint",std::string{})==snapshot.fingerprint){same_snapshot_found=true;out.revision=revision;out.previous_revision_uri=prior.value("previous_revision_uri",std::string{});}
    }
    if(!same_snapshot_found&&max_revision>0){out.revision=max_revision+1;out.previous_revision_uri=latest_report_uri;}
    out.coverage_complete=snapshot.coverage_complete;out.unresolved_gaps=snapshot.unresolved_gaps;
    bool report_coverage=snapshot.coverage_complete;
    for(const auto & profile_section:profile.sections){common_agent_artifact_section section;section.id=profile_section.id;section.title=profile_section.title;bool found=false;
        for(const auto & observation:plan.observations){if(observation.source!="artifact_section")continue;const auto value=json::parse(observation.summary,nullptr,false);if(!value.is_object()||value.value("id",std::string{})!=profile_section.id||!value.contains("blocks")||!value["blocks"].is_array())continue;
            // A completed section is reusable only for the exact frozen
            // source snapshot from which it was generated. In particular,
            // never let a previous explicit report revision silently carry
            // prose across a changed research snapshot.
            if(value.value("snapshot_fingerprint",std::string{})!=snapshot.fingerprint)continue;
            found=true;section.no_findings=value.value("no_findings",false);
            for(const auto & ref:value.value("omitted_source_refs",json::array()))if(ref.is_string()){report_coverage=false;out.unresolved_gaps.push_back("context_omission:"+profile_section.id+":"+ref.get<std::string>());}
            for(const auto & bj:value["blocks"]){common_agent_artifact_block b;b.id=bj.value("id",std::string{});b.text=bj.value("text",std::string{});b.validation_note=bj.value("validation_note",std::string{});const auto kind=bj.value("kind",std::string("observation"));if(!parse_kind(kind,b.kind)){error="stored artifact block kind is invalid";return false;}for(const auto & x:bj.value("evidence_ids",json::array()))if(x.is_string())b.evidence_ids.push_back(x.get<std::string>());for(const auto & x:bj.value("source_refs",json::array()))if(x.is_string())b.source_refs.push_back(x.get<std::string>());common_agent_artifact_block checked;if(!common_agent_artifact_validate_block(b,snapshot,checked,error))return false;if(checked.validation!=common_agent_artifact_validation::references_resolve){report_coverage=false;out.unresolved_gaps.push_back("section_reference_gap:"+profile_section.id);}section.blocks.push_back(std::move(checked));}}
        if(!found||(!section.no_findings&&section.blocks.empty())){common_agent_artifact_block gap;gap.id=profile_section.id+":gap";gap.kind=common_agent_artifact_block_kind::observation;gap.text="No validated content was produced for this section.";gap.validation=common_agent_artifact_validation::gap;gap.validation_note="Section synthesis is missing from the source plan.";gap.origin="host";section.blocks.push_back(std::move(gap));report_coverage=false;out.unresolved_gaps.push_back("missing_section:"+profile_section.id);}
        out.sections.push_back(std::move(section));}
    out.coverage_complete=report_coverage&&out.unresolved_gaps.empty();
    error.clear();return true;
}

bool common_agent_artifact_document_from_json(const std::string & text, common_agent_artifact_document & d, std::string & error) {
    const auto v=json::parse(text,nullptr,false);
    if (!v.is_object() || v.value("schema_version",0)!=1 || !v.contains("sections") || !v["sections"].is_array()) { error="unsupported or malformed artifact document"; return false; }
    d={}; d.id=v.value("id",std::string{}); d.title=v.value("title",std::string{}); d.purpose=v.value("purpose",std::string{});
    d.profile_id=v.value("profile_id",std::string{}); d.profile_revision=v.value("profile_revision",std::string{});
    d.revision=v.value("revision",uint64_t(1)); d.previous_revision_uri=v.value("previous_revision_uri",std::string{});
    d.snapshot_uri=v.value("snapshot_uri",std::string{});d.snapshot_fingerprint=v.value("snapshot_fingerprint",std::string{}); d.source_plan_id=v.value("source_plan_id",std::string{}); d.source_plan_version=v.value("source_plan_version",uint64_t(0));
    d.coverage_complete=v.value("coverage_complete",false);
    for(const auto & gap:v.value("unresolved_gaps",json::array())){if(!gap.is_string()){error="artifact document gap is not text";return false;}d.unresolved_gaps.push_back(gap.get<std::string>());}
    for (const auto & sj : v["sections"]) {
        if (!sj.is_object() || !sj.contains("blocks") || !sj["blocks"].is_array()) { error="artifact section is malformed"; return false; }
        common_agent_artifact_section s; s.id=sj.value("id",std::string{}); s.title=sj.value("title",std::string{});s.no_findings=sj.value("no_findings",false);
        for (const auto & bj : sj["blocks"]) {
            if (!bj.is_object() || !bj.contains("evidence_ids") || !bj["evidence_ids"].is_array() || !bj.contains("source_refs") || !bj["source_refs"].is_array()) { error="artifact block is malformed"; return false; }
            common_agent_artifact_block b; b.id=bj.value("id",std::string{}); b.text=bj.value("text",std::string{});
            b.origin=bj.value("origin",std::string("model")); b.validation_note=bj.value("validation_note",std::string{});
            if (!parse_kind(bj.value("kind",std::string{}),b.kind) || !parse_validation(bj.value("validation",std::string("unvalidated")),b.validation) || !parse_support(bj.value("support",std::string("unverified")),b.support)) { error="artifact block enum is invalid"; return false; }
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
        if(b.validation==common_agent_artifact_validation::gap||b.validation==common_agent_artifact_validation::contradicted) out+="  \n  **Reference status: "+std::string(validation_name(b.validation))+" — "+b.validation_note+"**";
        out+="  \n  Evidence relation: "+std::string(support_name(b.support))+" (reference resolution does not prove entailment).";
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
