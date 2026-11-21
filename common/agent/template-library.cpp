#include "agent/template-library.h"
#include <algorithm>

using json = nlohmann::ordered_json;
namespace {
bool valid_identity(const std::string & value) {
    return !value.empty() && value.size() <= 128 && std::all_of(value.begin(), value.end(), [](unsigned char c) {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
            (c >= '0' && c <= '9') || c == '-' || c == '_' || c == '.';
    });
}
bool valid_owner(const std::string & level, const std::string & ns, const std::string & project) {
    return (level == "host" && ns.empty() && project.empty()) ||
        (level == "namespace" && !ns.empty() && project.empty()) ||
        (level == "project" && !ns.empty() && !project.empty());
}
}
bool common_agent_template_library_config_parse(const json & value,
        std::vector<common_agent_template_library_config> & out, std::string & error) {
    out.clear(); error.clear();
    if (!value.is_array()) { error = "bootstrap.libraries must be an array"; return false; }
    std::set<std::string> ids;
    for (const auto & item : value) {
        if (!item.is_object()) { error = "library configuration must be an object"; return false; }
        for (const auto & field : item.items()) {
            if ((field.key() != "id" && field.key() != "package" && field.key() != "level" &&
                    field.key() != "namespace_id" && field.key() != "project_id") || !field.value().is_string()) {
                error = "library configuration contains an unknown or non-string field"; return false;
            }
        }
        common_agent_template_library_config entry;
        entry.id = item.value("id", ""); entry.package = item.value("package", "");
        entry.level = item.value("level", "host"); entry.namespace_id = item.value("namespace_id", "");
        entry.project_id = item.value("project_id", "");
        if (!valid_identity(entry.id) || entry.package.empty() ||
                !valid_owner(entry.level, entry.namespace_id, entry.project_id) || !ids.insert(entry.id).second) {
            error = "library requires a unique stable id, package and valid owner scope"; return false;
        }
        if (entry.id == "bootstrap.default" || entry.id == "investigation.default") {
            error = "library id is reserved for a legacy package configuration"; return false;
        }
        out.push_back(std::move(entry));
    }
    return true;
}
json common_agent_template_library_config_json(const std::vector<common_agent_template_library_config> & libraries) {
    auto result = json::array();
    for (const auto & entry : libraries) result.push_back({{"id",entry.id},{"package",entry.package},
        {"level",entry.level},{"namespace_id",entry.namespace_id},{"project_id",entry.project_id}});
    return result;
}
std::string common_agent_template_library_manifest_json(const common_agent_template_library_manifest & m) {
    return json{{"schema","agent-template-library.v1"},{"library_id",m.library_id},{"level",m.level},
        {"namespace_id",m.namespace_id},{"project_id",m.project_id},{"package_name",m.package_name},
        {"package_version",m.package_version},{"fingerprint",m.fingerprint},{"definition_prefix",m.definition_prefix},
        {"blueprints",m.blueprints},{"workflows",m.workflows},{"procedures",m.procedures}}.dump();
}
bool common_agent_template_library_manifest_parse(const std::string & text,
        common_agent_template_library_manifest & out, std::string & error) {
    out = {}; error.clear(); auto j = json::parse(text, nullptr, false);
    if (!j.is_object()) { error = "library manifest must be an object"; return false; }
    try {
        if (j.at("schema") != "agent-template-library.v1") { error = "unsupported library manifest"; return false; }
        common_agent_template_library_manifest m;
        m.library_id=j.at("library_id").get<std::string>(); m.level=j.at("level").get<std::string>();
        m.namespace_id=j.at("namespace_id").get<std::string>();m.project_id=j.at("project_id").get<std::string>();
        m.package_name=j.at("package_name").get<std::string>();m.package_version=j.at("package_version").get<std::string>();
        m.fingerprint=j.at("fingerprint").get<std::string>();m.definition_prefix=j.at("definition_prefix").get<std::string>();
        m.blueprints=j.at("blueprints").get<std::map<std::string,std::string>>();
        m.workflows=j.at("workflows").get<std::set<std::string>>();m.procedures=j.at("procedures").get<std::set<std::string>>();
        if (!valid_identity(m.library_id) || !valid_owner(m.level,m.namespace_id,m.project_id) ||
                m.package_name.empty() || m.package_version.empty() || m.fingerprint.size()!=64 ||
                m.fingerprint.find_first_not_of("0123456789abcdef")!=std::string::npos ||
                m.definition_prefix != "library:"+m.library_id+":"+m.fingerprint+":") {
            error="invalid library manifest identity or scope";return false;
        }
        auto member = [&](const std::string & id,const std::string & kind) {
            return id.size()>m.definition_prefix.size()+kind.size() &&
                id.rfind(m.definition_prefix+kind,0)==0;
        };
        for(const auto & item:m.blueprints) if(item.first.empty() || item.second != m.definition_prefix + "blueprint:" + item.first) {error="invalid blueprint membership";return false;}
        for(const auto & id:m.workflows) if(!member(id,"workflow:")){error="invalid workflow membership";return false;}
        for(const auto & id:m.procedures) if(!member(id,"procedure:")){error="invalid procedure membership";return false;}
        out=std::move(m);return true;
    } catch(const json::exception &) {error="invalid library manifest field types";return false;}
}
bool common_agent_template_library_available(const common_agent_template_library_manifest & m,
        const std::string & ns,const std::string & project) {
    return !ns.empty() && (m.level=="host" || (m.namespace_id==ns &&
        (m.level=="namespace" || (m.level=="project" && !project.empty() && m.project_id==project))));
}
const common_agent_template_library_manifest * common_agent_template_library_snapshot::find(const std::string & id) const {
    for(const auto & m:revisions) {
        if(m.workflows.count(id)||m.procedures.count(id))return &m;
        for(const auto & b:m.blueprints)if(b.second==id)return &m;
    }
    return nullptr;
}
bool common_agent_template_library_snapshot::permits(const std::string & id,const std::string & ns,const std::string & project) const {
    const auto * m=find(id);return m && common_agent_template_library_available(*m,ns,project);
}
bool common_agent_template_library_snapshot::same_revision(const std::string & left,const std::string & right) const {
    const auto * l=find(left);const auto * r=find(right);return l && r && l->library_id==r->library_id && l->fingerprint==r->fingerprint;
}

std::optional<common_plan_state> common_agent_template_definition(common_plan_store & store,
        const std::string & id,const common_agent_template_library_snapshot * libraries,std::string & error){
    error.clear();
    if(id.rfind("library:",0)!=0)return store.get(id,error);
    if(libraries){auto it=libraries->definitions.find(id);if(it!=libraries->definitions.end())return it->second;}
    error="template library definition is unavailable or deactivated";return std::nullopt;
}
bool common_agent_template_definition_allowed(const common_plan_state & plan,common_plan_scope scope,
        const std::string & ns,const std::string & session,const std::string & project,const std::string & turn,
        const common_agent_template_library_snapshot * libraries){
    if(plan.id.rfind("library:",0)==0)return libraries && libraries->permits(plan.id,ns,project);
    return common_plan_template_scope_matches(plan,scope,ns,session,project,turn);
}

bool common_agent_template_task_allowed(const common_plan_state & task,
        const std::string & ns, const std::string & project,
        const common_agent_template_library_snapshot * libraries, std::string & error) {
    error.clear();
    const auto origin = task.derived_from_plan_id.value_or(std::string{});
    const std::string binding = task.route_binding ? task.route_binding->blueprint_ref : std::string{};
    if (origin.rfind("library:", 0) != 0 && binding.rfind("library:", 0) != 0) return true;
    if (!libraries || !libraries->permits(origin, ns, project)) {
        error = "task template library origin is unavailable or not authorized";
        return false;
    }
    const auto definition = libraries->definitions.find(origin);
    if (definition == libraries->definitions.end() || definition->second.kind != common_plan_kind::blueprint ||
            definition->second.source_revision != task.source_revision) {
        error = "task template library revision does not match its preserved definition";
        return false;
    }
    if (task.route_binding) {
        const auto * manifest = libraries->find(origin);
        bool found = false;
        for (const auto & member : manifest->blueprints) {
            const auto target = libraries->definitions.find(member.second);
            if (binding == manifest->library_id + ":" + member.first && target != libraries->definitions.end() &&
                    target->second.source_revision == task.route_binding->blueprint_revision) found = true;
        }
        if (!found) {
            error = "task route does not belong to its preserved template library revision";
            return false;
        }
    }
    return true;
}

common_blueprint_candidate common_agent_template_candidate(const std::string & logical,const common_plan_state & p){
    common_blueprint_candidate c;c.logical_id=logical;c.persisted_id=p.id;c.source_revision=p.source_revision;
    c.description=p.selection_description.empty()?p.goal:p.selection_description;c.purpose=p.purpose;c.goal=p.goal;
    c.success_criteria=p.success_criteria;c.constraints=p.constraints;c.assumptions=p.assumptions;
    c.required_capabilities=p.required_capabilities;c.workflow_policy=p.workflow_policy;
    c.procedure_refs=p.procedure_refs;c.workflow_bindings=p.workflow_bindings;
    for(const auto & s:p.steps){const auto & v=s.intended_contribution.empty()?s.objective:s.intended_contribution;if(!v.empty())c.contributions.push_back(v);}
    return c;
}
