#include "method-phases.h"
#include "continuation-control.h"
#include "plan/plan-context.h"
#include <algorithm>
#include <map>
#include <set>

using json = nlohmann::ordered_json;
namespace {
json outline(const common_plan_state & plan) {
    for (const auto & observation : plan.observations) {
        if (observation.source != "blueprint_outline") continue;
        auto value = json::parse(observation.summary, nullptr, false);
        if (value.is_array()) return value;
    }
    return json::array();
}
json saved_progress(const common_plan_state & plan) {
    for (auto it=plan.observations.rbegin(); it!=plan.observations.rend(); ++it) {
        if (it->source != "method_phase_review") continue;
        auto value=json::parse(it->summary,nullptr,false);
        if (value.is_object() && value.value("schema",std::string{})=="method-phase-review.v1" &&
            value.value("method_revision",plan.source_revision)==plan.source_revision &&
            value.contains("phases") && value["phases"].is_object()) return value["phases"];
    }
    return json::object();
}
bool closed(const json & phases, const std::string & id) {
    if (!phases.contains(id) || !phases[id].is_object()) return false;
    const auto status=phases[id].value("status",std::string{});
    return status=="satisfied" || status=="not_needed" || status=="blocked";
}
bool remap(json & value, const std::map<std::string,std::string> & ids) {
    if (value.is_object()) {
        if (value.contains("$from_step")) {
            if (!value["$from_step"].is_string()) return false;
            const auto id=ids.find(value["$from_step"].get<std::string>());
            if (id==ids.end()) return false;
            value["$from_step"]=id->second;
        }
        for (auto & item:value.items()) if (!remap(item.value(),ids)) return false;
    } else if (value.is_array()) for (auto & item:value) if (!remap(item,ids)) return false;
    return true;
}
}

bool common_agent_method_review(const common_plan_state & plan,
        const std::vector<common_agent_method_phase_review> & reviews, json & progress, std::string & error) {
    error.clear(); progress=saved_progress(plan);
    if (!plan.phase_review_required) {
        if (!reviews.empty()) { error="phase reviews require a host-selected phased method"; return false; }
        return true;
    }
    const auto phases=outline(plan);
    if (phases.empty()) { error="phased method is missing its retained outline"; return false; }
    if (reviews.size()>16) { error="too many method phase reviews"; return false; }
    std::set<std::string> updated;
    for (const auto & review:reviews) {
        const auto phase=std::find_if(phases.begin(),phases.end(),[&](const auto & item) {
            return item.is_object() && item.value("id",std::string{})==review.phase_id;
        });
        if (phase==phases.end() || !updated.insert(review.phase_id).second) { error="unknown or duplicate method phase"; return false; }
        if (review.status!="satisfied" && review.status!="not_needed" && review.status!="blocked" && review.status!="remaining") {
            error="invalid method phase status"; return false;
        }
        if (review.reason.empty() || review.reason.size()>512 || review.evidence_ids.size()>8) { error="invalid method phase explanation or evidence bounds"; return false; }
        for (const auto & id:review.evidence_ids)
            if (std::none_of(plan.observations.begin(),plan.observations.end(),[&](const auto & observation){
                    return observation.id==id && observation.source!="method_phase_review" &&
                        observation.source!="blueprint_outline" && observation.source!="response_review";
                })) {
                error="method phase cites unknown evidence"; return false;
            }
        progress[review.phase_id]={{"status",review.status},{"reason",review.reason},{"evidence_ids",review.evidence_ids}};
    }
    // Evaluate dependency closure after collecting the whole review: the order
    // of independent verdicts in the model's array is not an execution order.
    for (const auto & review:reviews) {
        if (review.status!="satisfied" && review.status!="blocked") continue;
        const auto phase=std::find_if(phases.begin(),phases.end(),[&](const auto & item) {
            return item.is_object() && item.value("id",std::string{})==review.phase_id;
        });
        for (const auto & dependency:phase->value("depends_on",json::array())) {
            if (!dependency.is_string() || !closed(progress,dependency.get<std::string>())) {
                error="method phase precedes unresolved dependency"; return false;
            }
        }
    }
    // Reopening a prerequisite invalidates descendant completion; do not let
    // an old later-stage verdict silently survive new unresolved work.
    bool removed=false;
    do {
        removed=false;
        for (const auto & phase:phases) {
            for (const auto & dependency:phase.value("depends_on",json::array())) {
                if (!dependency.is_string() || !closed(progress,dependency.get<std::string>())) {
                    removed=progress.erase(phase.value("id",std::string{}))>0 || removed; break;
                }
            }
        }
    } while (removed);
    return true;
}

std::string common_agent_render_method_progress(const common_plan_state & plan) {
    if (!plan.phase_review_required) return {};
    return "\n[Host method phases and retained decisions]\n"+
        json{{"schema","method-phase-context.v1"},{"outline",outline(plan)},{"progress",saved_progress(plan)}}.dump()+
        "\nAssess phases in dependency order. Report phase_reviews using exact phase IDs, status "
        "satisfied/not_needed/blocked/remaining, a reason and known evidence_ids. A phase may need no tool. "
        "Satisfied describes the phase objective, never fabricated tool execution. "
        "Describe remaining needs in phase reasons and revision guidance. Multiple phases may remain unresolved. "
        "The existing planner selects the next ready phase and its operations. "
        "Use authorized research escalation for material knowledge gaps. "
        "Close unavailable or unauthorized operations as blocked and disclose the limitation. "
        "All phases require a decision before a final answer can be accepted.\n";
}

bool common_agent_method_has_remaining(const common_plan_state & plan) {
    if (!plan.phase_review_required) return false;
    const auto phases=outline(plan), progress=saved_progress(plan);
    return phases.empty() || std::any_of(phases.begin(),phases.end(),[&](const auto & phase) {
        return !closed(progress,phase.value("id",std::string{}));
    });
}

nlohmann::ordered_json common_agent_method_ready_phases(const common_plan_state & plan) {
    auto ready=json::array();
    const auto progress=saved_progress(plan);
    for (const auto & phase:outline(plan)) {
        const auto id=phase.value("id",std::string{});
        if (closed(progress,id)) continue;
        const auto dependencies=phase.value("depends_on",json::array());
        if (std::any_of(dependencies.begin(),dependencies.end(),[&](const json & dependency) {
                return !dependency.is_string() || !closed(progress,dependency.get<std::string>());
            })) continue;
        ready.push_back({{"handle","p"+std::to_string(ready.size()+1)},{"phase_id",id},{"phase",phase}});
    }
    return ready;
}

bool common_agent_method_resolve_choice(const common_agent_request & request,const std::string & handle,
        std::string & phase,std::string & error) {
    error.clear();
    for (const auto & choice:request.method_phase_choices) if (choice.first==handle) {
        phase=choice.second;return true;
    }
    error="continuation must select one host-issued ready phase; available handles:";
    for (const auto & choice:request.method_phase_choices) error+=" "+choice.first;
    return false;
}

bool common_agent_method_parse_selection(const common_agent_request & request,bool compact_dsl,
        std::string & text,std::string & phase,std::string & error) {
    auto body=text;
    std::string handle,selected;
    if (compact_dsl) {
        const auto newline=body.find('\n');
        if (body.rfind("phase ",0)==0 && newline!=std::string::npos) {
            handle=body.substr(6,newline-6);
            if (!handle.empty() && handle.back()=='\r') handle.pop_back();
            body.erase(0,newline+1);
        }
    } else {
        auto value=json::parse(body,nullptr,false);
        if (value.is_object() && value.contains("method_phase") && value["method_phase"].is_string()) {
            handle=value["method_phase"].get<std::string>();
            value.erase("method_phase");body=value.dump();
        }
    }
    if (!common_agent_method_resolve_choice(request,handle,selected,error)) return false;
    text=std::move(body);phase=std::move(selected);return true;
}

bool common_agent_method_validate_progress(const common_plan_state & plan,
        const std::vector<common_plan_operation> & operations,const std::string & phase,const std::string & turn,
        const std::function<bool(const std::string &)> & is_read_only,std::string & error) {
    error.clear();
    // Without a durable turn identity, old work may belong to a new user request.
    if (turn.empty()) return true;
    const bool has_tool=std::any_of(operations.begin(),operations.end(),[](const auto & operation) {
        return operation.step && operation.step->tool_call.has_value();
    });
    for (const auto & operation:operations) {
        if (!operation.step) continue;
        const auto & proposed=*operation.step;
        if (!proposed.tool_call && has_tool) continue;
        if (proposed.tool_call && (!is_read_only || !is_read_only(proposed.tool_call->name))) continue;
        size_t repeated=0;
        nlohmann::json last_result;
        for (auto it=plan.steps.rbegin();it!=plan.steps.rend();++it) {
            if (common_plan_step_effective_mode(*it)==common_plan_step_mode::final_response) continue;
            if (it->status!=common_plan_step_status::completed || it->method_phase_id!=phase ||
                    it->method_turn_id.value_or("")!=turn) break;
            std::string observation_id;
            if (proposed.tool_call) {
                if (!it->tool_call || it->tool_call->name!=proposed.tool_call->name) break;
                const auto args=nlohmann::json::parse(proposed.tool_call->arguments_json,nullptr,false);
                const auto old=nlohmann::json::parse(it->tool_call->arguments_json,nullptr,false);
                // Deferred bindings can resolve to a newer resource.
                if (!args.is_object() || args.dump().find("$from_step")!=std::string::npos || old!=args) break;
                observation_id="tool:"+it->id+":"+it->tool_call->name;
            } else {
                if (it->tool_call || it->objective!=proposed.objective) break;
                observation_id="reasoning:"+it->id;
            }
            const auto observation=std::find_if(plan.observations.rbegin(),plan.observations.rend(),[&](const auto & item) {
                return item.id==observation_id || (proposed.tool_call && item.id.rfind(observation_id+":attempt:",0)==0);
            });
            if (observation==plan.observations.rend() ||
                    observation->source!=(proposed.tool_call?proposed.tool_call->name:"reasoning")) break;
            const auto current=nlohmann::json::parse(observation->summary,nullptr,false);
            // A truncated projection cannot prove that the full result is unchanged.
            if (current.is_discarded() || (current.is_object() && current.contains("result_truncated") &&
                    (!current["result_truncated"].is_boolean() || current["result_truncated"].get<bool>()))) break;
            if (repeated && last_result!=current) break;
            last_result=current;
            ++repeated;
        }
        if (repeated>=2) {
            error=proposed.tool_call
                ? "method continuation repeated the same read twice with unchanged results; inspect a returned source, continue its paging, or select different evidence"
                : "method continuation repeated the same reasoning twice with unchanged results; resolve the remaining need or report its limitation";
            return false;
        }
    }
    return true;
}

bool common_agent_method_append_slice(const common_plan_state & plan,const common_plan_state & slice,
        std::vector<common_plan_operation> & operations,std::string & error,
        const std::map<std::string,std::string> & repair_targets) {
    operations.clear();error.clear();
    if (!plan.phase_review_required || plan.workflow_definition) { error="phase planning requires an unbound phased method";return false; }
    std::map<std::string,std::string> ids;
    for (const auto & old:plan.steps) ids[old.id]=old.id;
    size_t count=0;
    std::set<std::string> local;
    std::set<std::string> assigned;
    for (const auto & target : repair_targets)
        if (std::none_of(slice.steps.begin(),slice.steps.end(),[&](const auto & step){return step.id==target.first;})) {
            error="repair refers to an unknown slice step";return false;
        }
    for (const auto & step:slice.steps) {
        if (common_plan_step_effective_mode(step)==common_plan_step_mode::final_response) continue;
        if (step.id.empty() || !local.insert(step.id).second || ++count>8) {error="invalid bounded phase slice";return false;}
        const common_plan_step * failed=nullptr;
        const auto target=repair_targets.find(step.id);
        if (target!=repair_targets.end()) {
            const auto prior=std::find_if(plan.steps.begin(),plan.steps.end(),[&](const auto & old){return old.id==target->second;});
            if (prior==plan.steps.end() || prior->status!=common_plan_step_status::failed ||
                    !prior->tool_call || !step.tool_call || prior->tool_call->name!=step.tool_call->name) {
                error="explicit repair must bind a failed step and retain its tool";return false;
            }
            failed=&*prior;
        }
        const auto identity=failed?failed->id:plan.id+":method:"+std::to_string(plan.version)+":"+std::to_string(count);
        if (!assigned.insert(identity).second || (!failed && std::any_of(plan.steps.begin(),plan.steps.end(),
                [&](const auto & prior){return prior.id==identity;}))) {
            error="phase slice assigns the same execution identity twice";return false;
        }
        ids[step.id]=identity;
    }
    if (!count) {error="phase planner produced no executable or reasoning work";return false;}
    for (auto step:slice.steps) {
        if (common_plan_step_effective_mode(step)==common_plan_step_mode::final_response) continue;
        step.id=ids.at(step.id);step.status=common_plan_step_status::pending;step.blocked_by.clear();step.result_summary.reset();
        for (auto & dependency:step.depends_on) {
            const auto found=ids.find(dependency);
            if (found==ids.end()) {error="phase slice depends on an unknown or final step";return false;}
            dependency=found->second;
        }
        if (step.tool_call) {
            const auto prior=std::find_if(plan.steps.begin(),plan.steps.end(),[&](const auto & old){return old.id==step.id;});
            if (prior!=plan.steps.end() && prior->tool_call)
                step.tool_call->result_requirement_json=prior->tool_call->result_requirement_json;
            if (prior!=plan.steps.end()) {
                step.optional=prior->optional;
                step.required_evidence=prior->required_evidence;
                step.source_memory_ids=prior->source_memory_ids;
                step.generated_from_memory=prior->generated_from_memory;
                step.created_at=prior->created_at;
                step.semantic_alias=prior->semantic_alias;
                step.artifact_binding=prior->artifact_binding;
                for (const auto & dependency:prior->depends_on)
                    if (std::find(step.depends_on.begin(),step.depends_on.end(),dependency)==step.depends_on.end())
                        step.depends_on.push_back(dependency);
            }
            auto arguments=json::parse(step.tool_call->arguments_json,nullptr,false);
            if (!arguments.is_object() || !remap(arguments,ids)) {error="phase slice has an unknown argument binding";return false;}
            for (const auto & old:plan.steps) {
                if (old.status!=common_plan_step_status::failed || !old.tool_call ||
                        old.tool_call->name!=step.tool_call->name) continue;
                const auto failed_arguments=json::parse(old.tool_call->arguments_json,nullptr,false);
                if (!failed_arguments.is_discarded() && failed_arguments==arguments) {
                    if (old.id == step.id && common_agent_continuation_allows_same_call(plan, old.id)) continue;
                    error="phase repair cannot blindly replay the same failed arguments";return false;
                }
            }
            step.tool_call->arguments_json=arguments.dump();
        }
        common_plan_operation add;
        const bool replacement=std::any_of(plan.steps.begin(),plan.steps.end(),[&](const auto & prior){return prior.id==step.id;});
        add.kind=replacement?common_plan_operation_kind::replace_step:common_plan_operation_kind::add_step;
        if (replacement) add.step_id=step.id;
        add.step=std::move(step);
        add.reason_summary="bounded model-planned method work";operations.push_back(std::move(add));
    }
    std::vector<common_plan_operation> ordered;
    std::set<std::string> ready;
    for (const auto & prior:plan.steps) {
        if (std::none_of(operations.begin(),operations.end(),[&](const auto & op){return op.step->id==prior.id;})) ready.insert(prior.id);
    }
    while (ordered.size()<operations.size()) {
        bool progress=false;
        for (const auto & operation:operations) {
            if (std::any_of(ordered.begin(),ordered.end(),[&](const auto & done){return done.step->id==operation.step->id;})) continue;
            if (!std::all_of(operation.step->depends_on.begin(),operation.step->depends_on.end(),
                    [&](const auto & id){return ready.count(id)!=0;})) continue;
            ordered.push_back(operation);ready.insert(operation.step->id);progress=true;
        }
        if (!progress) {error="phase slice contains a dependency cycle";return false;}
    }
    operations=std::move(ordered);
    return true;
}
