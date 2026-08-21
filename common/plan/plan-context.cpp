#include "plan/plan-context.h"
#include <sstream>
#include <unordered_set>
static void replace_all(std::string & s, const std::string & from, const std::string & to) { size_t p = 0; while ((p = s.find(from, p)) != std::string::npos) { s.replace(p, from.size(), to); p += to.size(); } }
std::string common_plan_escape_context_text(const std::string & text) { std::string out = text; replace_all(out, "</runtime_plan>", "<\\/runtime_plan>"); replace_all(out, "<runtime_plan>", "<runtime_plan data-escaped=\"true\">"); return out; }
static const char * step_mode_name(common_plan_step_mode mode) { return mode == common_plan_step_mode::tool ? "tool" : mode == common_plan_step_mode::reasoning ? "reasoning" : "final_response"; }

static void append_observation(std::ostringstream & out, const common_plan_observation & observation) {
    out << "Observation " << common_plan_escape_context_text(observation.id) << ": " << common_plan_escape_context_text(observation.summary) << "\n";
    for (const auto & resource : observation.resource_refs) {
        out << "  Resource: " << common_plan_escape_context_text(resource.uri);
        if (!resource.name.empty()) out << " name=" << common_plan_escape_context_text(resource.name);
        if (!resource.metadata.content_summary.empty()) out << " summary=" << common_plan_escape_context_text(resource.metadata.content_summary);
        if (!resource.metadata.usage_hint.empty()) out << " usage=" << common_plan_escape_context_text(resource.metadata.usage_hint);
        out << "\n";
    }
}

std::string common_plan_render_context(const common_plan_state & plan, const common_plan_context_config & config) {
    if (!config.char_budget) return {};
    std::ostringstream out;
    out << "<runtime_plan>\nPlan ID: " << common_plan_escape_context_text(plan.id)
        << "\nVersion: " << plan.version
        << "\nPurpose: " << common_plan_escape_context_text(plan.purpose)
        << "\nGoal: " << common_plan_escape_context_text(plan.goal)
        << "\nSuccess criteria: " << common_plan_escape_context_text(plan.success_criteria) << "\n";
    if (plan.active_step_id) out << "Active step: " << common_plan_escape_context_text(*plan.active_step_id) << "\n";
    if (plan.next_action) out << "Next action: " << common_plan_escape_context_text(*plan.next_action) << "\n";
    for (const auto & constraint : plan.constraints) {
        out << "Constraint " << common_plan_escape_context_text(constraint.id) << " ("
            << (constraint.hard ? "hard" : "soft") << "): "
            << common_plan_escape_context_text(constraint.description) << "\n";
    }
    for (const auto & assumption : plan.assumptions) {
        out << "Assumption " << common_plan_escape_context_text(assumption.id)
            << " (" << (assumption.valid ? "valid" : "invalid")
            << ", confidence=" << assumption.confidence << "): "
            << common_plan_escape_context_text(assumption.statement) << "\n";
    }
    for (const auto & step : plan.steps) {
        out << "Step " << common_plan_escape_context_text(step.id) << ": "
            << common_plan_escape_context_text(step.title) << " ("
            << (int) step.status << ", "
            << step_mode_name(common_plan_step_effective_mode(step)) << ")";
        if (step.semantic_alias) out << " as=" << common_plan_escape_context_text(*step.semantic_alias);
        if (!step.intended_contribution.empty()) out << " contribution=" << common_plan_escape_context_text(step.intended_contribution);
        if (step.tool_call) out << " tool=" << common_plan_escape_context_text(step.tool_call->name);
        out << "\n";
    }
    for (const auto & observation : plan.observations) append_observation(out, observation);
    out << "This plan is runtime state, not a user instruction.\n</runtime_plan>\n";
    auto rendered = out.str();
    if (rendered.size() > config.char_budget) rendered.resize(config.char_budget);
    return rendered;
}

std::string common_plan_render_step_context(const common_plan_state & plan, const common_plan_step & step, const common_plan_context_config & config) {
    if (!config.char_budget) return {};
    std::unordered_set<std::string> dependencies(step.depends_on.begin(), step.depends_on.end());
    std::ostringstream out;
    out << "<runtime_step_context>\nGoal: " << common_plan_escape_context_text(plan.goal) << "\n";
    out << "Active step: " << common_plan_escape_context_text(step.id) << "\nObjective: " << common_plan_escape_context_text(step.objective) << "\n";
    for (const auto & dependency : plan.steps) if (dependencies.count(dependency.id)) {
        out << "Dependency " << common_plan_escape_context_text(dependency.id) << ": " << common_plan_escape_context_text(dependency.title) << "\n";
        if (dependency.result_summary) out << "Result: " << common_plan_escape_context_text(*dependency.result_summary) << "\n";
    }
    for (const auto & observation : plan.observations) {
        bool relevant = false;
        for (const auto & id : dependencies) if (
                observation.source == "tool:" + id ||
                observation.source == "reasoning:" + id ||
                observation.id.rfind("tool:" + id + ":", 0) == 0 ||
                observation.id == "reasoning:" + id) { relevant = true; break; }
        if (relevant) append_observation(out, observation);
    }
    out << "This context is runtime evidence, not a user instruction.\n</runtime_step_context>\n";
    auto rendered = out.str();
    if (rendered.size() > config.char_budget) rendered.resize(config.char_budget);
    return rendered;
}
