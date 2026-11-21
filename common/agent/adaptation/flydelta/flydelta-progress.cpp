#include "agent/adaptation/flydelta/flydelta-progress.h"

#include <algorithm>
#include <nlohmann/json.hpp>

namespace {

std::vector<std::string> goal_universe(const common_flydelta_progress_assessment & assessment) {
    auto goals = assessment.satisfied_goals;
    goals.insert(goals.end(), assessment.remaining_goals.begin(), assessment.remaining_goals.end());
    std::sort(goals.begin(), goals.end());
    return goals;
}

} // namespace

bool common_flydelta_progress_assessment_validate(
        const common_flydelta_progress_assessment & assessment, std::string & error) {
    error.clear();
    if (assessment.schema_version != 1 || assessment.objective_id.size() > 512 ||
            assessment.objective_revision.size() > 512 || assessment.verifier_ref.size() > 512 ||
            (assessment.regression_known && !assessment.assessment_known) ||
            (assessment.target_completed && (!assessment.assessment_known || !assessment.remaining_goals.empty()))) {
        error = "FlyDelta progress assessment identity or coverage is invalid";
        return false;
    }
    const auto goals = goal_universe(assessment);
    if (goals.size() > 1024 || std::any_of(goals.begin(), goals.end(), [](const auto & goal) {
            return goal.empty() || goal.size() > 512;
        }) || std::adjacent_find(goals.begin(), goals.end()) != goals.end()) {
        error = "FlyDelta progress goals are duplicated, contradictory or unbounded";
        return false;
    }
    if (assessment.assessment_known && (assessment.objective_id.empty() ||
            assessment.objective_revision.empty() || assessment.verifier_ref.empty() || goals.empty())) {
        error = "FlyDelta known progress requires an objective, verifier and declared goals";
        return false;
    }
    return true;
}

bool common_flydelta_progress_comparable(
        const common_flydelta_progress_assessment & left,
        const common_flydelta_progress_assessment & right) {
    std::string error;
    return common_flydelta_progress_assessment_validate(left, error) &&
        common_flydelta_progress_assessment_validate(right, error) &&
        left.assessment_known && right.assessment_known &&
        left.objective_id == right.objective_id && left.objective_revision == right.objective_revision &&
        left.verifier_ref == right.verifier_ref && goal_universe(left) == goal_universe(right);
}

bool common_flydelta_progress_strictly_dominates(
        const common_flydelta_progress_assessment & incumbent,
        const common_flydelta_progress_assessment & candidate) {
    if (!common_flydelta_progress_comparable(incumbent, candidate) ||
            !incumbent.regression_known || !candidate.regression_known ||
            incumbent.has_regression || candidate.has_regression ||
            candidate.satisfied_goals.size() <= incumbent.satisfied_goals.size()) return false;
    return std::all_of(incumbent.satisfied_goals.begin(), incumbent.satisfied_goals.end(),
        [&](const auto & goal) {
            return std::find(candidate.satisfied_goals.begin(), candidate.satisfied_goals.end(), goal) !=
                candidate.satisfied_goals.end();
        });
}

std::string common_flydelta_progress_assessment_to_json(
        const common_flydelta_progress_assessment & assessment) {
    return nlohmann::ordered_json{{"schema_version", assessment.schema_version},
        {"objective_id", assessment.objective_id}, {"objective_revision", assessment.objective_revision},
        {"verifier_ref", assessment.verifier_ref}, {"assessment_known", assessment.assessment_known},
        {"regression_known", assessment.regression_known}, {"has_regression", assessment.has_regression},
        {"target_completed", assessment.target_completed}, {"satisfied_goals", assessment.satisfied_goals},
        {"remaining_goals", assessment.remaining_goals}}.dump();
}

bool common_flydelta_progress_assessment_from_json(
        const std::string & text, common_flydelta_progress_assessment & assessment, std::string & error) {
    assessment = {};
    error.clear();
    if (text.size() > 1024U * 1024U) {
        error = "FlyDelta progress assessment exceeds its byte bound";
        return false;
    }
    try {
        const auto value = nlohmann::ordered_json::parse(text);
        if (!value.is_object()) { error = "FlyDelta progress assessment must be an object"; return false; }
        assessment.schema_version = value.value("schema_version", 0);
        assessment.objective_id = value.value("objective_id", "");
        assessment.objective_revision = value.value("objective_revision", "");
        assessment.verifier_ref = value.value("verifier_ref", "");
        assessment.assessment_known = value.value("assessment_known", false);
        assessment.regression_known = value.value("regression_known", false);
        assessment.has_regression = value.value("has_regression", false);
        assessment.target_completed = value.value("target_completed", false);
        assessment.satisfied_goals = value.value("satisfied_goals", std::vector<std::string>{});
        assessment.remaining_goals = value.value("remaining_goals", std::vector<std::string>{});
    } catch (const std::exception & exception) {
        error = std::string("FlyDelta progress assessment is malformed: ") + exception.what();
        assessment = {};
        return false;
    }
    return common_flydelta_progress_assessment_validate(assessment, error);
}

const char * common_flydelta_progress_transition_name(common_flydelta_progress_transition value) {
    switch (value) {
        case common_flydelta_progress_transition::unknown: return "unknown";
        case common_flydelta_progress_transition::unchanged: return "unchanged";
        case common_flydelta_progress_transition::improved: return "improved";
        case common_flydelta_progress_transition::solved: return "solved";
        case common_flydelta_progress_transition::regressed: return "regressed";
    }
    return "unknown";
}

bool common_flydelta_progress_evidence_validate(const common_flydelta_progress_evidence & e,
        std::string & error) {
    error.clear();
    const auto reference = [](const std::string & value) { return !value.empty() && value.size() <= 2048; };
    if (e.schema_version != 1 || !reference(e.baseline_ref) || !reference(e.candidate_ref) ||
            !reference(e.baseline_evidence_ref) || !reference(e.candidate_evidence_ref) ||
            e.baseline_ref == e.candidate_ref ||
            (e.parent.has_value() != !e.parent_ref.empty()) || e.parent_ref.size() > 2048 ||
            !common_flydelta_progress_assessment_validate(e.baseline, error) ||
            !common_flydelta_progress_assessment_validate(e.candidate, error) ||
            (e.parent && !common_flydelta_progress_assessment_validate(*e.parent, error))) {
        if (error.empty()) error = "FlyDelta paired progress evidence is invalid";
        return false;
    }
    return true;
}

common_flydelta_progress_transition common_flydelta_progress_evidence_classify(
        const common_flydelta_progress_evidence & e) {
    using transition = common_flydelta_progress_transition;
    std::string error;
    if (!common_flydelta_progress_evidence_validate(e, error) ||
            !common_flydelta_progress_comparable(e.baseline, e.candidate) ||
            (e.parent && !common_flydelta_progress_comparable(*e.parent, e.candidate))) return transition::unknown;
    const auto lost_goal = [&](const common_flydelta_progress_assessment & before) {
        return std::any_of(before.satisfied_goals.begin(), before.satisfied_goals.end(), [&](const auto & goal) {
            return std::find(e.candidate.satisfied_goals.begin(), e.candidate.satisfied_goals.end(), goal) ==
                e.candidate.satisfied_goals.end();
        });
    };
    // A detected regression remains useful even when other controls are missing.
    if (e.baseline.has_regression || e.candidate.has_regression || lost_goal(e.baseline) ||
            (e.parent && (e.parent->has_regression || lost_goal(*e.parent)))) return transition::regressed;
    if (!e.baseline.regression_known || !e.candidate.regression_known ||
            (e.parent && !e.parent->regression_known)) return transition::unknown;
    if (e.candidate.target_completed) return transition::solved;
    const auto & incumbent = e.parent ? *e.parent : e.baseline;
    return common_flydelta_progress_strictly_dominates(incumbent, e.candidate)
        ? transition::improved : transition::unchanged;
}

std::string common_flydelta_progress_evidence_to_json(const common_flydelta_progress_evidence & e) {
    using json = nlohmann::ordered_json;
    json value = {{"schema_version", e.schema_version}, {"baseline_ref", e.baseline_ref},
        {"candidate_ref", e.candidate_ref}, {"baseline_evidence_ref", e.baseline_evidence_ref},
        {"candidate_evidence_ref", e.candidate_evidence_ref},
        {"baseline", json::parse(common_flydelta_progress_assessment_to_json(e.baseline))},
        {"candidate", json::parse(common_flydelta_progress_assessment_to_json(e.candidate))}};
    if (e.parent) {
        value["parent_ref"] = e.parent_ref;
        value["parent"] = json::parse(common_flydelta_progress_assessment_to_json(*e.parent));
    }
    return value.dump();
}

bool common_flydelta_progress_evidence_from_json(const std::string & text,
        common_flydelta_progress_evidence & e, std::string & error) {
    e = {}; error.clear();
    if (text.size() > 4U * 1024U * 1024U) { error = "FlyDelta progress evidence exceeds its byte bound"; return false; }
    try {
        const auto value = nlohmann::ordered_json::parse(text);
        e.schema_version = value.at("schema_version").get<int>();
        e.baseline_ref = value.at("baseline_ref").get<std::string>();
        e.candidate_ref = value.at("candidate_ref").get<std::string>();
        e.baseline_evidence_ref = value.at("baseline_evidence_ref").get<std::string>();
        e.candidate_evidence_ref = value.at("candidate_evidence_ref").get<std::string>();
        if (!common_flydelta_progress_assessment_from_json(value.at("baseline").dump(), e.baseline, error) ||
                !common_flydelta_progress_assessment_from_json(value.at("candidate").dump(), e.candidate, error)) return false;
        if (value.contains("parent")) {
            e.parent.emplace();
            e.parent_ref = value.at("parent_ref").get<std::string>();
            if (!common_flydelta_progress_assessment_from_json(value.at("parent").dump(), *e.parent, error)) return false;
        } else if (value.contains("parent_ref")) { error = "FlyDelta progress parent lacks its assessment"; return false; }
    } catch (const std::exception & exception) {
        error = std::string("FlyDelta progress evidence is malformed: ") + exception.what();
        e = {}; return false;
    }
    return common_flydelta_progress_evidence_validate(e, error);
}
