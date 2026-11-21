#pragma once

#include <string>
#include <optional>
#include <vector>

// A verifier-owned description of goals reached in one evaluation. It is
// search evidence only and never implies promotion or activation authority.
struct common_flydelta_progress_assessment {
    int schema_version = 1;
    std::string objective_id;
    std::string objective_revision;
    std::string verifier_ref;
    bool assessment_known = false;
    bool regression_known = false;
    bool has_regression = false;
    bool target_completed = false;
    std::vector<std::string> satisfied_goals;
    std::vector<std::string> remaining_goals;
};

// A replacement must add a verified goal while retaining all previously
// satisfied goals, and both assessments must have the same provenance.
bool common_flydelta_progress_strictly_dominates(
        const common_flydelta_progress_assessment & incumbent,
        const common_flydelta_progress_assessment & candidate);

bool common_flydelta_progress_assessment_validate(
        const common_flydelta_progress_assessment & assessment, std::string & error);
bool common_flydelta_progress_comparable(
        const common_flydelta_progress_assessment & left,
        const common_flydelta_progress_assessment & right);
std::string common_flydelta_progress_assessment_to_json(
        const common_flydelta_progress_assessment & assessment);
bool common_flydelta_progress_assessment_from_json(
        const std::string & text, common_flydelta_progress_assessment & assessment, std::string & error);

// Paired host observations, never a model-supplied progress label. Optional
// parent evidence describes the incumbent independently of the original baseline.
struct common_flydelta_progress_evidence {
    int schema_version = 1;
    std::string baseline_ref;
    std::string candidate_ref;
    std::string baseline_evidence_ref;
    std::string candidate_evidence_ref;
    common_flydelta_progress_assessment baseline;
    common_flydelta_progress_assessment candidate;
    std::string parent_ref;
    std::optional<common_flydelta_progress_assessment> parent;
};

enum class common_flydelta_progress_transition { unknown, unchanged, improved, solved, regressed };
const char * common_flydelta_progress_transition_name(common_flydelta_progress_transition transition);
bool common_flydelta_progress_evidence_validate(const common_flydelta_progress_evidence & evidence,
        std::string & error);
common_flydelta_progress_transition common_flydelta_progress_evidence_classify(
        const common_flydelta_progress_evidence & evidence);
std::string common_flydelta_progress_evidence_to_json(const common_flydelta_progress_evidence & evidence);
bool common_flydelta_progress_evidence_from_json(const std::string & text,
        common_flydelta_progress_evidence & evidence, std::string & error);
