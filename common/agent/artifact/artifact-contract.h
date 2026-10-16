#pragma once

#include "plan/plan-types.h"

#include <string>
#include <vector>

// Report profiles are curated presentation/coverage contracts. They do not
// grant tool authority and never replace the source workspace as authority.
using common_agent_artifact_section_profile = common_plan_artifact_section_profile;
using common_agent_artifact_profile = common_plan_artifact_profile;

enum class common_agent_artifact_block_kind { observation, conclusion, recommendation };
enum class common_agent_artifact_validation { unvalidated, references_resolve, gap, contradicted };

struct common_agent_artifact_block {
    std::string id;
    common_agent_artifact_block_kind kind = common_agent_artifact_block_kind::observation;
    std::string text;
    std::vector<std::string> evidence_ids;
    std::vector<std::string> source_refs;
    common_agent_artifact_validation validation = common_agent_artifact_validation::unvalidated;
    std::string validation_note;
    std::string origin = "model";
};

struct common_agent_artifact_section {
    std::string id;
    std::string title;
    std::vector<common_agent_artifact_block> blocks;
};

struct common_agent_artifact_snapshot {
    std::string id;
    std::string source_plan_id;
    uint64_t plan_version = 0;
    std::string profile_id;
    std::string profile_revision;
    std::string fingerprint;
    std::vector<common_plan_observation> observations;
    std::vector<common_plan_assumption> assumptions;
    bool coverage_complete = false;
    std::vector<std::string> unresolved_gaps;
};

struct common_agent_artifact_document {
    std::string id;
    std::string title;
    std::string purpose;
    std::string profile_id;
    std::string profile_revision;
    uint64_t revision = 1;
    std::string previous_revision_uri;
    std::string snapshot_uri;
    std::string source_plan_id;
    uint64_t source_plan_version = 0;
    bool coverage_complete = false;
    std::vector<std::string> unresolved_gaps;
    std::vector<common_agent_artifact_section> sections;
};

bool common_agent_artifact_profile_valid(const common_agent_artifact_profile &, std::string & error);
bool common_agent_artifact_profile_from_json(const std::string &, common_agent_artifact_profile &, std::string & error);
bool common_agent_artifact_snapshot_from_plan(
    const common_plan_state &, const common_agent_artifact_profile &,
    common_agent_artifact_snapshot &, std::string & error);
bool common_agent_artifact_snapshot_from_json(
    const std::string &, common_agent_artifact_snapshot &, std::string & error);
bool common_agent_artifact_validate_block(
    const common_agent_artifact_block &, const common_agent_artifact_snapshot &,
    common_agent_artifact_block & validated, std::string & error);
bool common_agent_artifact_parse_section_dsl(
    const std::string & text, const common_agent_artifact_section_profile & profile,
    const common_agent_artifact_snapshot & snapshot, common_agent_artifact_section & out,
    std::string & error);
bool common_agent_artifact_document_to_json(const common_agent_artifact_document &, std::string & out, std::string & error);
bool common_agent_artifact_document_from_json(const std::string &, common_agent_artifact_document &, std::string & error);
bool common_agent_artifact_snapshot_to_json(const common_agent_artifact_snapshot &, std::string & out, std::string & error);
bool common_agent_artifact_completion_summary(const common_plan_state &, std::string & out);
bool common_agent_artifact_document_from_plan(
    const common_plan_state &, const common_agent_artifact_profile &,
    const std::string & snapshot_uri, common_agent_artifact_document &, std::string & error);
bool common_agent_artifact_render_markdown(const common_agent_artifact_document &, std::string & out, std::string & error);
