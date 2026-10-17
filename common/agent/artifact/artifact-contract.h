#pragma once

#include "plan/plan-types.h"
#include "agent/thinking/research/research-contract.h"

#include <string>
#include <vector>

// Report profiles are curated presentation/coverage contracts. They do not
// grant tool authority and never replace the source workspace as authority.
using common_agent_artifact_section_profile = common_plan_artifact_section_profile;
using common_agent_artifact_profile = common_plan_artifact_profile;

enum class common_agent_artifact_block_kind { observation, conclusion, recommendation };
enum class common_agent_artifact_validation { unvalidated, references_resolve, gap, contradicted };
enum class common_agent_artifact_support { unverified, linked_support, contradicted, mixed };
enum class common_agent_artifact_author { model, user, host };
enum class common_agent_artifact_review_state { reviewed, unreviewed, requires_research };
enum class common_agent_artifact_document_kind { report, document };

struct common_agent_artifact_block {
    std::string id;
    common_agent_artifact_block_kind kind = common_agent_artifact_block_kind::observation;
    std::string text;
    std::vector<std::string> evidence_ids;
    std::vector<std::string> source_refs;
    common_agent_artifact_validation validation = common_agent_artifact_validation::unvalidated;
    common_agent_artifact_support support = common_agent_artifact_support::unverified;
    std::string validation_note;
    std::string origin = "model";
    common_agent_artifact_author author = common_agent_artifact_author::model;
    common_agent_artifact_review_state review_state = common_agent_artifact_review_state::reviewed;
};

struct common_agent_artifact_section {
    std::string id;
    std::string title;
    bool no_findings = false;
    std::vector<common_agent_artifact_block> blocks;
    std::vector<std::string> out_of_scope_refs;
};

struct common_agent_artifact_source_citation {
    std::string source_id, title, origin, resource_uri, retrieved_at, content_hash;
    std::vector<std::string> evidence_ids, locations;
    std::string original_uri, resolved_uri, file_name, source_provider, source_operation, request_summary;
};

struct common_agent_artifact_section_context {
    std::string rendered_json;
    std::vector<std::string> relevant_refs, included_refs, omitted_refs, out_of_scope_refs;
    std::string omitted_reason;
};

struct common_agent_artifact_snapshot {
    int schema_version = 4;
    std::string id;
    std::string source_plan_id;
    uint64_t plan_version = 0;
    std::string profile_id;
    std::string profile_revision;
    std::string research_workspace_id;
    std::string research_checkpoint_id;
    std::string research_objective;
    common_agent_research_coverage research_coverage;
    std::vector<common_agent_research_source> research_sources;
    std::vector<common_agent_research_evidence> research_evidence;
    std::vector<common_agent_research_source_comparison> research_comparisons;
    std::vector<common_agent_research_gap> research_gaps;
    std::vector<std::string> omitted_source_refs;
    std::string fingerprint;
    std::vector<common_plan_observation> observations;
    std::vector<common_plan_assumption> assumptions;
    bool coverage_complete = false;
    std::vector<std::string> unresolved_gaps;
};

struct common_agent_artifact_document {
    uint32_t schema_version = 3;
    common_agent_artifact_document_kind kind = common_agent_artifact_document_kind::report;
    // Stored documents retain their renderer semantics for manifest readback.
    uint32_t markdown_renderer_version = 3;
    std::string id;
    std::string title;
    std::string purpose;
    std::string profile_id;
    std::string profile_revision;
    uint64_t revision = 1;
    std::string previous_revision_uri;
    std::string snapshot_fingerprint;
    std::string snapshot_uri;
    std::string content_fingerprint;
    std::string editor_parent_uri;
    std::string editor_operation_id;
    std::string editor_markdown_uri;
    std::string source_resource_uri;
    std::string source_content_fingerprint;
    std::string importer_version;
    std::string source_plan_id;
    uint64_t source_plan_version = 0;
    bool coverage_complete = false;
    std::vector<std::string> unresolved_gaps;
    std::vector<common_agent_artifact_source_citation> source_bibliography;
    std::vector<common_agent_artifact_section> sections;
};

bool common_agent_artifact_profile_valid(const common_agent_artifact_profile &, std::string & error);
bool common_agent_artifact_profile_from_json(const std::string &, common_agent_artifact_profile &, std::string & error);
bool common_agent_artifact_snapshot_from_plan(
    const common_plan_state &, const common_agent_artifact_profile &,
    common_agent_artifact_snapshot &, std::string & error,
    const common_agent_research_workspace * research = nullptr,
    const std::string & research_checkpoint_id = {});
bool common_agent_artifact_snapshot_from_json(
    const std::string &, common_agent_artifact_snapshot &, std::string & error);
bool common_agent_artifact_validate_block(
    const common_agent_artifact_block &, const common_agent_artifact_snapshot &,
    common_agent_artifact_block & validated, std::string & error,
    const std::vector<std::string> & included_refs = {}, bool restrict_to_included_refs = false);
bool common_agent_artifact_parse_section_dsl(
    const std::string & text, const common_agent_artifact_section_profile & profile,
    const common_agent_artifact_snapshot & snapshot, common_agent_artifact_section & out,
    std::string & error, const std::vector<std::string> & included_refs = {},
    bool restrict_to_included_refs = false);
bool common_agent_artifact_parse_section_json(
    const std::string & text, const common_agent_artifact_section_profile & profile,
    const common_agent_artifact_snapshot & snapshot, common_agent_artifact_section & out,
    std::string & error, const std::vector<std::string> & included_refs = {},
    bool restrict_to_included_refs = false);
// Restore host-persisted typed output. Unlike model parsers this accepts only
// canonical IDs and never treats stored content as a fresh model response.
bool common_agent_artifact_section_from_stored_json(
    const std::string &, const common_agent_artifact_section_profile &,
    const common_agent_artifact_snapshot &, const std::vector<std::string> & included_refs,
    common_agent_artifact_section &, std::string & error);
bool common_agent_artifact_document_to_json(const common_agent_artifact_document &, std::string & out, std::string & error);
bool common_agent_artifact_document_from_json(const std::string &, common_agent_artifact_document &, std::string & error);
bool common_agent_artifact_snapshot_to_json(const common_agent_artifact_snapshot &, std::string & out, std::string & error);
bool common_agent_artifact_render_section_context(
    const common_agent_artifact_snapshot &, const std::vector<std::string> & required_sources,
    size_t char_budget, std::string & context_json,
    std::vector<std::string> & included_refs, std::vector<std::string> & omitted_refs,
    std::string & error, const std::vector<std::string> & evidence_relations = {});
bool common_agent_artifact_prepare_section_context(
    const common_agent_artifact_snapshot &, const common_agent_artifact_section_profile &,
    size_t char_budget, common_agent_artifact_section_context &, std::string & error);
std::string common_agent_artifact_section_dsl_grammar(
    const common_agent_artifact_section_profile &, const std::string & rendered_context_json);
bool common_agent_artifact_completion_summary(const common_plan_state &, std::string & out);
bool common_agent_artifact_document_from_plan(
    const common_plan_state &, const common_agent_artifact_profile &,
    const std::string & snapshot_uri, common_agent_artifact_document &, std::string & error,
    const std::string & report_id = {});
bool common_agent_artifact_render_markdown(const common_agent_artifact_document &, std::string & out, std::string & error);
bool common_agent_artifact_render_editable_markdown(const common_agent_artifact_document &, std::string & out, std::string & error);
// Project an already authorized immutable document in source order. An empty
// selection means all blocks; neighbors retain their own selection status.
bool common_agent_artifact_render_document_context(
    const common_agent_artifact_document &, const std::string & document_uri,
    const std::vector<std::string> & block_ids, std::string & out, std::string & error);
// DOCX conversion has no published media attachments. Retain image alt text
// instead of exporting links into the processor's temporary filesystem.
bool common_agent_artifact_prepare_docx_markdown(
    const std::string &, std::string & out, bool & omitted_images, std::string & error);
bool common_agent_artifact_apply_markdown_edit(
    const common_agent_artifact_document &, const std::string & markdown,
    common_agent_artifact_document &, std::string & error);
