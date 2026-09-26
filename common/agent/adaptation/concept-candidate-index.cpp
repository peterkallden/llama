#include "agent/adaptation/concept-candidate-index.h"

#include "hash/hash.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>

namespace {

using json = nlohmann::ordered_json;

bool bounded(const std::string & value, size_t max_size = 512) {
    return !value.empty() && value.size() <= max_size;
}

template <typename T>
void append_unique(std::vector<T> & values, const T & value) {
    if (value.empty()) return;
    if (std::find(values.begin(), values.end(), value) == values.end()) values.push_back(value);
}

std::string scope_key(const common_agent_scope & scope) {
    return scope.namespace_id + "\n" + scope.project_id + "\n" + scope.session_id;
}

std::string candidate_id_for(const common_agent_concept_hypothesis & hypothesis) {
    const std::string key = hypothesis.concept_key + "\n" + scope_key(hypothesis.scope);
    return "concept://candidate/" + hash_sha256_hex(key.data(), key.size()).substr(0, 32);
}

const char * source_strength(common_agent_concept_source_kind source) {
    switch (source) {
        case common_agent_concept_source_kind::user_taught_concept:
        case common_agent_concept_source_kind::user_correction:
            return "explicit_user";
        case common_agent_concept_source_kind::conversation: return "inferred_dialogue";
        case common_agent_concept_source_kind::research: return "research";
        case common_agent_concept_source_kind::reflection: return "reflection";
    }
    return "inferred_dialogue";
}

int source_strength_rank(const std::string & value) {
    if (value == "explicit_user") return 4;
    if (value == "inferred_dialogue") return 3;
    if (value == "research") return 2;
    if (value == "reflection") return 1;
    return 0;
}

bool same_scope(const common_agent_scope & left, const common_agent_scope & right) {
    return left.namespace_id == right.namespace_id &&
        left.project_id == right.project_id && left.session_id == right.session_id;
}

} // namespace

bool common_agent_concept_candidate_validate(
        const common_agent_concept_candidate & candidate,
        std::string & error) {
    error.clear();
    if (candidate.schema_version != 1 || !bounded(candidate.candidate_id) ||
            !bounded(candidate.concept_key) || !bounded(candidate.canonical_statement_ref) ||
            candidate.scope.namespace_id.empty() || candidate.scope.session_id.empty() ||
            candidate.source_refs.size() > 64 || candidate.supporting_evidence_refs.size() > 128 ||
            candidate.disconfirming_evidence_refs.size() > 128 || candidate.relation_refs.size() > 128 ||
            candidate.independent_support_keys.size() > 128 || !bounded(candidate.grounding_state) ||
            !bounded(candidate.novelty_state) || !bounded(candidate.source_strength) ||
            candidate.contrast_strength.size() > 128 || candidate.independence.size() > 128) {
        error = "concept candidate index entry is incomplete or out of bounds";
        return false;
    }
    return true;
}

std::string common_agent_concept_candidate_to_json(
        const common_agent_concept_candidate & candidate) {
    return json{
        {"kind", "agent_concept_candidate"},
        {"schema_version", candidate.schema_version},
        {"revision", candidate.revision},
        {"candidate_id", candidate.candidate_id},
        {"concept_key", candidate.concept_key},
        {"canonical_statement_ref", candidate.canonical_statement_ref},
        {"scope", {
            {"namespace_id", candidate.scope.namespace_id},
            {"project_id", candidate.scope.project_id},
            {"session_id", candidate.scope.session_id},
        }},
        {"source_refs", candidate.source_refs},
        {"supporting_evidence_refs", candidate.supporting_evidence_refs},
        {"disconfirming_evidence_refs", candidate.disconfirming_evidence_refs},
        {"relation_refs", candidate.relation_refs},
        {"independent_support_keys", candidate.independent_support_keys},
        {"grounding_state", candidate.grounding_state},
        {"novelty_state", candidate.novelty_state},
        {"source_strength", candidate.source_strength},
        {"contrast_strength", candidate.contrast_strength},
        {"independence", candidate.independence},
        {"synthesis_eligible", candidate.synthesis_eligible},
    }.dump();
}

bool common_agent_concept_candidate_from_json(
        const std::string & text,
        common_agent_concept_candidate & candidate,
        std::string & error) {
    error.clear();
    try {
        const auto value = json::parse(text);
        if (value.value("kind", "") != "agent_concept_candidate") {
            error = "concept candidate payload kind is invalid";
            return false;
        }
        candidate = {};
        candidate.schema_version = value.value("schema_version", 0);
        candidate.revision = value.value("revision", size_t{0});
        candidate.candidate_id = value.value("candidate_id", "");
        candidate.concept_key = value.value("concept_key", "");
        candidate.canonical_statement_ref = value.value("canonical_statement_ref", "");
        const auto scope = value.value("scope", json::object());
        candidate.scope.namespace_id = scope.value("namespace_id", "");
        candidate.scope.project_id = scope.value("project_id", "");
        candidate.scope.session_id = scope.value("session_id", "");
        candidate.source_refs = value.value("source_refs", std::vector<std::string>{});
        candidate.supporting_evidence_refs = value.value("supporting_evidence_refs", std::vector<std::string>{});
        candidate.disconfirming_evidence_refs = value.value("disconfirming_evidence_refs", std::vector<std::string>{});
        candidate.relation_refs = value.value("relation_refs", std::vector<std::string>{});
        candidate.independent_support_keys = value.value("independent_support_keys", std::vector<std::string>{});
        candidate.grounding_state = value.value("grounding_state", "");
        candidate.novelty_state = value.value("novelty_state", "");
        candidate.source_strength = value.value("source_strength", "");
        candidate.contrast_strength = value.value("contrast_strength", "");
        candidate.independence = value.value("independence", "");
        candidate.synthesis_eligible = value.value("synthesis_eligible", false);
        return common_agent_concept_candidate_validate(candidate, error);
    } catch (const std::exception & exception) {
        error = std::string("concept candidate payload is malformed: ") + exception.what();
        return false;
    }
}

common_agent_concept_candidate_index::common_agent_concept_candidate_index(
        common_learning_lifecycle_store * lifecycle_store)
    : lifecycle_store_(lifecycle_store) {}

bool common_agent_concept_candidate_index::load(std::string & error) {
    error.clear();
    candidates_.clear();
    if (!lifecycle_store_) return true;
    const auto records = lifecycle_store_->list(error);
    if (!error.empty()) return false;
    for (const auto & record : records) {
        if (record.kind != common_learning_lifecycle_kind::candidate) continue;
        common_agent_concept_candidate candidate;
        std::string parse_error;
        if (!common_agent_concept_candidate_from_json(record.payload_json, candidate, parse_error)) continue;
        const auto it = std::find_if(candidates_.begin(), candidates_.end(), [&](const auto & value) {
            return value.candidate_id == candidate.candidate_id;
        });
        if (it == candidates_.end()) candidates_.push_back(std::move(candidate));
        else if (candidate.revision > it->revision) *it = std::move(candidate);
    }
    return true;
}

common_agent_concept_candidate & common_agent_concept_candidate_index::find_or_create(
        const common_agent_concept_hypothesis & hypothesis) {
    const std::string candidate_id = candidate_id_for(hypothesis);
    const auto it = std::find_if(candidates_.begin(), candidates_.end(), [&](const auto & value) {
        return value.candidate_id == candidate_id;
    });
    if (it != candidates_.end()) return *it;
    common_agent_concept_candidate candidate;
    candidate.candidate_id = candidate_id;
    candidate.concept_key = hypothesis.concept_key;
    candidate.canonical_statement_ref = hypothesis.canonical_statement_ref.empty()
        ? "statement://sha256/" + hash_sha256_hex(
            hypothesis.statement.data(), hypothesis.statement.size())
        : hypothesis.canonical_statement_ref;
    candidate.scope = hypothesis.scope;
    candidate.source_strength = source_strength(hypothesis.source_kind);
    candidate.grounding_state = hypothesis.status == common_agent_concept_hypothesis_status::grounded
        ? "grounded" : "proposed";
    candidates_.push_back(std::move(candidate));
    return candidates_.back();
}

bool common_agent_concept_candidate_index::persist(
        common_agent_concept_candidate & candidate,
        const common_learning_transaction & transaction,
        std::string & error) {
    error.clear();
    ++candidate.revision;
    if (!lifecycle_store_) return true;
    common_learning_lifecycle_record record;
    record.event_id = candidate.candidate_id + "/" + std::to_string(candidate.revision);
    record.subject_id = candidate.candidate_id;
    record.kind = common_learning_lifecycle_kind::candidate;
    record.status = candidate.synthesis_eligible
        ? common_learning_lifecycle_status::eligible
        : common_learning_lifecycle_status::observed;
    record.idempotency_key = record.event_id;
    record.source_id = transaction.id.empty() ? candidate.candidate_id : transaction.id;
    record.namespace_id = candidate.scope.namespace_id;
    record.project_id = candidate.scope.project_id;
    record.session_id = candidate.scope.session_id;
    const std::string payload = common_agent_concept_candidate_to_json(candidate);
    record.content_hash = hash_sha256_hex(payload.data(), payload.size());
    record.created_at = transaction.created_at.empty() ? "concept-index" : transaction.created_at;
    record.payload_json = payload;
    return lifecycle_store_->append(record, error);
}

bool common_agent_concept_candidate_index::observe_hypothesis(
        const common_agent_concept_hypothesis & hypothesis,
        const common_learning_transaction & transaction,
        std::string & error) {
    error.clear();
    if (!common_agent_concept_hypothesis_validate(hypothesis, error)) return false;
    const std::string candidate_id = candidate_id_for(hypothesis);
    const bool existed = std::find_if(candidates_.begin(), candidates_.end(), [&](const auto & value) {
        return value.candidate_id == candidate_id;
    }) != candidates_.end();
    auto & candidate = find_or_create(hypothesis);
    if (candidate.concept_key != hypothesis.concept_key || !same_scope(candidate.scope, hypothesis.scope)) {
        error = "concept candidate index identity collision";
        return false;
    }
    for (const auto & ref : hypothesis.source_refs) append_unique(candidate.source_refs, ref);
    for (const auto & ref : hypothesis.counterexamples) append_unique(candidate.disconfirming_evidence_refs, ref);
    const std::string incoming_strength = source_strength(hypothesis.source_kind);
    if (source_strength_rank(incoming_strength) > source_strength_rank(candidate.source_strength)) {
        candidate.source_strength = incoming_strength;
    }
    if (existed && candidate.novelty_state != "conflict") candidate.novelty_state = "same";
    candidate.grounding_state = hypothesis.status == common_agent_concept_hypothesis_status::grounded
        ? "grounded" : candidate.grounding_state;
    return persist(candidate, transaction, error);
}

bool common_agent_concept_candidate_index::observe_relation(
        const common_agent_concept_hypothesis & hypothesis,
        const common_flydelta_teaching_relation & relation,
        const common_learning_transaction & transaction,
        std::string & error) {
    error.clear();
    if (!common_agent_concept_hypothesis_validate(hypothesis, error) ||
            !common_flydelta_teaching_relation_validate(relation, error) ||
            relation.teaching_key != hypothesis.concept_key ||
            relation.scope.namespace_id != hypothesis.scope.namespace_id ||
            relation.scope.project_id != hypothesis.scope.project_id ||
            relation.scope.session_id != hypothesis.scope.session_id) {
        if (error.empty()) error = "concept candidate relation is incompatible with its hypothesis";
        return false;
    }
    auto & candidate = find_or_create(hypothesis);
    append_unique(candidate.relation_refs, relation.id);
    append_unique(candidate.supporting_evidence_refs, relation.evidence_ref);
    append_unique(candidate.independent_support_keys, relation.task_fingerprint);
    candidate.grounding_state = "resolved";
    candidate.contrast_strength = relation.baseline_origin == common_flydelta_teaching_origin::observed
        ? "observed_pair" : "host_verified";
    candidate.independence = candidate.independent_support_keys.size() > 1
        ? "separate_task" : "same_task";
    candidate.synthesis_eligible = candidate.independent_support_keys.size() >= 2;
    return persist(candidate, transaction, error);
}

bool common_agent_concept_candidate_index::observe_disconfirmation(
        const common_agent_concept_hypothesis & hypothesis,
        const std::string & evidence_ref,
        const common_learning_transaction & transaction,
        std::string & error) {
    error.clear();
    if (!common_agent_concept_hypothesis_validate(hypothesis, error) || !bounded(evidence_ref)) {
        if (error.empty()) error = "concept candidate disconfirmation is invalid";
        return false;
    }
    auto & candidate = find_or_create(hypothesis);
    append_unique(candidate.disconfirming_evidence_refs, evidence_ref);
    candidate.novelty_state = "conflict";
    candidate.synthesis_eligible = false;
    return persist(candidate, transaction, error);
}

bool common_agent_concept_candidate_index::resolve(
        const std::string & concept_key,
        const common_agent_scope & scope,
        common_agent_concept_candidate & candidate,
        std::string & error) const {
    error.clear();
    candidate = {};
    const auto it = std::find_if(candidates_.begin(), candidates_.end(), [&](const auto & value) {
        return value.concept_key == concept_key && same_scope(value.scope, scope);
    });
    if (it == candidates_.end()) return true;
    candidate = *it;
    return true;
}
