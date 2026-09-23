#include "agent/adaptation/semantic-teaching.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <map>

namespace {

using json = nlohmann::ordered_json;

constexpr size_t k_max_text = 1024;
constexpr size_t k_max_items = 32;

bool bounded(const std::string & value, size_t max_size = k_max_text) {
    return !value.empty() && value.size() <= max_size;
}

bool bounded_items(const std::vector<std::string> & values) {
    return values.size() <= k_max_items && std::all_of(values.begin(), values.end(),
        [](const std::string & value) { return bounded(value); });
}

json scope_to_json(const common_agent_scope & scope) {
    return {
        {"namespace_id", scope.namespace_id},
        {"session_id", scope.session_id},
        {"project_id", scope.project_id},
        {"turn_id", scope.turn_id},
        {"memory_global_opt_in", scope.memory_global_opt_in},
    };
}

void scope_from_json(const json & value, common_agent_scope & scope) {
    if (!value.is_object()) return;
    scope.namespace_id = value.value("namespace_id", scope.namespace_id);
    scope.session_id = value.value("session_id", scope.session_id);
    scope.project_id = value.value("project_id", scope.project_id);
    scope.turn_id = value.value("turn_id", scope.turn_id);
    scope.memory_global_opt_in = value.value("memory_global_opt_in", false);
}

bool parse_kind(const std::string & value, common_agent_concept_semantic_kind & kind) {
    if (value == "fact") kind = common_agent_concept_semantic_kind::fact;
    else if (value == "procedure") kind = common_agent_concept_semantic_kind::procedure;
    else if (value == "decision_rule") kind = common_agent_concept_semantic_kind::decision_rule;
    else if (value == "heuristic") kind = common_agent_concept_semantic_kind::heuristic;
    else if (value == "concept") kind = common_agent_concept_semantic_kind::concept;
    else return false;
    return true;
}

bool parse_source(const std::string & value, common_agent_concept_source_kind & source) {
    if (value == "conversation") source = common_agent_concept_source_kind::conversation;
    else if (value == "research") source = common_agent_concept_source_kind::research;
    else if (value == "user_correction") source = common_agent_concept_source_kind::user_correction;
    else if (value == "user_taught_concept") source = common_agent_concept_source_kind::user_taught_concept;
    else if (value == "reflection") source = common_agent_concept_source_kind::reflection;
    else return false;
    return true;
}

bool parse_status(const std::string & value, common_agent_concept_hypothesis_status & status) {
    if (value == "proposed") status = common_agent_concept_hypothesis_status::proposed;
    else if (value == "grounded") status = common_agent_concept_hypothesis_status::grounded;
    else if (value == "rejected") status = common_agent_concept_hypothesis_status::rejected;
    else if (value == "insufficient_evidence") status = common_agent_concept_hypothesis_status::insufficient_evidence;
    else return false;
    return true;
}

template<typename T>
bool string_array(const json & value, const char * key, std::vector<T> & output) {
    if (!value.contains(key) || !value.at(key).is_array()) return false;
    output.clear();
    for (const auto & item : value.at(key)) {
        if (!item.is_string()) return false;
        output.push_back(item.get<T>());
    }
    return true;
}

common_adaptation_evidence_source source_for(
        common_agent_concept_source_kind source) {
    switch (source) {
        case common_agent_concept_source_kind::research:
            return common_adaptation_evidence_source::research_alternative;
        case common_agent_concept_source_kind::reflection:
            return common_adaptation_evidence_source::reflection_alternative;
        case common_agent_concept_source_kind::user_correction:
            return common_adaptation_evidence_source::user_correction;
        case common_agent_concept_source_kind::user_taught_concept:
            return common_adaptation_evidence_source::user_taught_concept;
        case common_agent_concept_source_kind::conversation:
            return common_adaptation_evidence_source::user_taught_concept;
    }
    return common_adaptation_evidence_source::user_taught_concept;
}

} // namespace

const char * common_agent_concept_semantic_kind_name(
        common_agent_concept_semantic_kind kind) {
    switch (kind) {
        case common_agent_concept_semantic_kind::fact: return "fact";
        case common_agent_concept_semantic_kind::procedure: return "procedure";
        case common_agent_concept_semantic_kind::decision_rule: return "decision_rule";
        case common_agent_concept_semantic_kind::heuristic: return "heuristic";
        case common_agent_concept_semantic_kind::concept: return "concept";
    }
    return "unknown";
}

const char * common_agent_concept_source_kind_name(
        common_agent_concept_source_kind kind) {
    switch (kind) {
        case common_agent_concept_source_kind::conversation: return "conversation";
        case common_agent_concept_source_kind::research: return "research";
        case common_agent_concept_source_kind::user_correction: return "user_correction";
        case common_agent_concept_source_kind::user_taught_concept: return "user_taught_concept";
        case common_agent_concept_source_kind::reflection: return "reflection";
    }
    return "unknown";
}

const char * common_agent_concept_hypothesis_status_name(
        common_agent_concept_hypothesis_status status) {
    switch (status) {
        case common_agent_concept_hypothesis_status::proposed: return "proposed";
        case common_agent_concept_hypothesis_status::grounded: return "grounded";
        case common_agent_concept_hypothesis_status::rejected: return "rejected";
        case common_agent_concept_hypothesis_status::insufficient_evidence: return "insufficient_evidence";
    }
    return "unknown";
}

bool common_agent_concept_hypothesis_validate(
        const common_agent_concept_hypothesis & hypothesis,
        std::string & error) {
    error.clear();
    if (hypothesis.schema_version != 1 || !bounded(hypothesis.id) ||
            !bounded(hypothesis.concept_key) || !bounded(hypothesis.statement) ||
            hypothesis.scope.namespace_id.empty() || hypothesis.scope.session_id.empty() ||
            !bounded_items(hypothesis.preconditions) ||
            !bounded_items(hypothesis.invariants) ||
            !bounded_items(hypothesis.counterexamples) ||
            !bounded_items(hypothesis.source_refs) || hypothesis.source_refs.empty() ||
            hypothesis.transitions.size() > k_max_items ||
            !std::isfinite(hypothesis.confidence) || hypothesis.confidence < 0.0f ||
            hypothesis.confidence > 1.0f) {
        error = "semantic concept hypothesis is incomplete or out of bounds";
        return false;
    }
    for (const auto & transition : hypothesis.transitions) {
        if (!bounded(transition.state_before) || !bounded(transition.condition) ||
                !bounded(transition.required_transition) || !bounded(transition.state_after)) {
            error = "semantic concept transition is incomplete";
            return false;
        }
    }
    if (hypothesis.status == common_agent_concept_hypothesis_status::grounded &&
            (!hypothesis.host_grounded || !hypothesis.reusable)) {
        error = "grounded semantic concept hypothesis is not host approved and reusable";
        return false;
    }
    return true;
}

std::string common_agent_concept_hypothesis_to_json(
        const common_agent_concept_hypothesis & hypothesis) {
    json transitions = json::array();
    for (const auto & transition : hypothesis.transitions) {
        transitions.push_back({
            {"state_before", transition.state_before},
            {"condition", transition.condition},
            {"required_transition", transition.required_transition},
            {"state_after", transition.state_after},
        });
    }
    return json{
        {"kind", "agent_concept_hypothesis"},
        {"schema_version", hypothesis.schema_version},
        {"id", hypothesis.id},
        {"concept_key", hypothesis.concept_key},
        {"statement", hypothesis.statement},
        {"semantic_kind", common_agent_concept_semantic_kind_name(hypothesis.semantic_kind)},
        {"source_kind", common_agent_concept_source_kind_name(hypothesis.source_kind)},
        {"status", common_agent_concept_hypothesis_status_name(hypothesis.status)},
        {"scope", scope_to_json(hypothesis.scope)},
        {"preconditions", hypothesis.preconditions},
        {"transitions", std::move(transitions)},
        {"invariants", hypothesis.invariants},
        {"counterexamples", hypothesis.counterexamples},
        {"source_refs", hypothesis.source_refs},
        {"confidence", hypothesis.confidence},
        {"host_grounded", hypothesis.host_grounded},
        {"reusable", hypothesis.reusable},
    }.dump();
}

bool common_agent_concept_hypothesis_from_json(
        const std::string & text,
        common_agent_concept_hypothesis & hypothesis,
        std::string & error) {
    error.clear();
    try {
        const auto value = json::parse(text);
        if (value.value("kind", "") != "agent_concept_hypothesis") {
            error = "semantic concept hypothesis kind is invalid";
            return false;
        }
        hypothesis = {};
        hypothesis.schema_version = value.value("schema_version", 0);
        hypothesis.id = value.value("id", "");
        hypothesis.concept_key = value.value("concept_key", "");
        hypothesis.statement = value.value("statement", "");
        if (!parse_kind(value.value("semantic_kind", ""), hypothesis.semantic_kind) ||
                !parse_source(value.value("source_kind", ""), hypothesis.source_kind) ||
                !parse_status(value.value("status", ""), hypothesis.status) ||
                !string_array(value, "preconditions", hypothesis.preconditions) ||
                !string_array(value, "invariants", hypothesis.invariants) ||
                !string_array(value, "counterexamples", hypothesis.counterexamples) ||
                !string_array(value, "source_refs", hypothesis.source_refs)) {
            error = "semantic concept hypothesis contains invalid enum or list data";
            return false;
        }
        scope_from_json(value.value("scope", json::object()), hypothesis.scope);
        if (!value.contains("transitions") || !value.at("transitions").is_array()) {
            error = "semantic concept hypothesis transitions are missing";
            return false;
        }
        for (const auto & item : value.at("transitions")) {
            common_agent_concept_transition transition;
            transition.state_before = item.value("state_before", "");
            transition.condition = item.value("condition", "");
            transition.required_transition = item.value("required_transition", "");
            transition.state_after = item.value("state_after", "");
            hypothesis.transitions.push_back(std::move(transition));
        }
        hypothesis.confidence = value.value("confidence", 0.0f);
        hypothesis.host_grounded = value.value("host_grounded", false);
        hypothesis.reusable = value.value("reusable", false);
        return common_agent_concept_hypothesis_validate(hypothesis, error);
    } catch (const std::exception & exception) {
        error = std::string("semantic concept hypothesis JSON is malformed: ") + exception.what();
        return false;
    }
}

bool common_agent_concept_grounding_validate(
        const common_agent_concept_grounding & grounding,
        std::string & error) {
    error.clear();
    if (grounding.schema_version != 1 || !bounded(grounding.id) ||
            !bounded(grounding.hypothesis_ref) || !bounded(grounding.behavior_key) ||
            !bounded(grounding.verifier_ref) || !bounded(grounding.changed_dimension) ||
            grounding.scope.namespace_id.empty() || grounding.scope.session_id.empty() ||
            !bounded_items(grounding.invariant_dimensions) || grounding.minimum_contrasts < 2 ||
            grounding.minimum_contrasts > k_max_items || !grounding.host_approved ||
            !grounding.reusable) {
        error = "semantic concept grounding is incomplete or not host approved";
        return false;
    }
    for (const auto & invariant : grounding.invariant_dimensions) {
        if (invariant == grounding.changed_dimension) {
            error = "semantic concept grounding changes an invariant dimension";
            return false;
        }
    }
    return true;
}

const char * common_agent_concept_fixture_role_name(
        common_agent_concept_fixture_role role) {
    switch (role) {
        case common_agent_concept_fixture_role::baseline: return "baseline";
        case common_agent_concept_fixture_role::conditioned: return "conditioned";
        case common_agent_concept_fixture_role::control: return "control";
        case common_agent_concept_fixture_role::counterexample: return "counterexample";
    }
    return "unknown";
}

bool common_agent_concept_fixture_validate(
        const common_agent_concept_fixture & fixture,
        std::string & error) {
    error.clear();
    if (fixture.schema_version != 1 || !bounded(fixture.id) ||
            !bounded(fixture.grounding_ref) || !bounded(fixture.context_ref) ||
            !bounded(fixture.verifier_ref) || !bounded(fixture.semantic_expectation_ref) ||
            !bounded(fixture.independent_key)) {
        error = "semantic concept fixture is incomplete";
        return false;
    }
    return true;
}

bool common_agent_validate_concept_fixture_set(
        const common_agent_concept_grounding & grounding,
        const std::vector<common_agent_concept_fixture> & fixtures,
        std::string & error) {
    error.clear();
    if (!common_agent_concept_grounding_validate(grounding, error) || fixtures.empty()) {
        if (error.empty()) error = "semantic concept fixture set is empty or ungrounded";
        return false;
    }
    struct roles {
        bool baseline = false;
        bool conditioned = false;
        bool control = false;
        bool baseline_verified = false;
        bool conditioned_verified = false;
        bool control_verified = false;
    };
    std::map<std::string, roles> by_independent_key;
    std::vector<std::string> fixture_ids;
    for (const auto & fixture : fixtures) {
        if (!common_agent_concept_fixture_validate(fixture, error) ||
                fixture.grounding_ref != grounding.id ||
                fixture.verifier_ref != grounding.verifier_ref ||
                std::find(fixture_ids.begin(), fixture_ids.end(), fixture.id) != fixture_ids.end()) {
            if (error.empty()) error = "semantic concept fixture set contains an incompatible or duplicate fixture";
            return false;
        }
        fixture_ids.push_back(fixture.id);
        auto & group = by_independent_key[fixture.independent_key];
        if (fixture.role == common_agent_concept_fixture_role::baseline) {
            if (group.baseline) {
                error = "semantic concept fixture set has duplicate baseline roles";
                return false;
            }
            group.baseline = true;
            group.baseline_verified = fixture.host_verified;
        } else if (fixture.role == common_agent_concept_fixture_role::conditioned) {
            if (group.conditioned) {
                error = "semantic concept fixture set has duplicate conditioned roles";
                return false;
            }
            group.conditioned = true;
            group.conditioned_verified = fixture.host_verified;
        } else if (fixture.role == common_agent_concept_fixture_role::control) {
            if (group.control) {
                error = "semantic concept fixture set has duplicate control roles";
                return false;
            }
            group.control = true;
            group.control_verified = fixture.host_verified;
        }
    }
    size_t complete_groups = 0;
    for (const auto & item : by_independent_key) {
        if (item.second.baseline && item.second.conditioned && item.second.control &&
                item.second.baseline_verified && item.second.conditioned_verified &&
                item.second.control_verified) {
            ++complete_groups;
        }
    }
    if (complete_groups < grounding.minimum_contrasts) {
        error = "semantic concept fixture set lacks enough complete independent contrasts";
        return false;
    }
    return true;
}

bool common_agent_concept_contrast_validate(
        const common_agent_concept_contrast & contrast,
        const common_agent_concept_grounding & grounding,
        std::string & error) {
    error.clear();
    if (!common_agent_concept_grounding_validate(grounding, error) ||
            contrast.schema_version != 1 || !bounded(contrast.id) ||
            contrast.grounding_ref != grounding.id ||
            !bounded(contrast.behavior_key) ||
            contrast.behavior_key != grounding.behavior_key ||
            !bounded(contrast.task_fingerprint) || !bounded(contrast.baseline_ref) ||
            !bounded(contrast.conditioned_ref) || !bounded(contrast.control_ref) ||
            contrast.baseline_ref == contrast.conditioned_ref ||
            contrast.baseline_ref == contrast.control_ref ||
            contrast.conditioned_ref == contrast.control_ref ||
            !bounded(contrast.verifier_ref) || contrast.verifier_ref != grounding.verifier_ref ||
            !bounded(contrast.changed_dimension) ||
            contrast.changed_dimension != grounding.changed_dimension ||
            !bounded_items(contrast.invariant_dimensions) ||
            !bounded(contrast.independent_key) ||
            !std::isfinite(contrast.confidence) || contrast.confidence < 0.0f ||
            contrast.confidence > 1.0f || !contrast.host_verified) {
        if (error.empty()) error = "semantic concept contrast is incomplete or unverified";
        return false;
    }
    return true;
}

bool common_agent_concept_contrast_to_teaching_relation(
        const common_agent_concept_hypothesis & hypothesis,
        const common_agent_concept_grounding & grounding,
        const common_agent_concept_contrast & contrast,
        common_flydelta_teaching_relation & relation,
        std::string & error) {
    error.clear();
    if (!common_agent_concept_hypothesis_validate(hypothesis, error) ||
            hypothesis.status != common_agent_concept_hypothesis_status::grounded ||
            !common_agent_concept_grounding_validate(grounding, error) ||
            !common_agent_concept_contrast_validate(contrast, grounding, error) ||
            hypothesis.id != grounding.hypothesis_ref) {
        if (error.empty()) error = "semantic concept contrast is not admitted for teaching";
        return false;
    }
    relation = {};
    relation.id = contrast.id;
    relation.teaching_key = hypothesis.concept_key;
    relation.source = source_for(hypothesis.source_kind);
    relation.scope = grounding.scope;
    relation.behavior_key = grounding.behavior_key;
    relation.task_fingerprint = contrast.task_fingerprint;
    relation.baseline_ref = contrast.baseline_ref;
    relation.conditioned_ref = contrast.conditioned_ref;
    relation.control_ref = contrast.control_ref;
    relation.verifier_ref = contrast.verifier_ref;
    relation.evidence_ref = grounding.id;
    relation.contrast_ref = contrast.id;
    relation.status = common_flydelta_teaching_relation_status::resolved;
    relation.baseline_origin = contrast.baseline_origin;
    relation.conditioned_origin = contrast.conditioned_origin;
    relation.control_origin = contrast.control_origin;
    relation.confidence = std::min(hypothesis.confidence, contrast.confidence);
    relation.host_approved = true;
    return common_flydelta_teaching_relation_validate(relation, error);
}

bool common_agent_validate_concept_teaching_relations(
        const common_agent_concept_hypothesis & hypothesis,
        const common_agent_concept_grounding & grounding,
        const std::vector<common_flydelta_teaching_relation> & relations,
        std::string & error) {
    error.clear();
    if (!common_agent_concept_hypothesis_validate(hypothesis, error) ||
            hypothesis.status != common_agent_concept_hypothesis_status::grounded ||
            !common_agent_concept_grounding_validate(grounding, error) ||
            hypothesis.id != grounding.hypothesis_ref ||
            relations.size() < grounding.minimum_contrasts) {
        if (error.empty()) error = "semantic concept material has insufficient verified contrasts";
        return false;
    }
    std::vector<std::string> contrast_refs;
    std::vector<std::string> task_fingerprints;
    contrast_refs.reserve(relations.size());
    task_fingerprints.reserve(relations.size());
    for (const auto & relation : relations) {
        if (!common_flydelta_teaching_relation_validate(relation, error) ||
                relation.teaching_key != hypothesis.concept_key ||
                relation.behavior_key != grounding.behavior_key ||
                relation.evidence_ref != grounding.id ||
                relation.verifier_ref != grounding.verifier_ref ||
                relation.contrast_ref.empty()) {
            if (error.empty()) error = "semantic concept material contains an incompatible relation";
            return false;
        }
        if (std::find(contrast_refs.begin(), contrast_refs.end(), relation.contrast_ref) !=
                contrast_refs.end() ||
                std::find(task_fingerprints.begin(), task_fingerprints.end(), relation.task_fingerprint) !=
                task_fingerprints.end()) {
            error = "semantic concept material requires independent contrasts";
            return false;
        }
        contrast_refs.push_back(relation.contrast_ref);
        task_fingerprints.push_back(relation.task_fingerprint);
    }
    return true;
}
