#include "agent/adaptation/corpus-builder.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>

using json = nlohmann::ordered_json;

static uint64_t fnv1a(const std::string & text) {
    uint64_t hash = 1469598103934665603ULL;
    for (const unsigned char byte : text) { hash ^= byte; hash *= 1099511628211ULL; }
    return hash;
}

static std::string hash_text(const std::string & text) {
    std::ostringstream out;
    out << "identity:fnv1a64:" << std::hex << std::setw(16) << std::setfill('0') << fnv1a(text);
    return out.str();
}

static std::string split_for(const std::string & id, uint64_t seed) {
    const auto bucket = fnv1a(id + ":" + std::to_string(seed)) % 100;
    if (bucket < 80) return "train";
    if (bucket < 90) return "validation";
    return "test";
}

static std::vector<std::string> sorted_transaction_ids(const common_training_candidate & candidate) {
    auto transaction_ids = candidate.transaction_ids;
    std::sort(transaction_ids.begin(), transaction_ids.end());
    return transaction_ids;
}

static std::string candidate_fingerprint(const common_training_candidate & candidate) {
    std::ostringstream value;
    value << candidate.id << '\n';
    for (const auto & transaction_id : sorted_transaction_ids(candidate)) value << transaction_id << '\n';
    value << static_cast<int>(candidate.cause) << '\n'
          << candidate.hypothesis << '\n'
          << candidate.approved_prompt << '\n'
          << candidate.approved_target << '\n'
          << candidate.observed_occurrences << '\n'
          << candidate.verified_recoveries << '\n'
          << candidate.contradictions << '\n'
          << candidate.confidence << '\n'
          << candidate.redaction_policy_id << '\n'
          << candidate.redaction_method << '\n'
          << static_cast<int>(candidate.redaction_status) << '\n'
          << static_cast<int>(candidate.status) << '\n'
          << candidate.learning_domain << '\n'
          << candidate.tool_family << '\n'
          << candidate.provider_kind;
    return value.str();
}

static bool contains(const std::set<std::string> & values, const std::string & value) {
    return values.find(value) != values.end();
}

static json view_json(const common_learning_corpus_view & view) {
    return {
        {"learning_domain", view.learning_domain},
        {"tool_family", view.tool_family},
        {"provider_kinds", std::vector<std::string>(view.provider_kinds.begin(), view.provider_kinds.end())},
    };
}

bool common_learning_corpus_view_matches(
        const common_training_candidate & candidate,
        const common_learning_corpus_view & view) {
    if (!view.learning_domain.empty() && candidate.learning_domain != view.learning_domain) return false;
    if (!view.tool_family.empty() && candidate.tool_family != view.tool_family) return false;
    if (!view.provider_kinds.empty() && !contains(view.provider_kinds, candidate.provider_kind)) return false;
    return true;
}

bool common_learning_build_corpus(
        const std::vector<common_training_candidate> & candidates,
        const common_learning_corpus_policy & policy,
        common_learning_corpus_revision & revision,
        std::string & error) {
    error.clear();
    if (policy.builder_version.empty()) { error = "corpus builder requires version"; return false; }
    if (policy.validation_percent + policy.test_percent > 100) {
        error = "corpus split percentages exceed 100";
        return false;
    }
    if (policy.redaction_policy_id.empty()) {
        error = "corpus requires a redaction policy";
        return false;
    }
    if (policy.replay_candidate_ids.size() > policy.max_replay_candidates) {
        error = "corpus exceeds replay candidate bound";
        return false;
    }
    std::vector<const common_training_candidate *> selected;
    std::set<std::string> candidate_ids;
    std::set<std::string> transaction_ids;
    const std::set<std::string> held_out(policy.held_out_candidate_ids.begin(), policy.held_out_candidate_ids.end());
    const std::set<std::string> revoked(policy.revoked_candidate_ids.begin(), policy.revoked_candidate_ids.end());
    const std::set<std::string> replay(policy.replay_candidate_ids.begin(), policy.replay_candidate_ids.end());
    if (held_out.size() != policy.held_out_candidate_ids.size() ||
            revoked.size() != policy.revoked_candidate_ids.size() ||
            replay.size() != policy.replay_candidate_ids.size()) {
        error = "corpus policy contains duplicate candidate ids";
        return false;
    }
    for (const auto & id : replay) {
        if (contains(held_out, id)) {
            error = "held-out candidate cannot be selected for replay";
            return false;
        }
    }
    common_training_candidate_policy candidate_policy;
    std::map<std::string, std::string> fingerprints;
    for (const auto & candidate : candidates) {
        if (!common_learning_corpus_view_matches(candidate, policy.view)) continue;
        if (contains(revoked, candidate.id) || candidate.status == common_training_candidate_status::revoked) {
            error = "corpus contains a revoked candidate";
            return false;
        }
        if (contains(held_out, candidate.id)) continue;
        if (candidate.status != common_training_candidate_status::approved) {
            error = "corpus contains a candidate not approved for corpus";
            return false;
        }
        if (!common_training_candidate_qualifies(candidate, candidate_policy, error)) return false;
        if (candidate.redaction_policy_id != policy.redaction_policy_id ||
                candidate.redaction_method.empty() ||
                (candidate.redaction_status != common_learning_redaction_status::caller_asserted &&
                 candidate.redaction_status != common_learning_redaction_status::policy_checked)) {
            error = "corpus candidate lacks an accepted redaction attestation";
            return false;
        }
        const auto fingerprint = candidate_fingerprint(candidate);
        const auto existing = fingerprints.find(candidate.id);
        if (existing != fingerprints.end()) {
            if (existing->second != fingerprint) {
                error = "corpus contains conflicting duplicate candidate id";
                return false;
            }
            continue;
        }
        fingerprints.emplace(candidate.id, fingerprint);
        if (!candidate_ids.insert(candidate.id).second) return false;
        for (const auto & transaction_id : sorted_transaction_ids(candidate)) {
            if (!transaction_ids.insert(transaction_id).second) {
                error = "corpus contains overlapping candidate transactions";
                return false;
            }
        }
        selected.push_back(&candidate);
    }
    if (selected.size() > policy.max_candidates) { error = "corpus exceeds candidate bound"; return false; }
    for (const auto & id : replay) {
        if (!contains(candidate_ids, id)) {
            error = "replay candidate is not present in corpus candidates";
            return false;
        }
    }
    std::sort(selected.begin(), selected.end(), [](const auto * left, const auto * right) { return left->id < right->id; });
    revision = {};
    revision.builder_version = policy.builder_version;
    revision.seed = policy.seed;
    revision.held_out_candidate_ids.assign(held_out.begin(), held_out.end());
    revision.replay_candidate_ids.assign(replay.begin(), replay.end());
    revision.view = policy.view;
    std::map<std::string, size_t> split_counts;
    std::ostringstream jsonl;
    for (const auto * candidate : selected) {
        const auto split = split_for(candidate->id, policy.seed);
        const auto candidate_transaction_ids = sorted_transaction_ids(*candidate);
        ++split_counts[split];
        const auto line = json{
            {"candidate_id", candidate->id},
            {"split", split},
            {"replay", contains(replay, candidate->id)},
            {"transaction_ids", candidate_transaction_ids},
            {"redaction", {{"policy_id", candidate->redaction_policy_id},
                            {"method", candidate->redaction_method},
                            {"status", common_learning_redaction_status_name(candidate->redaction_status)}}},
            {"learning_domain", candidate->learning_domain},
            {"tool_family", candidate->tool_family},
            {"provider_kind", candidate->provider_kind},
            {"input", candidate->approved_prompt},
            {"target", candidate->approved_target},
        }.dump();
        jsonl << line << '\n';
        revision.candidate_ids.push_back(candidate->id);
    }
    revision.jsonl = jsonl.str();
    if (revision.jsonl.size() > policy.max_bytes) { error = "corpus exceeds byte bound"; return false; }
    const auto candidate_array = revision.candidate_ids;
    revision.manifest_json = json{
        {"schema_version", revision.schema_version},
        {"builder_version", revision.builder_version},
        {"seed", revision.seed},
        {"candidate_ids", candidate_array},
        {"replay_candidate_ids", revision.replay_candidate_ids},
        {"held_out_candidate_ids", revision.held_out_candidate_ids},
        {"view", view_json(revision.view)},
        {"transaction_ids", std::vector<std::string>(transaction_ids.begin(), transaction_ids.end())},
        {"split_counts", split_counts},
        {"redaction_policy_id", policy.redaction_policy_id},
        {"jsonl_hash", hash_text(revision.jsonl)},
    }.dump();
    revision.bundle_hash = hash_text(revision.manifest_json + "\n" + revision.jsonl);
    revision.id = "learning://corpus/" + revision.bundle_hash;
    return true;
}
