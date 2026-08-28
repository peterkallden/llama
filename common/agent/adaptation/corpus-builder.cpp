#include "agent/adaptation/corpus-builder.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <iomanip>
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

bool common_learning_build_corpus(
        const std::vector<common_training_candidate> & candidates,
        const common_learning_corpus_policy & policy,
        common_learning_corpus_revision & revision,
        std::string & error) {
    error.clear();
    if (policy.builder_version.empty()) { error = "corpus builder requires version"; return false; }
    if (candidates.size() > policy.max_candidates) { error = "corpus exceeds candidate bound"; return false; }
    std::vector<const common_training_candidate *> selected;
    common_training_candidate_policy candidate_policy;
    for (const auto & candidate : candidates) {
        if (candidate.status != common_training_candidate_status::approved) {
            error = "corpus contains a candidate not approved for corpus";
            return false;
        }
        if (!common_training_candidate_qualifies(candidate, candidate_policy, error)) return false;
        selected.push_back(&candidate);
    }
    std::sort(selected.begin(), selected.end(), [](const auto * left, const auto * right) { return left->id < right->id; });
    revision = {};
    revision.builder_version = policy.builder_version;
    revision.seed = policy.seed;
    std::ostringstream jsonl;
    for (const auto * candidate : selected) {
        const auto line = json{
            {"candidate_id", candidate->id},
            {"split", split_for(candidate->id, policy.seed)},
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
        {"jsonl_hash", hash_text(revision.jsonl)},
    }.dump();
    revision.bundle_hash = hash_text(revision.manifest_json + "\n" + revision.jsonl);
    revision.id = "learning://corpus/" + revision.bundle_hash;
    return true;
}
