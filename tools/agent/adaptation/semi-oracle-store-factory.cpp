#include "semi-oracle-store-factory.h"

#ifdef LLAMA_AGENT_ADAPTATION_USE_COZO
#include "semi-oracle-store-cozo.h"
#endif

#include <map>
#include <mutex>
#include <nlohmann/json.hpp>
#include <set>
#include <tuple>

namespace {
std::string evaluation_payload(const common_semi_oracle_result & result) {
    using json = nlohmann::ordered_json;
    return json{{"ruleset_id",result.ruleset_id},{"ruleset_revision",result.ruleset_revision},
        {"verdict",common_semi_oracle_verdict_name(result.verdict)},
        {"level",common_semi_oracle_level_name(result.level)},
        {"evaluator_ref",result.evaluator_ref},{"evaluator_revision",result.evaluator_revision},
        {"expansions",result.expansions},{"depth_reached",result.depth_reached},
        {"proof_json",result.proof_json},{"counterexample_json",result.counterexample_json},
        {"reason",result.reason},{"guidance_ref",result.guidance_ref},
        {"search_guidance_json",result.search_guidance_json},
        {"search_branch_quality_json",result.search_branch_quality_json},
        {"algorithm",result.applied_search_policy.algorithm==common_semi_oracle_search_algorithm::focal
            ?"focal":"exact"},{"suboptimality_bound",result.applied_search_policy.suboptimality_bound}}.dump();
}
class in_memory_semi_oracle_store final : public common_semi_oracle_store {
public:
    bool put_revision(const common_semi_oracle_ruleset & value, std::string & error) override {
        if (!common_semi_oracle_validate_ruleset(value, error)) return false;
        std::lock_guard<std::mutex> lock(mutex_);
        const auto key = std::make_pair(value.id, value.revision);
        const auto found = rulesets_.find(key);
        const auto payload = common_semi_oracle_ruleset_to_json(value);
        if (found != rulesets_.end()) {
            if (common_semi_oracle_ruleset_to_json(found->second) == payload) return true;
            error = "RuleSet revisions are immutable"; return false;
        }
        rulesets_.emplace(key, value); return true;
    }
    bool get_revision(const std::string & id, uint64_t revision,
            common_semi_oracle_ruleset & value, bool & found, std::string & error) const override {
        std::lock_guard<std::mutex> lock(mutex_); error.clear();
        const auto it = rulesets_.find({id, revision}); found = it != rulesets_.end();
        if (found) value = it->second; return true;
    }
    bool promote_to_host(const std::string & id, uint64_t revision, std::string & error) override {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto it = rulesets_.find({id, revision});
        if (it == rulesets_.end()) { error = "cannot promote missing RuleSet revision"; return false; }
        it->second.scope = common_semi_oracle_scope::host; it->second.scope_id = "host";
        host_rulesets_.insert(id); return true;
    }
    bool bind_session(const std::string & session, const std::string & id,
            uint64_t revision, std::string & error) override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!rulesets_.count({id, revision})) { error = "cannot bind session to missing RuleSet revision"; return false; }
        bindings_[session] = {id, revision}; return true;
    }
    bool get_session_binding(const std::string & session, std::string & id,
            uint64_t & revision, bool & found, std::string & error) const override {
        std::lock_guard<std::mutex> lock(mutex_); error.clear();
        const auto it = bindings_.find(session); found = it != bindings_.end();
        if (found) { id = it->second.first; revision = it->second.second; } return true;
    }
    bool get_session_game_state(const std::string & session, const std::string & id,
            uint64_t revision, std::string & state, bool & found, std::string & error) const override {
        std::lock_guard<std::mutex> lock(mutex_); error.clear();
        const auto it = states_.find({session, id, revision}); found = it != states_.end();
        if (found) state = it->second; return true;
    }
    bool put_session_game_state(const std::string & session, const std::string & id,
            uint64_t revision, const std::string & state, std::string & error) override {
        std::lock_guard<std::mutex> lock(mutex_); error.clear();
        states_[{session, id, revision}] = state; return true;
    }
    bool append_evaluation(const std::string & id, const common_semi_oracle_result & result,
            std::string & error) override {
        std::lock_guard<std::mutex> lock(mutex_); error.clear();
        if (id.empty() || result.ruleset_id.empty()) { error="semi-oracle evaluation requires ids"; return false; }
        const auto existing = evaluations_.find(id);
        if (existing != evaluations_.end()) {
            if (evaluation_payload(existing->second) == evaluation_payload(result)) return true;
            error = "SemiOracle evaluation identities are immutable"; return false;
        }
        evaluations_.emplace(id, result);
        ++search_history_revisions_[{result.ruleset_id, result.ruleset_revision}];
        return true;
    }
    bool get_evaluation_payload(const std::string & id, std::string & payload,
            bool & found, std::string & error) const override {
        std::lock_guard<std::mutex> lock(mutex_); error.clear();
        const auto it = evaluations_.find(id); found = it != evaluations_.end();
        if (found) payload = it->second.proof_json; return true;
    }
    bool list_search_history(const std::string & id, uint64_t revision, size_t limit,
            std::vector<common_semi_oracle_search_history_record> & records,
            std::string & error) const override {
        std::lock_guard<std::mutex> lock(mutex_); error.clear(); records.clear();
        if (limit == 0 || limit > 1024) { error = "invalid search history limit"; return false; }
        for (const auto & item : evaluations_) {
            if (item.second.ruleset_id == id && item.second.ruleset_revision == revision &&
                    !item.second.search_guidance_json.empty()) {
                records.push_back({item.first, item.second.search_guidance_json});
                if (records.size() == limit) break;
            }
        }
        return true;
    }
    bool get_search_history_revision(const std::string & id, uint64_t revision,
            uint64_t & watermark, bool & supported, std::string & error) const override {
        std::lock_guard<std::mutex> lock(mutex_);
        error.clear(); supported = true; watermark = 0;
        const auto it = search_history_revisions_.find({id, revision});
        if (it != search_history_revisions_.end()) watermark = it->second;
        return true;
    }
    bool put_interaction(const common_semi_oracle_interaction & value,
            std::string & error) override {
        if (!common_semi_oracle_interaction_validate(value, error)) return false;
        std::lock_guard<std::mutex> lock(mutex_);
        const auto found = interactions_.find(value.id);
        if (found != interactions_.end()) {
            if (common_semi_oracle_interaction_to_json(found->second) ==
                    common_semi_oracle_interaction_to_json(value)) { error.clear(); return true; }
            error = "SemiOracle interactions are immutable"; return false;
        }
        interactions_[value.id] = value; error.clear(); return true;
    }
    bool get_interaction(const std::string & id, common_semi_oracle_interaction & value,
            bool & found, std::string & error) const override {
        std::lock_guard<std::mutex> lock(mutex_); error.clear();
        const auto it = interactions_.find(id); found = it != interactions_.end();
        if (found) value = it->second; else value = {};
        return true;
    }
    std::vector<std::string> list_host_rulesets(std::string & error) const override {
        std::lock_guard<std::mutex> lock(mutex_); error.clear();
        return {host_rulesets_.begin(), host_rulesets_.end()};
    }
private:
    mutable std::mutex mutex_;
    std::map<std::pair<std::string, uint64_t>, common_semi_oracle_ruleset> rulesets_;
    std::map<std::string, std::pair<std::string, uint64_t>> bindings_;
    std::map<std::tuple<std::string, std::string, uint64_t>, std::string> states_;
    std::map<std::string, common_semi_oracle_result> evaluations_;
    std::map<std::pair<std::string, uint64_t>, uint64_t> search_history_revisions_;
    std::map<std::string, common_semi_oracle_interaction> interactions_;
    std::set<std::string> host_rulesets_;
};
}

std::unique_ptr<common_semi_oracle_store> make_agent_semi_oracle_store(
        const std::string & requested, const std::string & path, std::string & error) {
    error.clear();
    const auto backend = requested.empty() ? "auto" : requested;
    if (backend == "auto") {
        if (path.empty()) return std::make_unique<in_memory_semi_oracle_store>();
#ifdef LLAMA_AGENT_ADAPTATION_USE_COZO
        auto store = std::make_unique<agent_semi_oracle_cozo_store>();
        if (!store->open(path, error)) return nullptr;
        return store;
#else
        error = "persistent SemiOracle store requires a build with Cozo"; return nullptr;
#endif
    }
    if (backend == "in_memory") {
        if (!path.empty()) { error = "in-memory SemiOracle store does not accept a path"; return nullptr; }
        return std::make_unique<in_memory_semi_oracle_store>();
    }
    if (backend == "cozo") {
#ifdef LLAMA_AGENT_ADAPTATION_USE_COZO
        auto store = std::make_unique<agent_semi_oracle_cozo_store>();
        if (!store->open(path, error)) return nullptr;
        return store;
#else
        error = "SemiOracle Cozo store requires a build with Cozo"; return nullptr;
#endif
    }
    error = "SemiOracle store currently supports auto, in_memory or cozo";
    return nullptr;
}
