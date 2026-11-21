#pragma once

#include "agent/adaptation/semi-oracle.h"
#include "agent/adaptation/semi-oracle-teaching.h"
#include "agent/prediction/outcome-guidance.h"

#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <atomic>

struct common_semi_oracle_history_projection_cache_entry {
    std::string key;
    common_outcome_guidance_projection suffix_projection;
    common_outcome_guidance_projection branch_projection;
    size_t bytes = 0;
    size_t read_rows = 0;
    size_t rejected_rows = 0;
    size_t duplicate_rows = 0;
    size_t conflicting_rows = 0;
    std::vector<std::string> evaluation_refs;
};

struct common_semi_oracle_search_history_record {
    std::string evaluation_id;
    std::string guidance_json;
};

class common_semi_oracle_store {
public:
    common_semi_oracle_store() {
        static std::atomic<uint64_t> next_identity{1};
        cache_identity_ = next_identity.fetch_add(1, std::memory_order_relaxed);
    }
    virtual ~common_semi_oracle_store() = default;
    common_semi_oracle_store(const common_semi_oracle_store &) = delete;
    common_semi_oracle_store & operator=(const common_semi_oracle_store &) = delete;
    uint64_t cache_identity() const { return cache_identity_; }
    // A store object can be reopened against another backing database. Never
    // let an equal watermark in that database reuse the previous projection.
    void clear_history_projection_cache() {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        projection_cache_index_.clear();
        projection_cache_lru_.clear();
        projection_cache_bytes_ = 0;
    }
    void set_history_projection_cache_budget(size_t bytes) {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        projection_cache_budget_ = bytes;
        while (!projection_cache_lru_.empty() && projection_cache_bytes_ > bytes) {
            projection_cache_bytes_ -= projection_cache_lru_.back()->bytes;
            projection_cache_index_.erase(projection_cache_lru_.back()->key);
            projection_cache_lru_.pop_back();
        }
    }
    bool get_history_projection_cache(const std::string & key,
            std::shared_ptr<const common_semi_oracle_history_projection_cache_entry> & entry) const {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        const auto found = projection_cache_index_.find(key);
        if (found == projection_cache_index_.end()) { entry.reset(); return false; }
        entry = *found->second;
        projection_cache_lru_.splice(projection_cache_lru_.begin(), projection_cache_lru_, found->second);
        return true;
    }
    void put_history_projection_cache(common_semi_oracle_history_projection_cache_entry entry) {
        constexpr size_t entry_limit = 16;
        std::lock_guard<std::mutex> lock(cache_mutex_);
        const size_t byte_limit = projection_cache_budget_;
        if (byte_limit == 0 || entry.bytes > byte_limit || entry.key.empty()) return;
        if (projection_cache_index_.count(entry.key)) return;
        auto value = std::make_shared<const common_semi_oracle_history_projection_cache_entry>(
            std::move(entry));
        projection_cache_lru_.push_front(value);
        try {
            projection_cache_index_.emplace(value->key, projection_cache_lru_.begin());
        } catch (...) {
            projection_cache_lru_.pop_front();
            throw;
        }
        projection_cache_bytes_ += value->bytes;
        while (projection_cache_lru_.size() > entry_limit || projection_cache_bytes_ > byte_limit) {
            projection_cache_bytes_ -= projection_cache_lru_.back()->bytes;
            projection_cache_index_.erase(projection_cache_lru_.back()->key);
            projection_cache_lru_.pop_back();
        }
    }
    virtual bool put_revision(const common_semi_oracle_ruleset & ruleset, std::string & error) = 0;
    virtual bool get_revision(const std::string & id, uint64_t revision,
            common_semi_oracle_ruleset & ruleset, bool & found, std::string & error) const = 0;
    virtual bool promote_to_host(const std::string & id, uint64_t revision, std::string & error) = 0;
    virtual bool bind_session(const std::string & session_id, const std::string & id,
            uint64_t revision, std::string & error) = 0;
    virtual bool get_session_binding(const std::string & session_id, std::string & id,
            uint64_t & revision, bool & found, std::string & error) const = 0;
    // Domain state is host-owned and keyed by session and RuleSet revision. A
    // runtime must not fall back to model-supplied state. The compatibility
    // hooks below retain the original placement-game store implementations
    // while new runtime code uses the domain-neutral names.
    virtual bool get_session_state(const std::string & session_id,
            const std::string & ruleset_id, uint64_t revision,
            std::string & state_json, bool & found, std::string & error) const {
        return get_session_game_state(session_id, ruleset_id, revision,
            state_json, found, error);
    }
    virtual bool put_session_state(const std::string & session_id,
            const std::string & ruleset_id, uint64_t revision,
            const std::string & state_json, std::string & error) {
        return put_session_game_state(session_id, ruleset_id, revision,
            state_json, error);
    }
    // Legacy storage hook. Implementations may retain this physical relation
    // name; callers must use get_session_state/put_session_state.
    virtual bool get_session_game_state(const std::string & session_id,
            const std::string & ruleset_id, uint64_t revision,
            std::string & state_json, bool & found, std::string & error) const {
        (void) session_id; (void) ruleset_id; (void) revision;
        state_json.clear(); found = false;
        error = "semi-oracle game-state storage is unsupported by this store";
        return false;
    }
    virtual bool put_session_game_state(const std::string & session_id,
            const std::string & ruleset_id, uint64_t revision,
            const std::string & state_json, std::string & error) {
        (void) session_id; (void) ruleset_id; (void) revision; (void) state_json;
        error = "semi-oracle game-state storage is unsupported by this store";
        return false;
    }
    virtual bool append_evaluation(const std::string & evaluation_id,
            const common_semi_oracle_result & result, std::string & error) = 0;
    virtual bool get_evaluation_payload(const std::string & evaluation_id,
            std::string & payload_json, bool & found, std::string & error) const {
        (void) evaluation_id;
        payload_json.clear();
        found = false;
        error = "semi-oracle evaluation reads are unsupported by this store";
        return false;
    }
    // Bounded deterministic view of the existing evaluation relation. Order
    // is by evaluation ID, not claimed to be chronology or recency.
    virtual bool list_search_history(const std::string & ruleset_id, uint64_t revision,
            size_t limit, std::vector<common_semi_oracle_search_history_record> & records,
            std::string & error) const {
        (void) ruleset_id; (void) revision; (void) limit;
        records.clear(); error = "semi-oracle search history is unsupported by this store";
        return false;
    }
    // Monotonic identity for the append-only evaluation snapshot. Stores that
    // cannot provide one leave supported=false; callers must then read and
    // fingerprint rows before reusing a cached projection.
    virtual bool get_search_history_revision(const std::string & ruleset_id,
            uint64_t revision, uint64_t & watermark, bool & supported,
            std::string & error) const {
        (void) ruleset_id; (void) revision;
        watermark = 0; supported = false; error.clear();
        return true;
    }
    // A bounded history and its watermark must describe the same snapshot.
    // Legacy stores without revision support still supply a fingerprinted view.
    virtual bool read_search_history_snapshot(const std::string & ruleset_id,
            uint64_t revision, size_t limit,
            std::vector<common_semi_oracle_search_history_record> & records,
            uint64_t & watermark, bool & supported, std::string & error) const {
        for (size_t attempt = 0; attempt < 3; ++attempt) {
            uint64_t before = 0;
            if (!get_search_history_revision(ruleset_id, revision, before, supported, error) ||
                    !list_search_history(ruleset_id, revision, limit, records, error)) return false;
            if (!supported) { watermark = 0; return true; }
            bool after_supported = false;
            if (!get_search_history_revision(ruleset_id, revision, watermark, after_supported, error)) return false;
            if (after_supported && before == watermark) return true;
        }
        records.clear(); error = "search history changed while reading snapshot";
        return false;
    }
    virtual bool put_interaction(const common_semi_oracle_interaction & interaction,
            std::string & error) {
        (void) interaction;
        error = "semi-oracle interaction storage is unsupported by this store";
        return false;
    }
    virtual bool get_interaction(const std::string & interaction_id,
            common_semi_oracle_interaction & interaction, bool & found,
            std::string & error) const {
        (void) interaction_id;
        interaction = {}; found = false;
        error = "semi-oracle interaction reads are unsupported by this store";
        return false;
    }
    virtual std::vector<std::string> list_host_rulesets(std::string & error) const = 0;

private:
    uint64_t cache_identity_ = 0;
    mutable std::mutex cache_mutex_;
    mutable std::list<std::shared_ptr<const common_semi_oracle_history_projection_cache_entry>> projection_cache_lru_;
    mutable std::map<std::string, decltype(projection_cache_lru_)::iterator> projection_cache_index_;
    mutable size_t projection_cache_bytes_ = 0;
    size_t projection_cache_budget_ = 16 * 1024 * 1024;
};
