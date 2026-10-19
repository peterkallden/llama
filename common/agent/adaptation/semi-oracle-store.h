#pragma once

#include "agent/adaptation/semi-oracle.h"
#include "agent/adaptation/semi-oracle-teaching.h"

#include <string>
#include <vector>

struct common_semi_oracle_search_history_record {
    std::string evaluation_id;
    std::string guidance_json;
};

class common_semi_oracle_store {
public:
    virtual ~common_semi_oracle_store() = default;
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
};
