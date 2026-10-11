#include "agent/adaptation/semi-oracle-teaching.h"

#define CHECK(x) do { if (!(x)) return __LINE__; } while (false)

int main() {
    std::string error;
    common_semi_oracle_ruleset ruleset;
    CHECK(common_semi_oracle_parse_ruleset_dsl(
        "name: Three in a row\ndescription: test\ndomain: placement_game\nrows: 3\ncolumns: 3\nwin_length: 3\nplayers: X, O\n",
        "ruleset://host/test", "session-test", ruleset, error,
        common_semi_oracle_rule_authority::declared));
    common_semi_oracle_result rejected, accepted;
    for (auto * result : {&rejected, &accepted}) {
        result->ruleset_id = ruleset.id;
        result->ruleset_revision = ruleset.revision;
        result->evaluator_ref = "semi-oracle://evaluator/placement_game";
        result->evaluator_revision = "v1";
    }
    rejected.verdict = common_semi_oracle_verdict::violated;
    rejected.counterexample_json = "{}";
    accepted.verdict = common_semi_oracle_verdict::satisfied;
    accepted.proof_json = "{}";

    common_flydelta_teaching_relation relation;
    common_agent_scope scope;
    CHECK(common_semi_oracle_make_teaching_relation(ruleset, rejected, accepted,
        scope, "relation-1", "run://bad", "run://good", "evidence://pair",
        relation, error));
    CHECK(relation.source == common_adaptation_evidence_source::user_taught_concept);
    CHECK(relation.host_approved && relation.oracle_revision.find(ruleset.semantic_fingerprint) != std::string::npos);
    accepted.ruleset_revision++;
    CHECK(!common_semi_oracle_make_teaching_relation(ruleset, rejected, accepted,
        scope, "relation-2", "run://bad", "run://good", "evidence://pair",
        relation, error));
    return 0;
}
