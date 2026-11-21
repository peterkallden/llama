#include "agent-daemon-flydelta-internal.h"

#include <memory>
#include <string>

#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (false)

int main() {
    using namespace agent_daemon_flydelta_internal;

    auto source = std::make_shared<daemon_flydelta_resource_provider>();
    common_flydelta_oracle_evaluator_registration registration;
    registration.evaluator_ref = "test://oracle/placement";
    registration.evaluator_revision = "test-v1";
    registration.semantic_kind = "placement_game";
    source->oracle_registry.evaluators.push_back(registration);

    common_agent_scope scope;
    scope.namespace_id = "test-namespace";
    scope.project_id = "test-project";
    scope.session_id = "test-session";
    scope.turn_id = "test-turn";

    const auto scoped = daemon_flydelta_provider_for_scope(source, scope);
    CHECK(scoped != nullptr);
    CHECK(scoped.get() != source.get());
    CHECK(scoped->oracle_registry.evaluators.size() == 1);
    CHECK(scoped->oracle_registry.evaluators.front().evaluator_ref ==
        "test://oracle/placement");
    CHECK(scoped->oracle_registry.evaluators.front().evaluator_revision == "test-v1");
    CHECK(scoped->authority.namespace_id == "test-namespace");
    CHECK(scoped->authority.project_id == "test-project");
    CHECK(scoped->authority.session_id == "test-session");
    CHECK(scoped->authority.turn_id == "test-turn");
    return 0;
}
