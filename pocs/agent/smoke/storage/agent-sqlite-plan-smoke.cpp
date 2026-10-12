#include "plan/sqlite/plan-sqlite.h"

#include <cassert>
#include <filesystem>
#include <iostream>

int main() {
    const auto path = std::filesystem::temp_directory_path() / "llama-agent-plan-sqlite-smoke.sqlite";
    std::error_code ignored;
    std::filesystem::remove(path, ignored);

    common_plan_state plan;
    plan.id = "plan-1";
    plan.session_id = "session-1";
    plan.goal = "Persist a plan with SQLite";
    plan.purpose = "storage smoke";
    plan.scope = common_plan_scope::session;

    std::string error;
    {
        common_plan_sqlite_store store;
        assert(store.open(path.string(), error));
        assert(store.create(plan, error));

        common_plan_operation operation;
        operation.kind = common_plan_operation_kind::add_step;
        operation.plan_id = plan.id;
        operation.expected_version = 0;
        common_plan_step step;
        step.id = "step-1";
        step.title = "Persisted step";
        step.objective = "Verify plan persistence";
        step.mode = common_plan_step_mode::reasoning;
        operation.step = step;

        common_plan_state updated;
        assert(store.apply(operation, updated, error));
        assert(updated.version == 1);
        assert(updated.steps.size() == 1);
        assert(store.history(plan.id, error).size() == 1);
    }

    {
        common_plan_sqlite_store store;
        assert(store.open(path.string(), error));
        const auto restored = store.get(plan.id, error);
        assert(restored.has_value());
        assert(restored->version == 1);
        assert(restored->steps.size() == 1);
        assert(restored->steps.front().id == "step-1");
        assert(store.history(plan.id, error).size() == 1);
        assert(store.erase(plan.id, error));
        assert(!store.get(plan.id, error).has_value());
    }

    std::filesystem::remove(path, ignored);
    std::cout << "SQLite plan persistence smoke passed\n";
    return 0;
}
