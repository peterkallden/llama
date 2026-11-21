#include "agent/adaptation/flydelta/flydelta-worker-budget.h"

#include <iostream>
#include <string>

#define CHECK(condition) do { \
    if (!(condition)) { \
        std::cerr << "check failed at line " << __LINE__ << "\n"; \
        return 1; \
    } \
} while (false)

int main() {
    std::string error;
    common_flydelta_worker_budget budget;

    CHECK(common_flydelta_worker_budget_compute(4, false, 0, budget, error));
    CHECK(budget.agent_workers == 4);
    CHECK(budget.flydelta_workers == 0);

    CHECK(common_flydelta_worker_budget_compute(4, true, 1, budget, error));
    CHECK(budget.agent_workers == 3);
    CHECK(budget.flydelta_workers == 1);

    CHECK(common_flydelta_worker_budget_compute(2, true, 2, budget, error));
    CHECK(budget.agent_workers == 0);
    CHECK(budget.flydelta_workers == 2);

    CHECK(!common_flydelta_worker_budget_compute(4, false, 1, budget, error));
    CHECK(!error.empty());
    CHECK(!common_flydelta_worker_budget_compute(4, true, 0, budget, error));
    CHECK(!common_flydelta_worker_budget_compute(4, true, 5, budget, error));
    CHECK(!common_flydelta_worker_budget_compute(0, false, 0, budget, error));

    std::cout << "FlyDelta worker budget contract passed\n";
    return 0;
}
