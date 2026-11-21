#include "pocs/agent/smoke/support/method-phase-scenario.h"
#include <iostream>

int main() {
    try {
        const auto plan=method_phase_scenario::run();
        std::cout << "PASS: local inspection -> retained phase decision -> normal planner continuation -> "
            "external evidence -> final review; plan=" << plan.id << "\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "FAIL: " << error.what() << '\n';return 1;
    }
}
