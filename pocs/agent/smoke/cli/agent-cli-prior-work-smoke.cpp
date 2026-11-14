#include "../support/prior-work-scenario.h"
#include <iostream>

int main() {
    try {
        common_plan_in_memory_store store;
        std::string error;
        if (!store.open("", error)) throw std::runtime_error(error);
        prior_work_scenario::run(store, true);
        std::cout << "prior-work smoke passed; model-free; tracing enabled\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
