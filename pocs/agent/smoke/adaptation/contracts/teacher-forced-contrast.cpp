#include "agent/agent-inference.h"

#include <iostream>
#include <string>

namespace {

int fail(const std::string & message) {
    std::cerr << "flydelta teacher-forced contrast contract failed: " << message << '\n';
    return 1;
}

} // namespace

int main() {
    std::string error;
    if (!common_agent_teacher_forced_contrast_is_valid(
            "{\"name\":\"", "dataset.inspect", "statistics.describe", error)) {
        return fail("distinct tool continuations were rejected: " + error);
    }
    if (common_agent_teacher_forced_contrast_is_valid(
            "{\"name\":\"", "same-tool-call", "same-tool-call", error)) {
        return fail("identical continuations were accepted");
    }
    if (error.find("distinct") == std::string::npos) {
        return fail("identical continuations did not report a distinctness error");
    }
    if (common_agent_teacher_forced_contrast_is_valid(
            "{\"name\":\"", "", "statistics.describe", error)) {
        return fail("an empty continuation was accepted");
    }
    std::cout << "flydelta_teacher_forced_contrast_contract=passed\n";
    return 0;
}
