#include "tools/agent/tooling/agent-sandbox-helper.h"

#include <cstdio>

int main() {
    common_agent_sandbox_local_test_runtime runtime;
    common_agent_sandbox_policy policy;
    policy.execution_class = "readonly-analysis";
    policy.limits.timeout_ms = 60000;
    policy.limits.cpu_count = 1;
    policy.limits.max_output_bytes = 65536;
    policy.network = common_agent_sandbox_network_scope::none;
    policy.filesystem = common_agent_sandbox_filesystem_scope::readonly;

    common_agent_sandbox_tool_helper helper(runtime, policy);
    common_agent_sandbox_request request;
    request.operation_id = "sandbox-smoke-1";
    request.execution_class = "readonly-analysis";
    request.command.program = "test-program";
    request.limits.timeout_ms = 1000;
    request.limits.cpu_count = 1;
    request.limits.max_output_bytes = 4096;

    const auto result = helper.run(request);
    if (!result.ok || result.output.find("local-test/sandbox-smoke-1") == std::string::npos ||
            runtime.last_request.command.program != "test-program") {
        std::fprintf(stderr, "sandbox local helper did not return the expected result\n");
        return 1;
    }

    request.network = common_agent_sandbox_network_scope::research_web;
    const auto rejected = helper.run(request);
    if (rejected.ok || rejected.failure_class != common_tool_failure_class::policy ||
            rejected.failure_code != "sandbox.policy_rejected") {
        std::fprintf(stderr, "sandbox policy did not reject an excessive network scope\n");
        return 1;
    }

    common_agent_sandbox_unavailable_runtime unavailable_runtime;
    common_agent_sandbox_tool_helper unavailable_helper(unavailable_runtime, policy);
    request.network = common_agent_sandbox_network_scope::none;
    const auto unavailable = unavailable_helper.run(request);
    if (unavailable.ok || unavailable.failure_code != "sandbox.backend_unavailable" ||
            unavailable.failure_class != common_tool_failure_class::execution) {
        std::fprintf(stderr, "no-backend runtime did not return the expected result\n");
        return 1;
    }

    std::printf("sandbox_status=ok\n");
    std::printf("sandbox_backend=%s\n", runtime.last_request.command.program.c_str());
    return 0;
}
