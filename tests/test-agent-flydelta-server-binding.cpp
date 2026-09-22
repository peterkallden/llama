#include "tools/agent/runtime/agent-server-context-host.h"
#include "agent/adaptation/flydelta/flydelta-evaluator.h"

#include <string>
#include <utility>

#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (false)

int main() {
    common_agent_server_flydelta_binding_callbacks callbacks;
    callbacks.primitives.capture = true;
    callbacks.primitives.generation = true;
    callbacks.primitives.host_verification = true;
    callbacks.prepare_arm = [](
            const common_flydelta_arm_request &,
            common_agent_generation_request &,
            std::string &) { return true; };
    callbacks.finalize_arm = [](
            const common_flydelta_arm_request &,
            const common_agent_generation_result &,
            common_flydelta_arm_result &,
            std::string &) { return true; };
    callbacks.register_evaluator = [](
            common_flydelta_evaluator_config &,
            common_flydelta_evaluator_callbacks &,
            std::string &) { return true; };

    auto binding = common_agent_server_flydelta_binding_from_callbacks(
        std::move(callbacks));
    CHECK(binding.primitives.capture);
    CHECK(binding.primitives.generation);
    CHECK(binding.primitives.host_verification);
    CHECK(static_cast<bool>(binding.prepare_arm));
    CHECK(static_cast<bool>(binding.finalize_arm));
    CHECK(static_cast<bool>(binding.register_evaluator));

    common_flydelta_arm_request arm;
    common_agent_generation_request request;
    std::string error;
    CHECK(binding.prepare_arm(arm, request, error));
    common_flydelta_arm_result result;
    common_agent_generation_result generation;
    CHECK(binding.finalize_arm(arm, generation, result, error));

    common_flydelta_evaluator_config evaluator_config;
    common_flydelta_evaluator_callbacks evaluator_callbacks;
    CHECK(binding.register_evaluator(
        evaluator_config, evaluator_callbacks, error));

    auto factory = common_agent_server_flydelta_binding_factory_from_callbacks({
        binding.primitives,
        {},
        binding.prepare_arm,
        binding.finalize_arm,
        binding.register_evaluator,
        {},
        {},
    });
    common_agent_server_flydelta_binding produced;
    auto resident_host = std::make_shared<common_agent_server_context_host>();
    CHECK(factory(resident_host, produced, error));
    CHECK(error.empty());
    CHECK(static_cast<bool>(produced.prepare_arm));

    auto incomplete_factory =
        common_agent_server_flydelta_binding_factory_from_callbacks({});
    CHECK(!incomplete_factory(resident_host, produced, error));
    CHECK(error.find("requires prepare, finalize and evaluator callbacks") !=
        std::string::npos);

    return 0;
}
